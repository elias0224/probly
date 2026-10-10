"""Calibration of the probability sources on CIFAR-10, raw and temperature scaled, clean and under corruption.

Same seed x split protocol as ``evaluate_shift.py``: ``n_splits`` permutations of the 10k clean test set, the first half
selects (here: fits the temperature by NLL), the second half is the test half. Shifted sets use the same test-half
indices, and are evaluated with the temperature fitted on the clean selection half. Per source (one model each: base
softmax, dropout softmax, MC mean, the ensemble of the base models, and the post-training methods) the script reports
ECE, NLL and Brier, and how temperature scaling changes the selective ranking (AURC, E-AURC) of three scores.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import warnings

from evaluate import setup_fonts
from evaluate_shift import CRITERIA, SHIFT_DATASETS, Tables, mean_of, save, style_axes
from fixed_threshold import ece, fit_temperature, nll, scale
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.metrics.selective_prediction import aurc  # noqa: E402
from sgr_experiment.calibration import brier_score, ranking_scores  # noqa: E402
from sgr_experiment.metrics import e_aurc  # noqa: E402
from sgr_experiment.shift import CORRUPTIONS, SEVERITIES, corrupted_name  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

METHODS = ["finetune", "swag", "laplace", "gda", "ddu", "vbll", "sngp", "sngp_long", "sngp_scratch", "dropout_scratch"]
SOURCES = ["sr_base", "sr_dropout", "mc", "ens", *METHODS, "swa"]
SOURCE_STYLE = {
    "sr_base": (CRITERIA["sr_base"]["label"], "sr_base", "o"),
    "sr_dropout": (CRITERIA["sr_dropout"]["label"], "sr_dropout", "s"),
    "mc": ("MC dropout (mean)", "mc_maxprob", "^"),
    "ens": ("Deep ensemble (mean)", "ens_maxprob", "v"),
    "swa": ("SWA mean", "swa_maxprob", "D"),
    **{m: (CRITERIA[f"{m}_maxprob"]["label"].split(",")[0] if f"{m}_maxprob" in CRITERIA else m, f"{m}_maxprob", ".") for m in METHODS},
}
# the plot shows only these sources, styled like the shift plot of the report
PLOT_SOURCES = {
    "sr_base": ("#16a085", 1.5),
    "ens": ("#1e3a8a", 2.4),
    "mc": ("#1e88e5", 1.5),
}
RANK_SCORES = ["msr", "tu_log", "tu_brier"]
VARIANTS = ["raw", "ts"]
EPS = 1e-30
CLEAN = "cifar10"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "calibration")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    return p.parse_args()


def load_dataset(runs: Path, seeds: list[int], name: str) -> dict | None:
    """Labels and the probabilities (float32) per source as a list of units (seeds, or the one ensemble).

    Returns None if a seed has no file for the dataset. A method is included only if its file exists for every seed.
    """
    files = [runs / f"seed{s}" / "shift" / f"{name}.npz" for s in seeds]
    if not all(f.exists() for f in files):
        return None
    units: dict[str, list[np.ndarray]] = {"sr_base": [], "sr_dropout": [], "mc": [], "ens": []}
    base = []
    labels = None
    for f in files:
        with np.load(f) as d:
            labels = d["labels"] if labels is None else labels
            np.testing.assert_array_equal(d["labels"], labels)
            units["sr_base"].append(d["softmax_base"].astype(np.float32))
            units["sr_dropout"].append(d["softmax_dropout"].astype(np.float32))
            units["mc"].append(d["mean_probs"].astype(np.float32))
            base.append(units["sr_base"][-1])
    units["ens"].append(np.mean(np.stack(base).astype(np.float64), axis=0).astype(np.float32))
    for m in METHODS:
        mfiles = [runs / f"seed{s}" / "shift" / m / f"{name}.npz" for s in seeds]
        if not all(f.exists() for f in mfiles):
            continue
        for f in mfiles:
            with np.load(f) as d:
                np.testing.assert_array_equal(d["labels"], labels)
                units.setdefault(m, []).append(d["mean_probs"].astype(np.float32))
                if m == "swag" and "softmax_swa" in d:
                    units.setdefault("swa", []).append(d["softmax_swa"].astype(np.float32))
    return {"labels": labels, "units": units}


def calibration_metrics(p: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """ECE (15 bins), NLL, Brier and accuracy of probabilities ``p`` (float64)."""
    nll_val = float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], EPS, None))))
    return {
        "ece": ece(p, labels),
        "nll": nll_val,
        "brier": brier_score(p, labels),
        "acc": float((p.argmax(1) == labels).mean()),
    }


def run_protocol(data: dict[str, dict], sources: list[str], n_splits: int, split_seed: int) -> dict[tuple, list[float]]:
    """Evaluate every source, unit and split; returns the list of values per result key."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    labels = data[CLEAN]["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(split_seed)
    splits = [rng.permutation(n) for _ in range(n_splits)]
    for src in sources:
        for u, p_all in enumerate(data[CLEAN]["units"][src]):
            p_all = p_all.astype(np.float64)
            logp = np.log(np.clip(p_all, EPS, None))
            for perm in splits:
                sel, test = perm[:half], perm[half:]
                t = fit_temperature(logp[sel], labels[sel])
                acc["clean", "temperature", "ts", src].append(t)
                for name in sorted(data):
                    if src not in data[name]["units"]:
                        continue
                    pd = data[name]["units"][src][u].astype(np.float64)[test]
                    variants = {"raw": pd, "ts": scale(np.log(np.clip(pd, EPS, None)), t)}
                    for v, pv in variants.items():
                        for k, val in calibration_metrics(pv, labels[test]).items():
                            acc["shift", name, k, v, src].append(val)
                            if name == CLEAN:
                                acc["clean", k, v, src].append(val)
                        if name == CLEAN:
                            loss = (pv.argmax(1) != labels[test]).astype(np.float64)
                            for s, crit in ranking_scores(pv).items():
                                acc["rank", s, "aurc", v, src].append(aurc(crit, loss) * 1000)
                                acc["rank", s, "eaurc", v, src].append(e_aurc(crit, loss) * 1000)
    return acc


def build_tables(acc: dict, data: dict[str, dict], sources: list[str], out: Path, n_seeds: int, n_splits: int) -> str:
    """Write all tables as csv files and ``table.md``; returns the markdown."""
    t = Tables(out)
    metrics = [("ece", "ECE", 4), ("nll", "NLL", 4), ("brier", "Brier", 4)]
    specs = [{"header": "accuracy", "digits": 4, "mode": "ms"}, {"header": "T", "digits": 3, "mode": "ms"}]
    for k, label, digits in metrics:
        specs += [{"header": f"{label} raw", "digits": digits, "mode": "ms"}, {"header": f"{label} TS", "digits": digits, "mode": "ms"}]
    rows = [([s], [("clean", "acc", "raw", s), ("clean", "temperature", "ts", s)] + [("clean", k, v, s) for k, _, _ in metrics for v in VARIANTS]) for s in sources]
    t.add("clean_calibration", "a) Clean calibration on the test half (TS: temperature fitted on the selection half)", ["source"], rows, specs, acc)
    for key, title in (("aurc", "AURC"), ("eaurc", "E-AURC")):
        specs = [{"header": f"{s} {v}", "digits": 2, "mode": "ms"} for s in RANK_SCORES for v in ("raw", "TS")]
        rows = [([src], [("rank", s, key, v, src) for s in RANK_SCORES for v in VARIANTS]) for src in sources]
        t.add(f"ranking_{key}", f"b) Selective ranking: {title} x1000 on the clean test half (lower is better)", ["source"], rows, specs, acc)
    for corruption in CORRUPTIONS:
        sets = [(s, corrupted_name(corruption, s)) for s in SEVERITIES if corrupted_name(corruption, s) in data]
        if not sets:
            continue
        specs = [{"header": "accuracy", "digits": 3, "mode": "mean"}]
        for k, label, digits in metrics:
            specs += [{"header": f"{label} raw", "digits": digits, "mode": "ms"}, {"header": f"{label} TS", "digits": digits, "mode": "ms"}]
        rows = []
        for s, d in [(0, CLEAN), *sets]:
            for src in sources:
                if ("shift", d, "ece", "raw", src) in acc:
                    rows.append(([str(s), src], [("shift", d, "acc", "raw", src)] + [("shift", d, k, v, src) for k, _, _ in metrics for v in VARIANTS]))
        t.add(f"shift_{corruption}", f"c) Shift: {corruption} (severity 0 is the clean test half; TS uses the clean-fitted T)", ["severity", "source"], rows, specs, acc)
    head = [
        "# Calibration of the probability sources (CIFAR-10 VGG-16)",
        "",
        f"{n_seeds} seeds x {n_splits} random 5k/5k splits of the clean test set; entries are mean +- std over all seed x split",
        "pairs. A temperature T is fitted by NLL on the selection half; everything is reported on the test half, raw and",
        "temperature scaled (TS). ECE uses 15 equal-width bins on the top-class confidence. Brier is the mean over samples of",
        "sum_k (p_k - onehot_k)^2. Ranking scores: `msr` = 1 - max p, `tu_log` = entropy in nats, `tu_brier` = 1 - sum_k p_k^2;",
        "AURC and E-AURC are multiplied by 1000. TS does not change the predicted class, but it can change the `msr` ranking",
        "for more than two classes, so raw and TS numbers can differ. Shifted sets use the same test-half",
        "images (corrupted) and the clean-fitted T. `ens` is the mean of the base models over seeds, a single unit whose spread",
        "comes from the random splits only.",
        "",
    ]
    md = "\n".join(head + t.md)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def plot_ece(path: Path, acc: dict, data: dict[str, dict], sources: list[str]) -> None:
    """ECE versus corruption severity (0 is clean) for the key sources, solid raw and dashed temperature scaled."""
    corruptions = [c for c in CORRUPTIONS if any(corrupted_name(c, s) in data for s in SEVERITIES)]
    shown = [s for s in PLOT_SOURCES if s in sources]
    fig, axes = plt.subplots(1, len(corruptions), figsize=(10, 2.9), squeeze=False)
    for ax, corruption in zip(axes[0], corruptions, strict=True):
        sevs = [s for s in SEVERITIES if corrupted_name(corruption, s) in data]
        for src in shown:
            label = CRITERIA[SOURCE_STYLE[src][1]]["label"]
            color, lw = PLOT_SOURCES[src]
            for v, ls in (("raw", "-"), ("ts", "--")):
                ys = [mean_of(acc, ("shift", CLEAN, "ece", v, src))]
                ys += [mean_of(acc, ("shift", corrupted_name(corruption, s), "ece", v, src)) for s in sevs]
                name = label if v == "raw" else f"{label}, temperature scaled"
                ax.plot([0, *sevs], ys, color=color, ls=ls, marker="o", ms=3, lw=lw, label=name)
        ax.set_title(corruption.replace("_", " "), fontweight="semibold")
        ax.set_xticks([0, *sevs])
        ax.set_xlabel("severity")
        ax.grid(alpha=0.25)
        style_axes(ax)
    axes[0][0].set_ylabel("ECE")
    axes[0][0].yaxis.label.set_fontweight("semibold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=200)
    plt.close(fig)


def summary(acc: dict, sources: list[str]) -> None:
    """Print per source the clean ECE raw -> TS, the temperature and the AURC (msr) raw -> TS."""
    print("source        ECE raw -> TS        T       AURC msr x1000 raw -> TS")
    for s in sources:
        print(
            f"{s:12s}  {mean_of(acc, ('clean', 'ece', 'raw', s)):.4f} -> {mean_of(acc, ('clean', 'ece', 'ts', s)):.4f}"
            f"   {mean_of(acc, ('clean', 'temperature', 'ts', s)):.3f}"
            f"   {mean_of(acc, ('rank', 'msr', 'aurc', 'raw', s)):.2f} -> {mean_of(acc, ('rank', 'msr', 'aurc', 'ts', s)):.2f}"
        )


def main() -> None:
    """Load the dumps, run the protocol, write tables and the plot."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(f.parent.parent.name[4:]) for f in args.runs.glob("seed*/shift/cifar10.npz"))
    if not seeds:
        msg = f"No seed with shift/cifar10.npz under {args.runs}; run scripts/dump_shift.py first."
        raise SystemExit(msg)
    data = {}
    for name in [CLEAN, *SHIFT_DATASETS]:
        loaded = load_dataset(args.runs, seeds, name)
        if loaded is None:
            print(f"skipping {name}: not dumped for all seeds {seeds}.")
        else:
            data[name] = loaded
    clean_units = data[CLEAN]["units"]
    sources = [s for s in SOURCES if s in clean_units and len(clean_units[s]) > 0]
    if "gda" in sources and all(np.array_equal(a, b) for a, b in zip(clean_units["gda"], clean_units["sr_base"], strict=True)):
        print("skipping gda: its probabilities are identical to sr_base.")
        sources.remove("gda")
    print(f"skipping sources without dumps: {', '.join(s for s in SOURCES if s not in sources and s != 'gda') or 'none'}.")
    setup_fonts()
    acc = run_protocol(data, sources, args.n_splits, args.split_seed)
    build_tables(acc, data, sources, args.out, len(seeds), args.n_splits)
    written = ["table.md"]
    if any(d in data for d in SHIFT_DATASETS):
        plot_ece(args.out / "ece_vs_severity.png", acc, data, sources)
        written.append("ece_vs_severity.png")
    summary(acc, sources)
    print(f"Wrote {', '.join(written)} and the csv tables to {args.out}")


if __name__ == "__main__":
    main()
