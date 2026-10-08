"""Selective-prediction quality against forward passes per input: is the extra inference of ensembles and MC dropout worth it?

Two families are swept over their inference cost. Deep ensemble: the base models (one per seed) are the members; for
each size ``k`` the units are the subsets of ``k`` members (all of them, or ``--max-subsets`` drawn with a fixed rng),
and a unit is the mean of the member softmaxes. MC dropout: for each sample count ``T`` the units are the first ``T``
MC samples of every seed, plus ``--mc-repeats`` random ``T``-subsets per seed (``T`` below the stored maximum). The
criteria are ``maxprob`` (1 - max of the mean probabilities) and ``epistemic`` (mutual information, as in
``evaluate_shift.load_dataset``; undefined for one pass). The deterministic dropout softmax ("dropout SR") costs one pass.
The MC samples are those of the fine-tuned dropout model (``predictions_dropout.npz``, the ``mc_*`` criteria), not of
``dropout_scratch``. A pass is counted as a full forward pass; dropout only sits in the classifier head, so an MC
implementation that reuses the trunk would be cheaper than this axis suggests.

Same protocol as ``paper_table.py``: per unit, ``n_splits`` permutations of the test set (one list shared by all units);
the first half selects, the second half is the test half. probly's ``SGRSelector`` is fitted at each r*; a split where
it certifies nothing has test coverage 0 and no test risk. Accuracy and AURC are computed on the full test set.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from math import comb
from pathlib import Path
import warnings

from evaluate import THIN_FONT, setup_fonts
from evaluate_shift import sgr_selector_threshold
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sgr_experiment.metrics import apply_threshold, aurc  # noqa: E402
from sgr_experiment.uncertainty import decompose, member_representation, one_minus_max  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

GAIN_RISK = 0.02
STYLES = {
    ("ensemble", "maxprob"): {"label": "Ensemble, maxprob", "color": "#1e88e5", "ls": "--", "marker": "o"},
    ("ensemble", "epistemic"): {"label": "Ensemble, epistemic", "color": "#43a047", "ls": "--", "marker": "s"},
    ("mc_dropout", "maxprob"): {"label": "MC dropout, maxprob", "color": "#ff8f00", "ls": "-", "marker": "o"},
    ("mc_dropout", "epistemic"): {"label": "MC dropout, epistemic", "color": "#9b59b6", "ls": "-", "marker": "s"},
}
FAMILY_TITLES = {"ensemble": "Deep ensemble (size k)", "mc_dropout": "MC dropout (samples T)"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "cost")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of the SGR bound.")
    p.add_argument("--risks", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05])
    p.add_argument("--max-subsets", type=int, default=30, help="Cap on the ensemble subsets per size k.")
    p.add_argument("--mc-samples", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100])
    p.add_argument("--mc-repeats", type=int, default=3, help="Extra random T-subsets per seed for T below the maximum.")
    return p.parse_args()


def load_predictions(runs: Path, seeds: list[int]) -> dict:
    """Labels, base softmaxes, dropout softmaxes and MC samples of the seeds that have both files."""
    labels = None
    out: dict = {"seeds": [], "base": [], "dropout": [], "mc": []}
    for s in seeds:
        fb = runs / f"seed{s}" / "predictions_base.npz"
        fd = runs / f"seed{s}" / "predictions_dropout.npz"
        if not (fb.exists() and fd.exists()):
            print(f"seed {s}: missing {fb.name} or {fd.name} in {fb.parent}, skipped")
            continue
        base, drop = np.load(fb), np.load(fd)
        for d in (base, drop):
            if labels is None:
                labels = d["labels"]
            np.testing.assert_array_equal(d["labels"], labels)
        out["seeds"].append(s)
        out["base"].append(base["softmax"].astype(np.float64))
        out["dropout"].append(drop["softmax"].astype(np.float64))
        out["mc"].append(drop["mc_probs"])
    if labels is None:
        msg = f"no seed has both prediction files for seeds {seeds} in {runs}"
        raise SystemExit(msg)
    out["labels"] = labels
    return out


def make_units(probs: torch.Tensor, criteria: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Criteria and prediction of the mean of member probabilities ``(members, n, classes)``."""
    mean = probs.mean(0).numpy()
    units = {"maxprob": (one_minus_max(mean), mean.argmax(1))}
    if "epistemic" in criteria and probs.shape[0] > 1:
        units["epistemic"] = (decompose(member_representation(probs))["epistemic"], mean.argmax(1))
    return units


def ensemble_groups(data: dict, max_subsets: int, rng: np.random.Generator) -> dict[int, dict[str, list]]:
    """Units per ensemble size ``k``: criterion -> list of ``(criterion values, prediction)``."""
    members = np.stack(data["base"])
    n_members = len(members)
    groups: dict[int, dict[str, list]] = {}
    for k in range(1, n_members + 1):
        if comb(n_members, k) <= max_subsets:
            subsets = list(combinations(range(n_members), k))
        else:
            subsets = [tuple(sorted(rng.choice(n_members, k, replace=False))) for _ in range(max_subsets)]
        groups[k] = {"maxprob": [], "epistemic": []}
        for sub in subsets:
            for crit, unit in make_units(torch.from_numpy(members[list(sub)]), ["maxprob", "epistemic"]).items():
                groups[k][crit].append(unit)
    return groups


def mc_groups(data: dict, samples: list[int], repeats: int, rng: np.random.Generator) -> dict[int, dict[str, list]]:
    """Units per MC sample count ``T``: criterion -> list of ``(criterion values, prediction)``."""
    groups: dict[int, dict[str, list]] = {}
    for t in samples:
        groups[t] = {"maxprob": [], "epistemic": []}
        for mc in data["mc"]:
            t_max = mc.shape[0]
            if t > t_max:
                continue
            idx = [np.arange(t)]
            if t < t_max:
                idx += [np.sort(rng.choice(t_max, t, replace=False)) for _ in range(repeats)]
            for ix in idx:
                for crit, unit in make_units(torch.from_numpy(mc[ix].astype(np.float64)), ["maxprob", "epistemic"]).items():
                    groups[t][crit].append(unit)
    return groups


def evaluate_units(units: list, labels: np.ndarray, splits: list[np.ndarray], risks: list[float], delta: float) -> dict:
    """Accuracy, AURC and per r* certified flag, test coverage and test risk over units x splits."""
    half = len(labels) // 2
    res: dict = {"n_units": len(units), "acc": [], "aurc": []}
    for r in risks:
        res[r] = {"cert": [], "cov": [], "risk": []}
    for crit, pred in units:
        loss = (pred != labels).astype(np.float64)
        res["acc"].append(1 - loss.mean())
        res["aurc"].append(aurc(crit, loss))
        for perm in splits:
            sel, test = perm[:half], perm[half:]
            for r in risks:
                thr, _ = sgr_selector_threshold(crit[sel], loss[sel], r, delta)
                certified = thr != -np.inf
                risk, cov = apply_threshold(crit[test], loss[test], thr) if certified else (np.nan, 0.0)
                res[r]["cert"].append(float(certified))
                res[r]["cov"].append(cov if certified else 0.0)
                res[r]["risk"].append(risk if certified else np.nan)
    return res


def collect(data: dict, args: argparse.Namespace) -> list[dict]:
    """One result row per (family, criterion, size)."""
    labels = data["labels"]
    rng_split = np.random.default_rng(args.split_seed)
    splits = [rng_split.permutation(len(labels)) for _ in range(args.n_splits)]
    rng_units = np.random.default_rng(args.split_seed + 1)
    rows = []

    def add(family: str, crit: str, size: int, passes: int, units: list) -> None:
        if units:
            rows.append({"family": family, "criterion": crit, "size": size, "passes": passes,
                         **evaluate_units(units, labels, splits, args.risks, args.delta)})
            print(f"{family} {crit} size {size}: {len(units)} units")

    dropout_units = [(one_minus_max(p), p.argmax(1)) for p in data["dropout"]]
    add("dropout_sr", "maxprob", 1, 1, dropout_units)
    for k, by_crit in ensemble_groups(data, args.max_subsets, rng_units).items():
        for crit, units in by_crit.items():
            add("ensemble", crit, k, k, units)
    for t, by_crit in mc_groups(data, args.mc_samples, args.mc_repeats, rng_units).items():
        for crit, units in by_crit.items():
            add("mc_dropout", crit, t, t, units)
    return rows


def ms(values: list[float], scale: float = 1.0) -> tuple[float, float]:
    """Mean and std (NaN-ignoring) of ``values`` times ``scale``."""
    v = np.asarray(values, dtype=float) * scale
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def cell(values: list[float], digits: int = 3, scale: float = 1.0) -> str:
    """Format ``mean +- std`` (or ``n/a``)."""
    mean, std = ms(values, scale)
    return "n/a" if not np.isfinite(mean) else f"{mean:.{digits}f} +- {std:.{digits}f}"


def build_markdown(args: argparse.Namespace, rows: list[dict], n_test: int, seeds: list[int]) -> str:
    """One table per family and criterion; rows are sizes, with the marginal SGR coverage gain at r* 0.02."""
    gain_r = GAIN_RISK if GAIN_RISK in args.risks else args.risks[0]
    lines = [
        "# Selective prediction against forward passes per input", "",
        f"Seeds {seeds}, {n_test} test points, {args.n_splits} splits, delta {args.delta}. Cells are mean +- std over units x "
        f"splits; `cert` is the share of splits where SGR certifies, `cov` the SGR test coverage (0 if not certified). "
        f"`gain` is the change of `cov` at r* = {gain_r} from the previous row.", "",
    ]
    header = ["size", "passes", "units", "acc", "AURC (x1000)"]
    for r in args.risks:
        header += [f"cert r*={r}", f"cov r*={r}"]
    header.append(f"gain r*={gain_r}")
    groups = [(f, c) for f in ("dropout_sr", "ensemble", "mc_dropout") for c in ("maxprob", "epistemic")]
    for fam, crit in groups:
        sub = sorted((x for x in rows if x["family"] == fam and x["criterion"] == crit), key=lambda x: x["size"])
        if not sub:
            continue
        title = "Dropout SR (deterministic softmax)" if fam == "dropout_sr" else FAMILY_TITLES[fam]
        lines += [f"## {title}, {crit}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        prev = None
        for x in sub:
            cells = [str(x["size"]), str(x["passes"]), str(x["n_units"]), cell(x["acc"], 4), cell(x["aurc"], 3, 1000)]
            for r in args.risks:
                cells += [f"{np.mean(x[r]['cert']) * 100:.0f}%", cell(x[r]["cov"], 3)]
            mean_cov = float(np.mean(x[gain_r]["cov"]))
            cells.append("-" if prev is None else f"{mean_cov - prev:+.3f}")
            prev = mean_cov
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict], risks: list[float]) -> None:
    """Long format: one line per (family, criterion, size, r*)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "criterion", "size", "passes", "n_units", "r_star", "acc_mean", "acc_std", "aurc_mean", "aurc_std",
                    "certified_mean", "cov_mean", "cov_std", "risk_mean", "risk_std"])
        for x in rows:
            acc, aurc_ = ms(x["acc"]), ms(x["aurc"])
            for r in risks:
                w.writerow([x["family"], x["criterion"], x["size"], x["passes"], x["n_units"], r, *acc, *aurc_,
                            float(np.mean(x[r]["cert"])), *ms(x[r]["cov"]), *ms(x[r]["risk"])])


def style_axes(ax: plt.Axes, has_fira: bool) -> None:
    """Semibold axis labels and thin tick labels."""
    ax.xaxis.label.set_fontweight("semibold")
    ax.yaxis.label.set_fontweight("semibold")
    if has_fira:
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(THIN_FONT)


def plot(path: Path, rows: list[dict], risks: list[float]) -> None:
    """SGR test coverage per r* and AURC against forward passes (log axis)."""
    has_fira = setup_fonts()
    fig, axes = plt.subplots(1, len(risks) + 1, figsize=(3.6 * (len(risks) + 1), 3.8))
    for ax, key in zip(axes, [*risks, "aurc"], strict=True):
        for (fam, crit), st in STYLES.items():
            sub = sorted((x for x in rows if x["family"] == fam and x["criterion"] == crit), key=lambda x: x["size"])
            if not sub:
                continue
            xs = [x["passes"] for x in sub]
            vals = [ms(x["aurc"], 1000) if key == "aurc" else ms(x[key]["cov"]) for x in sub]
            mean, std = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            ax.errorbar(xs, mean, yerr=std, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, lw=1.4, capsize=2,
                        label=st["label"])
        for x in (x for x in rows if x["family"] == "dropout_sr"):
            val = ms(x["aurc"], 1000) if key == "aurc" else ms(x[key]["cov"])
            ax.errorbar([1], [val[0]], yerr=[val[1]], color="black", marker="*", ms=9, ls="none", capsize=2, label="Dropout SR")
        base = next((x for x in rows if x["family"] == "ensemble" and x["criterion"] == "maxprob" and x["size"] == 1), None)
        if base is not None:
            val = ms(base["aurc"], 1000) if key == "aurc" else ms(base[key]["cov"])
            ax.errorbar([1], [val[0]], yerr=[val[1]], color="#555555", marker="D", ms=6, ls="none", capsize=2, label="Base SR (k = 1)")
        ax.set_xscale("log")
        ax.set_xlabel("Forward passes per input")
        ax.set_ylabel("AURC (x1000)" if key == "aurc" else "SGR test coverage")
        ax.set_title("AURC" if key == "aurc" else f"r* = {key:.2f}")
        ax.grid(alpha=0.3)
        style_axes(ax, has_fira)
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=True))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=len(uniq), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Evaluate the cost sweep, write ``table.md``, ``table.csv`` and ``cost.png``."""
    args = parse_args()
    data = load_predictions(args.runs, args.seeds)
    rows = collect(data, args)
    args.out.mkdir(parents=True, exist_ok=True)
    md = build_markdown(args, rows, len(data["labels"]), data["seeds"])
    (args.out / "table.md").write_text(md)
    write_csv(args.out / "table.csv", rows, args.risks)
    plot(args.out / "cost.png", rows, args.risks)
    print(md)


if __name__ == "__main__":
    main()
