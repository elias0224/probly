"""Selection criteria on CIFAR-10, under covariate shift and on OOD data (thresholds always from clean CIFAR-10).

Same seed x split protocol as ``evaluate.py``: ``n_splits`` permutations of the 10k clean test set, the first half
selects the threshold, the second half is the test half. Per desired risk r* the threshold is chosen three ways: ``emp``
(largest coverage with empirical selection-half risk <= r*), ``sgr`` (probly's ``SGRSelector``, Algorithm 1 of Geifman
and El-Yaniv, with a confidence bound; ``metrics.sgr_threshold`` is kept as a reference and compared) and ``cov``
(probly's label-free ``CoverageSelector``, calibrated to the coverage that ``emp`` reached on the selection half). Shifted and OOD sets are evaluated at these clean thresholds. The deep-ensemble criteria use the
base models (one per seed) as members; there is only one ensemble, so its spread comes from the random splits only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import warnings

from evaluate import PAPER, THIN_FONT, setup_fonts
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from probly.selective_prediction import CoverageSelector, SGRSelector  # noqa: E402
from sgr_experiment.metrics import (  # noqa: E402
    apply_threshold,
    aurc,
    auroc,
    e_aurc,
    risk_coverage_curve,
    sgr_threshold,
    threshold_for_risk,
)
from sgr_experiment.shift import CORRUPTIONS, OOD_DATASETS, SEVERITIES, corrupted_name  # noqa: E402
from sgr_experiment.uncertainty import decompose, member_representation, one_minus_max  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

# label, color and line style per criterion; the order is the order of all tables and plots.
CRITERIA = {
    "sr_base": {"label": "SR, base model", "color": "#16a085", "ls": "-"},
    "sr_dropout": {"label": "SR, dropout model", "color": "#9b59b6", "ls": "-"},
    "mc_maxprob": {"label": "MC, 1 - max mean prob", "color": "#1e88e5", "ls": "-"},
    "mc_total": {"label": "MC, total entropy", "color": "#0d2f6e", "ls": "-"},
    "mc_aleatoric": {"label": "MC, aleatoric", "color": "#fb8c00", "ls": "-"},
    "mc_epistemic": {"label": "MC, epistemic", "color": "#43a047", "ls": "-"},
    "mc_variance": {"label": "MC, paper variance", "color": "#ff0d57", "ls": "-"},
    "ens_maxprob": {"label": "Ensemble, 1 - max mean prob", "color": "#1e88e5", "ls": "--"},
    "ens_total": {"label": "Ensemble, total entropy", "color": "#0d2f6e", "ls": "--"},
    "ens_aleatoric": {"label": "Ensemble, aleatoric", "color": "#fb8c00", "ls": "--"},
    "ens_epistemic": {"label": "Ensemble, epistemic", "color": "#43a047", "ls": "--"},
}
# Post-training methods (``dump_methods.py``); they are added when their dumps exist for all seeds.
_METHOD_STYLE = {
    "finetune": ("Fine-tuned base", "#795548"),
    "swag": ("SWAG", "#00acc1"),
    "laplace": ("Laplace (last layer)", "#c0ca33"),
    "gda": ("GDA", "#e91e63"),
    "ddu": ("DDU-style", "#5e35b1"),
    "vbll": ("VBLL", "#f4511e"),
    "sngp": ("SNGP", "#2e7d32"),
}
_QUANTITY_LS = {"maxprob": "-", "total": ":", "aleatoric": "-.", "epistemic": "--", "density": "--", "ds": "--"}
# keys of the method npz files that become criteria (``{method}_{key}``)
METHOD_KEYS = {
    "finetune": ["maxprob"],
    "swag": ["maxprob", "total", "aleatoric", "epistemic"],
    "laplace": ["maxprob", "total", "aleatoric", "epistemic"],
    "gda": ["density"],
    "ddu": ["maxprob", "density"],
    "vbll": ["maxprob", "total", "aleatoric", "epistemic"],
    "sngp": ["maxprob", "ds"],
}
for _m, _keys in METHOD_KEYS.items():
    for _k in _keys:
        CRITERIA[f"{_m}_{_k}"] = {
            "label": f"{_METHOD_STYLE[_m][0]}, {_k}",
            "color": _METHOD_STYLE[_m][1],
            "ls": _QUANTITY_LS[_k],
        }
CRITERIA["swa_maxprob"] = {"label": "SWA mean, 1 - max prob", "color": "#4dd0e1", "ls": "-"}
MAIN_GROUP = [c for c in CRITERIA if c.startswith(("sr_", "mc_", "ens_"))]
# Figures of the methods: confidence scores (1 - max prob) in one, epistemic and density scores in the other.
FIGURE_GROUPS = {
    "": MAIN_GROUP,
    "_methods_maxprob": [
        "sr_base", "mc_maxprob", "ens_maxprob", "finetune_maxprob", "swa_maxprob", "swag_maxprob", "laplace_maxprob", "ddu_maxprob", "vbll_maxprob", "sngp_maxprob",
    ],
    "_methods_uncertainty": [
        "mc_epistemic", "ens_epistemic", "swag_epistemic", "laplace_epistemic", "vbll_epistemic", "gda_density", "ddu_density", "sngp_ds",
    ],
}
ID_RISKS = [r for r, _, _ in PAPER]
SHIFT_RISKS = [0.01, 0.03, 0.05]
MODES = ["emp", "sgr", "cov"]
SHIFT_DATASETS = [corrupted_name(c, s) for c in CORRUPTIONS for s in SEVERITIES]
PLOT_RISK = 0.03
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "shift")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of the SGR bound.")
    return p.parse_args()


def load_dataset(runs: Path, seeds: list[int], name: str) -> dict | None:
    """Labels and per criterion a list of ``(criterion, predicted class)`` units (seeds, or the one ensemble).

    Returns None if a seed has no file for the dataset.
    """
    files = [runs / f"seed{s}" / "shift" / f"{name}.npz" for s in seeds]
    if not all(f.exists() for f in files):
        return None
    ds = [np.load(f) for f in files]
    labels = ds[0]["labels"]
    for d in ds:
        np.testing.assert_array_equal(d["labels"], labels)
    units: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in CRITERIA if c.startswith(("sr_", "mc_", "ens_"))}
    for d in ds:
        for c, key in (("sr_base", "softmax_base"), ("sr_dropout", "softmax_dropout")):
            p = d[key].astype(np.float64)
            units[c].append((one_minus_max(p), p.argmax(1)))
        pred = d["mean_probs"].argmax(1)
        # Recomputed from the stored mean, the stored float32 1 - max has ties at 0 (see one_minus_max).
        units["mc_maxprob"].append((one_minus_max(d["mean_probs"]), pred))
        for c in ("mc_total", "mc_aleatoric", "mc_epistemic", "mc_variance"):
            units[c].append((d[c].astype(np.float64), pred))
    for m, keys in METHOD_KEYS.items():
        mfiles = [runs / f"seed{sd}" / "shift" / m / f"{name}.npz" for sd in seeds]
        if not all(f.exists() for f in mfiles):
            continue
        for f in mfiles:
            md = np.load(f)
            np.testing.assert_array_equal(md["labels"], labels)
            pred = md["mean_probs"].argmax(1)
            for k in keys:
                crit = one_minus_max(md["mean_probs"]) if k == "maxprob" else md[k].astype(np.float64)
                units.setdefault(f"{m}_{k}", []).append((crit, pred))
            if m == "swag":
                p = md["softmax_swa"].astype(np.float64)
                units.setdefault("swa_maxprob", []).append((one_minus_max(p), p.argmax(1)))
    members = torch.from_numpy(np.stack([d["softmax_base"] for d in ds]).astype(np.float64))
    mean = members.mean(0).numpy()
    uq = decompose(member_representation(members))
    pred = mean.argmax(1)
    units["ens_maxprob"].append((one_minus_max(mean), pred))
    for k in ("total", "aleatoric", "epistemic"):
        units[f"ens_{k}"].append((uq[k], pred))
    return {"labels": labels, "units": units}


def fmt(values: list[float], digits: int, mode: str, mark_above: float | None = None) -> str:
    """Format the values of one cell: ``ms`` mean +- std, ``pct`` mean as a percentage, ``mean`` the mean only."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    if mode == "pct":
        return f"{v.mean() * 100:.0f}%"
    if mode == "count":
        return f"{int(v.sum())} of {v.size}"
    star = " *" if mark_above is not None and v.mean() > mark_above + EPS else ""
    if mode == "mean":
        return f"{v.mean():.{digits}f}{star}"
    return f"{v.mean():.{digits}f} +- {v.std():.{digits}f}{star}"


def active_criteria(data: dict[str, dict]) -> list[str]:
    """Criteria available on the clean test set, in the order of ``CRITERIA``."""
    return [c for c in CRITERIA if data["cifar10"]["units"].get(c)]


def sgr_selector_threshold(crit: np.ndarray, loss: np.ndarray, r: float, delta: float) -> tuple[float, float]:
    """Threshold and bound of probly's ``SGRSelector`` (-inf and NaN if nothing is certified, without the warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sel = SGRSelector(r, delta).calibrate(crit, loss)
    return float(sel.threshold), float(sel.bound)


def run_protocol(data: dict[str, dict], n_splits: int, split_seed: int, delta: float) -> dict[tuple, list[float]]:
    """Evaluate every criterion, unit and split; returns the list of values per result key."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    clean = data["cifar10"]
    labels = clean["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(split_seed)
    splits = [rng.permutation(n) for _ in range(n_splits)]
    ood = [d for d in OOD_DATASETS if d in data]
    shifted = [d for d in SHIFT_DATASETS if d in data]
    rng_ood = np.random.default_rng(split_seed + 1)
    subsets = {d: [rng_ood.choice(len(data[d]["labels"]), n - half, replace=len(data[d]["labels"]) < n - half) for _ in splits] for d in ood}

    for c in active_criteria(data):
        ood_c = [d for d in ood if data[d]["units"].get(c)]
        shifted_c = [d for d in shifted if data[d]["units"].get(c)]
        for u, (crit, pred) in enumerate(clean["units"][c]):
            loss = (pred != labels).astype(np.float64)
            for si, perm in enumerate(splits):
                sel, test = perm[:half], perm[half:]
                acc["aurc", c].append(aurc(crit[test], loss[test]) * 1000)
                acc["eaurc", c].append(e_aurc(crit[test], loss[test]) * 1000)
                acc["acc", c].append(1 - loss[test].mean())
                for d in ood_c:
                    acc["auroc", d, c].append(auroc(crit[test], data[d]["units"][c][u][0]))
                for r in sorted(set(ID_RISKS) | set(SHIFT_RISKS)):
                    thr_emp = threshold_for_risk(crit[sel], loss[sel], r)
                    thr_sgr, bound = sgr_selector_threshold(crit[sel], loss[sel], r, delta)
                    acc["refdiff", c].append(float(thr_sgr != sgr_threshold(crit[sel], loss[sel], r, delta)[0]))
                    acc["uncertified", r, c].append(float(thr_sgr == -np.inf))
                    # Label-free: the coverage the emp threshold reached on the selection half is the target.
                    target = apply_threshold(crit[sel], loss[sel], thr_emp)[1]
                    thr_cov = CoverageSelector(target).calibrate(crit[sel]).threshold if target > 0 else -np.inf
                    thresholds = {"emp": thr_emp, "sgr": thr_sgr, "cov": thr_cov}
                    if r in ID_RISKS:
                        acc["bound", r, c].append(bound)
                    for mode, thr in thresholds.items():
                        risk, cov = apply_threshold(crit[test], loss[test], thr)
                        if r in ID_RISKS:
                            acc["id", mode, r, c, "risk"].append(risk)
                            acc["id", mode, r, c, "cov"].append(cov)
                            acc["id", mode, r, c, "viol"].append(float(risk > r + EPS) if np.isfinite(risk) else 0.0)
                        for d in ood_c:
                            ood_crit = data[d]["units"][c][u][0]
                            if r in SHIFT_RISKS:
                                acc["ood_acc", d, mode, r, c].append(float((ood_crit <= thr).mean()))
                            mix_crit = np.concatenate([crit[test], ood_crit[subsets[d][si]]])
                            mix_loss = np.concatenate([loss[test], np.ones(len(test))])
                            risk, cov = apply_threshold(mix_crit, mix_loss, thr)
                            acc["mix", d, mode, r, c, "risk"].append(risk)
                            acc["mix", d, mode, r, c, "cov"].append(cov)
                        if r in SHIFT_RISKS:
                            for d in shifted_c:
                                s_crit, s_pred = data[d]["units"][c][u]
                                s_loss = (s_pred != labels).astype(np.float64)
                                risk, cov = apply_threshold(s_crit[test], s_loss[test], thr)
                                acc["shift", d, mode, r, c, "risk"].append(risk)
                                acc["shift", d, mode, r, c, "cov"].append(cov)
                for d in shifted_c:
                    acc["shift_acc", d, c].append(1 - (data[d]["units"][c][u][1] != labels)[test].mean())
    return acc


class Tables:
    """Collects markdown sections and writes one CSV (mean and std columns) per table."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.md: list[str] = []

    def add(self, name: str, title: str, id_cols: list[str], rows: list[tuple[list[str], list[tuple]]], specs: list[dict], acc: dict) -> None:
        """Add a table: each row has id strings and one result key per metric spec (header, digits, mode, mark)."""
        header = id_cols + [s["header"] for s in specs]
        self.md += [f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        csv_header = id_cols + [f"{s['header']}_{k}" for s in specs for k in ("mean", "std")]
        csv_rows = []
        for ids, keys in rows:
            cells, nums = [], []
            for s, key in zip(specs, keys, strict=True):
                vals = acc.get(key, [])
                cells.append(fmt(vals, s["digits"], s["mode"], s.get("mark")))
                v = np.asarray(vals, dtype=float)
                v = v[np.isfinite(v)]
                nums += [f"{v.mean():.6g}", f"{v.std():.6g}"] if v.size else ["", ""]
            self.md.append("| " + " | ".join([*ids, *cells]) + " |")
            csv_rows.append([*ids, *nums])
        self.md.append("")
        with (self.out / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(csv_header)
            w.writerows(csv_rows)


def build_tables(acc: dict, data: dict[str, dict], out: Path, n_seeds: int, n_splits: int) -> str:
    """Write all tables as csv files and ``table.md``; returns the markdown."""
    t = Tables(out)
    ood = [d for d in OOD_DATASETS if d in data]
    shifted = [d for d in SHIFT_DATASETS if d in data]
    crits = active_criteria(data)
    t.add(
        "id_aurc", "a) ID: AURC and E-AURC (x1000, lower is better) and accuracy", ["criterion"],
        [([c], [("aurc", c), ("eaurc", c), ("acc", c)]) for c in crits],
        [{"header": "AURC x1000", "digits": 2, "mode": "ms"}, {"header": "E-AURC x1000", "digits": 2, "mode": "ms"}, {"header": "accuracy", "digits": 4, "mode": "ms"}],
        acc,
    )
    t.add(
        "sgr_check", "SGR: probly SGRSelector vs the reference sgr_threshold, and uncertified thresholds", ["criterion"],
        [([c], [("refdiff", c)] + [("uncertified", r, c) for r in ID_RISKS]) for c in crits],
        [{"header": "thresholds differing from reference", "digits": 0, "mode": "count"}]
        + [{"header": f"uncertified r*={r}", "digits": 0, "mode": "count"} for r in ID_RISKS],
        acc,
    )
    specs, rows = [], []
    for mode in MODES:
        specs += [{"header": f"{mode} risk", "digits": 4, "mode": "ms"}, {"header": f"{mode} cov", "digits": 4, "mode": "ms"}, {"header": f"{mode} viol", "digits": 0, "mode": "pct"}]
    specs.append({"header": "sgr bound", "digits": 4, "mode": "mean"})
    for r in ID_RISKS:
        for c in crits:
            keys = [("id", m, r, c, k) for m in MODES for k in ("risk", "cov", "viol")] + [("bound", r, c)]
            rows.append(([f"{r:.2f}", c], keys))
    t.add("id_thresholds", "a) ID: test-half risk, coverage and violation share (risk > r*) at thresholds chosen on the selection half", ["r*", "criterion"], rows, specs, acc)
    for d in ood:
        specs = [{"header": "AUROC", "digits": 4, "mode": "ms"}]
        for r in SHIFT_RISKS:
            specs += [{"header": f"{m} accepted r*={r}", "digits": 3, "mode": "ms"} for m in MODES]
        rows = [([c], [("auroc", d, c)] + [("ood_acc", d, m, r, c) for r in SHIFT_RISKS for m in MODES]) for c in crits]
        t.add(f"ood_{d}", f"b) OOD {d}: AUROC (ID test half vs OOD, criterion as score) and share of OOD images accepted", ["criterion"], rows, specs, acc)
        specs = [{"header": f"{m} {k}", "digits": 4, "mode": "ms"} for m in MODES for k in ("cov", "risk")]
        rows = [([f"{r:.2f}", c], [("mix", d, m, r, c, k) for m in MODES for k in ("cov", "risk")]) for r in ID_RISKS for c in crits]
        t.add(f"mixed_{d}", f"b) Mixed test set, CIFAR-10 test half + equally many {d} images (OOD loss 1)", ["r*", "criterion"], rows, specs, acc)
    for corruption in CORRUPTIONS:
        sets = [(s, corrupted_name(corruption, s)) for s in SEVERITIES if corrupted_name(corruption, s) in data]
        if not sets:
            continue
        specs = [{"header": "accuracy", "digits": 3, "mode": "mean"}]
        for r in SHIFT_RISKS:
            for m in MODES:
                specs += [{"header": f"r*={r} {m} risk", "digits": 4, "mode": "ms", "mark": r}, {"header": f"r*={r} {m} cov", "digits": 3, "mode": "mean"}]
        rows = []
        for s, d in sets:
            for c in crits:
                keys = [("shift_acc", d, c)] + [("shift", d, m, r, c, k) for r in SHIFT_RISKS for m in MODES for k in ("risk", "cov")]
                rows.append(([str(s), c], keys))
        t.add(f"shift_{corruption}", f"c) Shift: {corruption} (* marks mean risk > r*)", ["severity", "criterion"], rows, specs, acc)
    head = [
        "# Selective prediction under distribution shift (CIFAR-10 VGG-16)",
        "",
        f"{n_seeds} seeds x {n_splits} random 5k/5k splits of the clean test set; entries are mean +- std over all seed x split",
        "pairs. Thresholds come from the clean selection half: `emp` is the largest coverage with empirical risk <= r*,",
        "`sgr` is probly's SGRSelector (Algorithm 1 of Geifman and El-Yaniv, delta 0.001) and `cov` probly's",
        "CoverageSelector, calibrated without labels to the selection-half coverage of `emp`. Ensemble criteria (`ens_*`)",
        "use the base models (one per seed) as",
        "members; there is only one ensemble, so its spread comes from the random splits only. The mixed set and the shifted",
        "sets use the test half of the split (shifted sets: the corrupted versions of the same images).",
        "",
    ]
    md = "\n".join(head + t.md)
    (out / "table.md").write_text(md + "\n", encoding="utf-8")
    return md


def style_axes(ax: plt.Axes) -> None:
    """Semibold axis labels and thin tick labels."""
    ax.xaxis.label.set_fontweight("semibold")
    ax.yaxis.label.set_fontweight("semibold")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(THIN_FONT)


def save(fig: plt.Figure, path: Path) -> None:
    """Save without font warnings."""
    fig.tight_layout()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=200)
    plt.close(fig)


def mean_of(acc: dict, key: tuple) -> float:
    """Mean of the finite values of a result key (NaN if there are none)."""
    v = np.asarray(acc.get(key, []), dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def plot_id(path: Path, clean: dict, crits: list[str]) -> None:
    """Mean risk-coverage curves of the criteria on the full clean test set, with the paper's points."""
    grid = np.linspace(0.7, 1.0, 601)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in crits:
        style = CRITERIA[c]
        ys = []
        for crit, pred in clean["units"][c]:
            cov, rsk = risk_coverage_curve(crit, (pred != clean["labels"]).astype(np.float64))
            ys.append(np.interp(grid, cov, rsk))
        ax.plot(grid, np.mean(ys, axis=0), color=style["color"], ls=style["ls"], lw=1.6, label=style["label"])
    ax.scatter([pc for _, _, pc in PAPER], [pr for _, pr, _ in PAPER], marker="D", color="black", zorder=5, label="paper (SGR)")
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    style_axes(ax)
    save(fig, path)


def plot_ood(path: Path, acc: dict, ood: list[str], crits: list[str]) -> None:
    """Share of OOD images accepted at the clean threshold for r* = 0.03 (one row per mode)."""
    fig, axes = plt.subplots(len(MODES), 1, figsize=(9, 3 * len(MODES)), sharex=True, sharey=True)
    width = 0.8 / len(ood)
    for ax, mode in zip(axes, MODES, strict=True):
        for j, d in enumerate(ood):
            vals = [np.asarray(acc[("ood_acc", d, mode, PLOT_RISK, c)], dtype=float) for c in crits]
            ax.bar(
                np.arange(len(crits)) + (j - (len(ood) - 1) / 2) * width,
                [v.mean() for v in vals],
                width,
                yerr=[v.std() for v in vals],
                color=[CRITERIA[c]["color"] for c in crits],
                hatch="//" if j else None,
                edgecolor="white",
                error_kw={"lw": 0.8},
            )
        ax.set_ylabel(f"OOD accepted ({mode})")
        ax.grid(alpha=0.25, axis="y")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="gray", hatch="//" if j else None, ec="white") for j in range(len(ood))]
    axes[0].legend(handles, ood, frameon=False, title=f"r* = {PLOT_RISK}")
    axes[-1].set_xticks(np.arange(len(crits)), crits, rotation=40, ha="right")
    for ax in axes:
        style_axes(ax)
    save(fig, path)


def plot_shift(path: Path, acc: dict, data: dict[str, dict], crits: list[str]) -> None:
    """Risk at the emp threshold for r* = 0.03 versus corruption severity (0 is the clean test half)."""
    corruptions = [c for c in CORRUPTIONS if any(corrupted_name(c, s) in data for s in SEVERITIES)]
    fig, axes = plt.subplots(1, len(corruptions), figsize=(4 * len(corruptions), 4), sharey=True, squeeze=False)
    for ax, corruption in zip(axes[0], corruptions, strict=True):
        sevs = [s for s in SEVERITIES if corrupted_name(corruption, s) in data]
        for c in crits:
            style = CRITERIA[c]
            ys = [mean_of(acc, ("id", "emp", PLOT_RISK, c, "risk"))]
            ys += [mean_of(acc, ("shift", corrupted_name(corruption, s), "emp", PLOT_RISK, c, "risk")) for s in sevs]
            ax.plot([0, *sevs], ys, color=style["color"], ls=style["ls"], marker="o", ms=3, lw=1.4, label=style["label"])
        ax.axhline(PLOT_RISK, color="black", lw=0.8, ls=":")
        ax.set_title(corruption)
        ax.set_xticks([0, *sevs])
        ax.set_xlabel("severity")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(f"risk at emp threshold, r* = {PLOT_RISK}")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    for ax in axes[0]:
        ax.title.set_fontweight("semibold")
        style_axes(ax)
    save(fig, path)


def main() -> None:
    """Load the dumps, run the protocol, write tables and plots."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(f.parent.parent.name[4:]) for f in args.runs.glob("seed*/shift/cifar10.npz"))
    if not seeds:
        msg = f"No seed with shift/cifar10.npz under {args.runs}; run scripts/dump_shift.py first."
        raise SystemExit(msg)
    if len(seeds) < 2:
        print("WARNING: a single seed gives a degenerate ensemble (epistemic uncertainty 0).")
    data = {}
    for name in ["cifar10", *OOD_DATASETS, *SHIFT_DATASETS]:
        loaded = load_dataset(args.runs, seeds, name)
        if loaded is None:
            print(f"skipping {name}: not dumped for all seeds {seeds}.")
        else:
            data[name] = loaded
    setup_fonts()
    acc = run_protocol(data, args.n_splits, args.split_seed, args.delta)
    md = build_tables(acc, data, args.out, len(seeds), args.n_splits)
    written = ["table.md"]
    ood = [d for d in OOD_DATASETS if d in data]
    available = active_criteria(data)
    for suffix, group in FIGURE_GROUPS.items():
        crits = [c for c in group if c in available]
        if not crits or (suffix and not any(c.startswith(tuple(METHOD_KEYS)) or c == "swa_maxprob" for c in crits)):
            continue
        plot_id(args.out / f"id_risk_coverage{suffix}.png", data["cifar10"], crits)
        written.append(f"id_risk_coverage{suffix}.png")
        ood_crits = [c for c in crits if all(("ood_acc", d, "emp", PLOT_RISK, c) in acc for d in ood)]
        if ood and ood_crits:
            plot_ood(args.out / f"ood_acceptance{suffix}.png", acc, ood, ood_crits)
            written.append(f"ood_acceptance{suffix}.png")
        if any(d in data for d in SHIFT_DATASETS):
            plot_shift(args.out / f"shift_risk{suffix}.png", acc, data, crits)
            written.append(f"shift_risk{suffix}.png")
    summary(md, acc, data)
    print(f"Wrote {', '.join(written)} and the csv tables to {args.out}")


def summary(md: str, acc: dict, data: dict[str, dict]) -> None:
    """Print the AURC table and the r* = 0.03 headline numbers."""
    section = md.split("## ")[1]
    print("## " + section)
    print(f"r* = {PLOT_RISK}, test-half coverage / violation share (emp, sgr) and OOD accepted (emp, sgr):")
    for c in active_criteria(data):
        line = f"{c:16s}"
        for m in MODES:
            line += f" {m} cov {mean_of(acc, ('id', m, PLOT_RISK, c, 'cov')):.3f} viol {mean_of(acc, ('id', m, PLOT_RISK, c, 'viol')) * 100:3.0f}%"
        for d in (x for x in OOD_DATASETS if x in data):
            line += f" | {d} " + " ".join(f"{mean_of(acc, ('ood_acc', d, m, PLOT_RISK, c)):.3f}" for m in MODES)
        print(line)
    ref = sum(sum(acc.get(("refdiff", c), [])) for c in active_criteria(data))
    total = sum(len(acc.get(("refdiff", c), [])) for c in active_criteria(data))
    unc = sum(sum(v) for k, v in acc.items() if k[0] == "uncertified")
    print(f"SGRSelector vs reference sgr_threshold: {int(ref)} of {total} thresholds differ; {int(unc)} uncertified (-inf).")
    print("Ensemble spread comes from the random splits only (one ensemble of the base models, one per seed).")


if __name__ == "__main__":
    main()
