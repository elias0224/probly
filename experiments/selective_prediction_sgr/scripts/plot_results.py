"""Summary figures built from the result csv files only (no prediction dumps needed).

Inputs, all under ``--results`` (written by ``evaluate_shift.py``, ``evaluate_options.py`` and
``evaluate_calibration.py``):

- ``options/id_r{r}.csv``: per criterion and accept/abstain mode, test risk, coverage and violation share.
- ``shift/id_aurc.csv``, ``shift/id_thresholds.csv``, ``shift/ood_svhn.csv``, ``shift/shift_{corruption}.csv``.
- ``calibration/clean_calibration.csv`` and ``calibration/ranking_aurc.csv``.

Outputs, written to ``--out`` (default ``results/summary``):

- ``modes.png``: heatmap of the test coverage of every mode per criterion, violating cells outlined in red.
- ``main_scores.png``: clean AURC, SGR coverage, SVHN AUROC and SGR risk under severity-5 shift for the best
  criterion of every method.
- ``shift_risk.png``: SGR test risk against shift severity for the main score of every method.
- ``calibration.png``: clean ECE and msr AURC, raw versus temperature scaled, per source.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate import setup_fonts
from evaluate_shift import save, style_axes
import matplotlib as mpl

mpl.use("Agg")
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

MODES = ["emp", "sgr", "cov", "LTT Bonf", "LTT FS", "Chow raw", "Chow TS", "Conf LAC", "Conf APS"]
CORRUPTIONS = ["contrast", "gaussian_blur", "gaussian_noise", "pixelate"]
SPECIAL_METHODS = {"sr_base": "sr_base", "sr_dropout": "sr_dropout", "swa_maxprob": "swa"}


Table = dict[tuple[str, ...], dict[str, float]]


def read_table(path: Path, keys: tuple[str, ...] = ("criterion",)) -> Table:
    """Read a csv file into ``{key values: {column: float}}`` (empty cells become NaN), keeping the row order."""
    out: Table = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[tuple(row[k] for k in keys)] = {k: float(v) if v != "" else float("nan") for k, v in row.items() if k not in keys}
    return out


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=EXPERIMENT_DIR / "results")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: <results>/summary).")
    return p.parse_args()


def method_of(criterion: str) -> str:
    """Method of a criterion: its prefix before the first underscore (``sr_*`` and ``swa`` handled separately)."""
    return SPECIAL_METHODS.get(criterion, criterion.split("_", maxsplit=1)[0])


def plot_modes(options_dir: Path, path: Path, risks: tuple[float, ...] = (0.01, 0.03), flag: float = 0.01) -> None:
    """Heatmap of the mean test coverage per criterion (rows) and mode (columns), one panel per risk level.

    Cells whose mean violation share exceeds ``flag`` get a thick red border and the violation percentage under the
    coverage; modes that do not apply to a criterion are grey. Thin lines separate the method groups.

    Args:
        options_dir: Directory holding ``id_r{r}.csv`` of ``evaluate_options.py``.
        path: Output image path.
        risks: Risk levels r* of the panels.
        flag: Violation share above which a cell is marked.
    """
    tables = [read_table(options_dir / f"id_r{r}.csv") for r in risks]
    crits = [k[0] for k in tables[0]]
    n, m = len(crits), len(MODES)
    groups = [method_of(c) for c in crits]
    fig = plt.figure(figsize=(7.0 * len(risks) + 0.8, 11))
    grid = fig.add_gridspec(1, len(risks) + 1, width_ratios=[*([1.0] * len(risks)), 0.03])
    axes = []
    for k in range(len(risks)):
        axes.append(fig.add_subplot(grid[0, k], sharey=axes[0] if axes else None))
    cax = fig.add_subplot(grid[0, len(risks)])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    im = None
    for ax, r, tab in zip(axes, risks, tables, strict=True):
        cov = np.array([[tab[(c,)][f"{k} cov_mean"] for k in MODES] for c in crits])
        viol = np.array([[tab[(c,)][f"{k} viol_mean"] for k in MODES] for c in crits])
        im = ax.imshow(np.ma.masked_invalid(cov), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(n):
            for j in range(m):
                if np.isnan(cov[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888888")
                    continue
                color = "white" if cov[i, j] < 0.55 else "black"
                if viol[i, j] > flag:
                    ax.text(j, i - 0.14, f"{cov[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
                    ax.text(j, i + 0.27, f"viol {viol[i, j] * 100:.0f}%", ha="center", va="center", fontsize=5.5, color=color)
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e53935", lw=2.2))
                else:
                    ax.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center", fontsize=8, color=color)
        for i in range(1, n):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color="black", lw=0.8)
        ax.set_xticks(range(m), MODES, rotation=40, ha="right")
        ax.set_yticks(range(n), crits)
        ax.tick_params(length=0)
        ax.set_title(f"r* = {r:g}", fontweight="semibold")
        for side in ax.spines.values():
            side.set_visible(False)
        style_axes(ax)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("mean test coverage", fontweight="semibold")
    fig.text(0.5, 0.005, f"red border: violation share (test risk > r*) above {flag:g}", ha="center", va="bottom", fontsize=9)
    save(fig, path)


def main_scores(aurc: Table) -> dict[str, str]:
    """Best criterion (lowest clean AURC) per method, ordered by that AURC (best first)."""
    best: dict[str, str] = {}
    for (c,), row in sorted(aurc.items(), key=lambda kv: kv[1]["AURC x1000_mean"]):
        best.setdefault(method_of(c), c)
    return best


def shift_means(results: Path) -> Table:
    """Mean over the four corruptions of every shift column, keyed by ``(severity, criterion)``."""
    tabs = [read_table(results / "shift" / f"shift_{c}.csv", ("severity", "criterion")) for c in CORRUPTIONS]
    return {key: {col: float(np.mean([t[key][col] for t in tabs])) for col in row} for key, row in tabs[0].items()}


def plot_main_scores(results: Path, path: Path, best: dict[str, str]) -> None:
    """Four horizontal bar panels (clean AURC, SGR coverage, SVHN AUROC, SGR risk under severity 5)."""
    aurc = read_table(results / "shift" / "id_aurc.csv")
    thr = read_table(results / "shift" / "id_thresholds.csv", ("r*", "criterion"))
    svhn = read_table(results / "shift" / "ood_svhn.csv")
    sh = shift_means(results)
    methods = list(best)
    crits = [best[k] for k in methods]
    ypos = np.arange(len(methods))

    def col(tab: Table, keys: list[tuple[str, ...]], name: str) -> np.ndarray:
        return np.array([tab[k][name] for k in keys])

    plain = [(c,) for c in crits]
    panels = [
        ("clean AURC x 1000 (lower is better)", col(aurc, plain, "AURC x1000_mean"), col(aurc, plain, "AURC x1000_std"), None),
        ("SGR test coverage, r* = 0.02", col(thr, [("0.02", c) for c in crits], "sgr cov_mean"), col(thr, [("0.02", c) for c in crits], "sgr cov_std"), None),
        ("SVHN OOD AUROC", col(svhn, plain, "AUROC_mean"), col(svhn, plain, "AUROC_std"), None),
        ("SGR test risk, r* = 0.03, severity 5", col(sh, [("5", c) for c in crits], "r*=0.03 sgr risk_mean"), col(sh, [("5", c) for c in crits], "r*=0.03 sgr risk_std"), 0.03),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 0.42 * len(methods) + 2.2), sharey=True)
    for ax, (label, val, err, ref) in zip(axes, panels, strict=True):
        ax.barh(ypos, val, xerr=err, color="#1e88e5", ecolor="#444444", error_kw={"lw": 0.8}, height=0.7)
        if ref is not None:
            ax.axvline(ref, color="#e53935", ls="--", lw=1.2)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25, axis="x")
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos, [f"{k}  ({best[k]})" for k in methods])
    axes[0].invert_yaxis()
    axes[2].set_xlim(left=max(0.0, float(panels[2][1].min()) - 0.1))
    for ax in axes:
        style_axes(ax)
    save(fig, path)


def plot_shift_risk(results: Path, path: Path, best: dict[str, str]) -> None:
    """SGR test risk at r* = 0.03 against severity (0 = clean, then the severities present in the shift csv files) for the main score of every method."""
    thr = read_table(results / "shift" / "id_thresholds.csv", ("r*", "criterion"))
    sh = shift_means(results)
    sevs = sorted({int(k[0]) for k in sh})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (k, c) in enumerate(best.items()):
        ys = [thr[("0.03", c)]["sgr risk_mean"], *[sh[(str(s), c)]["r*=0.03 sgr risk_mean"] for s in sevs]]
        ax.plot([0, *sevs], ys, marker="o", ms=4, lw=1.6, color=cmap(i % 20), label=k)
    ax.axhline(0.03, color="black", ls="--", lw=1.0)
    ax.text(0.02, 0.0305, "r* = 0.03", fontsize=8, va="bottom")
    ax.set_xlabel("severity (0 = clean)")
    ax.set_ylabel("SGR test risk")
    ax.set_xticks([0, *sevs])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1)
    style_axes(ax)
    save(fig, path)


def plot_calibration(results: Path, path: Path) -> None:
    """Grouped bars per source: clean ECE and msr AURC, raw versus temperature scaled."""
    panels = [
        ("clean ECE", read_table(results / "calibration" / "clean_calibration.csv", ("source",)), "ECE raw", "ECE TS"),
        ("msr AURC x 1000", read_table(results / "calibration" / "ranking_aurc.csv", ("source",)), "msr raw", "msr TS"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (label, tab, raw, ts) in zip(axes, panels, strict=True):
        names = [k[0] for k in tab]
        x = np.arange(len(names))
        for dx, name, color, lab in [(-0.2, raw, "#9e9e9e", "raw"), (0.2, ts, "#1e88e5", "temperature scaled")]:
            ax.bar(x + dx, [tab[(s,)][f"{name}_mean"] for s in names], 0.4, yerr=[tab[(s,)][f"{name}_std"] for s in names], color=color, label=lab, error_kw={"lw": 0.8})
        ax.set_xticks(x, names, rotation=40, ha="right")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    for ax in axes:
        style_axes(ax)
    save(fig, path)


def main() -> None:
    """Write all summary figures."""
    args = parse_args()
    out = args.out or args.results / "summary"
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()
    plot_modes(args.results / "options", out / "modes.png")
    best = main_scores(read_table(args.results / "shift" / "id_aurc.csv"))
    print("Main score per method (lowest clean AURC):")
    for k, c in best.items():
        print(f"  {k:12s} -> {c}")
    plot_main_scores(args.results, out / "main_scores.png", best)
    plot_shift_risk(args.results, out / "shift_risk.png", best)
    plot_calibration(args.results, out / "calibration.png")
    print(f"Wrote modes.png, main_scores.png, shift_risk.png and calibration.png to {out}")


if __name__ == "__main__":
    main()
