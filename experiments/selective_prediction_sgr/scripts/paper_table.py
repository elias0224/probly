"""Reproduction of Table 1 of Geifman and El-Yaniv (2017), arXiv 1705.08500, with the paper's model.

The paper's model is ``dropout_scratch`` (VGG-16 with fc dropout 0.5, trained from scratch for 250 epochs); its
criterion is SR, one minus the max of the deterministic eval-mode softmax (``dropout_scratch_sr``). MC dropout of the
same model (``dropout_scratch_maxprob``) and the base and fine-tuned dropout models are references.

Same protocol as ``evaluate_shift.py``: per seed, ``n_splits`` permutations of the 10k CIFAR-10 test set; the first
half ("train" in the paper's table) fits probly's ``SGRSelector`` at each r*, the second half is the test half. A split
where SGR certifies nothing has coverage 0 and no risk; the risk and bound columns average over the certified
splits only. ``cov @ paper risk`` is the largest test-half coverage whose risk is at most the paper's test risk.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from evaluate import PAPER, PAPER_ACC, THIN_FONT, setup_fonts
from evaluate_shift import CRITERIA, load_dataset, sgr_selector_threshold
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probly.metrics.selective_prediction import coverage_at_risk  # noqa: E402
from sgr_experiment.metrics import apply_threshold  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

ROWS = ["dropout_scratch_sr", "dropout_scratch_maxprob", "sr_dropout", "sr_base"]
COLUMNS = [
    ("certified", "certified"),
    ("sel_risk", "train risk"),
    ("sel_cov", "train cov"),
    ("test_risk", "test risk"),
    ("test_cov", "test cov"),
    ("bound", "bound"),
    ("cov_at_paper", "cov @ paper risk"),
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "paper")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001, help="Confidence parameter of the SGR bound.")
    return p.parse_args()


def evaluate(data: dict, rows: list[str], n_splits: int, split_seed: int, delta: float) -> dict[tuple, list[float]]:
    """Values per ``(criterion, r*, column)`` over seeds x splits, plus ``(criterion, "acc")``."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    labels = data["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(split_seed)
    splits = [rng.permutation(n) for _ in range(n_splits)]
    for c in rows:
        for crit, pred in data["units"][c]:
            loss = (pred != labels).astype(np.float64)
            acc[c, "acc"].append(1 - loss.mean())
            for perm in splits:
                sel, test = perm[:half], perm[half:]
                for r, paper_risk, _ in PAPER:
                    thr, bound = sgr_selector_threshold(crit[sel], loss[sel], r, delta)
                    certified = thr != -np.inf
                    sel_risk, sel_cov = apply_threshold(crit[sel], loss[sel], thr)
                    test_risk, test_cov = apply_threshold(crit[test], loss[test], thr)
                    acc[c, r, "certified"].append(float(certified))
                    acc[c, r, "sel_cov"].append(sel_cov)
                    acc[c, r, "test_cov"].append(test_cov)
                    acc[c, r, "sel_risk"].append(sel_risk if certified else np.nan)
                    acc[c, r, "test_risk"].append(test_risk if certified else np.nan)
                    acc[c, r, "bound"].append(bound if certified else np.nan)
                    acc[c, r, "cov_at_paper"].append(coverage_at_risk(crit[test], loss[test], paper_risk))
    return acc


def mean(values: list[float]) -> float:
    """Mean over the finite values (NaN if there are none)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def cell(values: list[float], col: str) -> str:
    """Markdown cell of one column."""
    m = mean(values)
    if not np.isfinite(m):
        return "n/a"
    if col == "certified":
        return f"{m * 100:.0f}%"
    v = np.asarray(values, dtype=float)
    return f"{m:.4f} +- {v[np.isfinite(v)].std():.4f}"


def write_tables(out: Path, acc: dict, rows: list[str], n_units: dict[str, int], args: argparse.Namespace) -> None:
    """Markdown table per criterion with the paper's numbers next to it, and one long CSV."""
    lines = [
        "# Table 1 reproduction (Geifman and El-Yaniv 2017, CIFAR-10, SR)",
        "",
        f"Seeds {' '.join(map(str, args.seeds))} x {args.n_splits} selection/test splits (5k/5k), delta {args.delta}. "
        "train = selection half; risk and bound columns average over the certified splits, coverage columns over "
        "all splits (0 when SGR certifies nothing). cov @ paper risk: largest test-half coverage at risk <= the "
        "paper's test risk.",
        "",
        f"Paper accuracy {PAPER_ACC * 100:.2f}%.",
        "",
    ]
    for c in rows:
        lines += [
            f"## {CRITERIA[c]['label']} (`{c}`)",
            "",
            f"Accuracy {mean(acc[c, 'acc']) * 100:.2f}% over {n_units[c]} model(s).",
            "",
            "| r* | " + " | ".join(h for _, h in COLUMNS) + " | paper test risk | paper test cov |",
            "|" + "---|" * (len(COLUMNS) + 3),
        ]
        for r, p_risk, p_cov in PAPER:
            cells = [cell(acc[c, r, k], k) for k, _ in COLUMNS]
            lines.append(f"| {r:.2f} | " + " | ".join(cells) + f" | {p_risk:.4f} | {p_cov:.4f} |")
        lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "r_star", *(f"{k}_{s}" for k, _ in COLUMNS for s in ("mean", "std")), "paper_test_risk", "paper_test_cov"])
        for c in rows:
            for r, p_risk, p_cov in PAPER:
                vals = []
                for k, _ in COLUMNS:
                    v = np.asarray(acc[c, r, k], dtype=float)
                    v = v[np.isfinite(v)]
                    vals += [f"{v.mean():.6g}", f"{v.std():.6g}"] if v.size else ["", ""]
                w.writerow([c, r, *vals, p_risk, p_cov])


def plot(path: Path, acc: dict, rows: list[str]) -> None:
    """Test coverage of SGR and coverage at the paper's risk against r*, with the paper's coverage."""
    has_fira = setup_fonts()
    rs = [r for r, _, _ in PAPER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in zip(axes, ("test_cov", "cov_at_paper"), ("SGR test coverage", "Test coverage at the paper's risk"), strict=True):
        ax.plot(rs, [p for _, _, p in PAPER], color="black", marker="o", lw=1.5, label="Paper (SR)")
        for c in rows:
            st = CRITERIA[c]
            ax.plot(rs, [mean(acc[c, r, col]) for r in rs], color=st["color"], ls=st["ls"], marker=".", lw=1.5, label=st["label"])
        ax.set_title(title)
        ax.set_xlabel("Desired risk r*", fontweight="semibold")
        ax.grid(alpha=0.3)
        if has_fira:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontfamily(THIN_FONT)
    axes[0].set_ylabel("Coverage", fontweight="semibold")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Evaluate, write ``table.md``, ``table.csv`` and ``coverage.png``."""
    args = parse_args()
    data = load_dataset(args.runs, args.seeds, "cifar10")
    if data is None:
        msg = f"no cifar10 dumps for seeds {args.seeds} in {args.runs}/seed*/shift"
        raise SystemExit(msg)
    rows = [c for c in ROWS if data["units"].get(c)]
    missing = sorted(set(ROWS) - set(rows))
    if missing:
        print(f"skipped (no dumps for every seed): {', '.join(missing)}")
    args.out.mkdir(parents=True, exist_ok=True)
    acc = evaluate(data, rows, args.n_splits, args.split_seed, args.delta)
    write_tables(args.out, acc, rows, {c: len(data["units"][c]) for c in rows}, args)
    plot(args.out / "coverage.png", acc, rows)
    print((args.out / "table.md").read_text())


if __name__ == "__main__":
    main()
