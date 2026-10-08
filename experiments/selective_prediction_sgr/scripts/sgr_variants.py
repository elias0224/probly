"""SGR variants on the paper's model, next to Table 1 of Geifman and El-Yaniv (2017), arXiv 1705.08500.

Same seed x split protocol as ``paper_table.py``. Per r* every variant picks a threshold on the selection half; the
test half gives risk and coverage (coverage 0 if nothing is certified). Variants:

- ``sgr``: probly's ``SGRSelector`` (binary search, ``ceil(log2 m)`` steps, delta split over the steps).
- ``sgr_last_tested``: Algorithm 1 as printed: the threshold of the last tested step, whether it passed or not.
- ``binary_grid``: binary search over a fixed grid of ``grid`` coverages; fewer steps, so a larger delta per step.
- ``fixed_seq_{c0}``: fixed-sequence testing with the full delta: start at coverage ``c0`` and grow the accepted set
  in 1% steps while the bound stays below r*; stop at the first failure.
- ``sgr_delta_{d}``: probly's SGR with a larger delta (not the paper's 0.001); shows which delta the paper's
  coverages correspond to.

The first section checks the paper's own Table 1: the bound ``B*`` that its train risk and train coverage give on
5000 selection points with delta 0.001 / 13 (Lemma 3.1, the Clopper-Pearson bound).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from evaluate import PAPER, THIN_FONT, setup_fonts
from evaluate_shift import CRITERIA, load_dataset, sgr_selector_threshold
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sgr_experiment.metrics import apply_threshold, risk_bound  # noqa: E402
from sgr_experiment.utils import EXPERIMENT_DIR  # noqa: E402

ROWS = ["dropout_scratch_sr", "dropout_scratch_maxprob", "sr_dropout", "sr_base"]
# Table 1 of the paper: r*, train risk, train coverage, test risk, test coverage, bound
PAPER_TABLE = [
    (0.01, 0.0079, 0.7822, 0.0092, 0.7856, 0.0099),
    (0.02, 0.0160, 0.8482, 0.0149, 0.8466, 0.0199),
    (0.03, 0.0260, 0.8988, 0.0261, 0.8966, 0.0298),
    (0.04, 0.0362, 0.9348, 0.0380, 0.9318, 0.0399),
    (0.05, 0.0454, 0.9610, 0.0486, 0.9596, 0.0491),
    (0.06, 0.0526, 0.9778, 0.0572, 0.9784, 0.0600),
]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "sgr_variants")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--grid", type=int, default=64, help="Coverage grid size of binary_grid.")
    p.add_argument("--starts", type=float, nargs="+", default=[0.3, 0.5], help="Start coverages of fixed_seq.")
    p.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.1], help="Larger deltas of sgr_delta.")
    return p.parse_args()


def _prefix(crit: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted criterion, cumulative errors and the last index of each tie group."""
    order = np.argsort(crit, kind="stable")
    c = crit[order]
    cum = np.cumsum(loss[order])
    ends = np.flatnonzero(np.append(c[1:] != c[:-1], True))
    return c, cum, ends


def _bound_at(cum: np.ndarray, ends: np.ndarray, g: int, delta: float) -> float:
    """Bound of the accepted set that ends with tie group ``g``."""
    return risk_bound(round(cum[ends[g]]), int(ends[g]) + 1, delta)


def _group_at_coverage(ends: np.ndarray, n: int, cov: float) -> int:
    """Index of the smallest tie group whose accepted set covers at least ``cov`` (at least one point)."""
    return int(min(np.searchsorted(ends + 1, max(1, math.ceil(cov * n))), len(ends) - 1))


def sgr_last_tested(crit: np.ndarray, loss: np.ndarray, r: float, delta: float) -> tuple[float, float]:
    """Algorithm 1 as printed: binary search over tie groups, returns the last tested threshold and its bound."""
    c, cum, ends = _prefix(crit, loss)
    steps = max(1, math.ceil(math.log2(len(c))))
    lo, hi = -1, len(ends)
    mid, bound = 0, 1.0
    for _ in range(steps):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        bound = _bound_at(cum, ends, mid, delta / steps)
        if bound <= r + EPS:
            lo = mid
        else:
            hi = mid
    return float(c[ends[mid]]), bound


def binary_grid(crit: np.ndarray, loss: np.ndarray, r: float, delta: float, grid: int) -> tuple[float, float]:
    """Binary search over ``grid`` coverages ``1/grid, ..., 1`` with ``delta / ceil(log2 grid)`` per step."""
    c, cum, ends = _prefix(crit, loss)
    groups = [_group_at_coverage(ends, len(c), (i + 1) / grid) for i in range(grid)]
    steps = max(1, math.ceil(math.log2(grid)))
    lo, hi = -1, grid
    best = (-np.inf, 1.0)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        bound = _bound_at(cum, ends, groups[mid], delta / steps)
        if bound <= r + EPS:
            lo, best = mid, (float(c[ends[groups[mid]]]), bound)
        else:
            hi = mid
    return best


def fixed_sequence(crit: np.ndarray, loss: np.ndarray, r: float, delta: float, start: float) -> tuple[float, float]:
    """Fixed-sequence test with the full delta: coverages ``start, start + 0.01, ...`` until the first failure."""
    c, cum, ends = _prefix(crit, loss)
    best = (-np.inf, 1.0)
    for cov in np.arange(start, 1 + EPS, 0.01):
        g = _group_at_coverage(ends, len(c), cov)
        bound = _bound_at(cum, ends, g, delta)
        if bound > r + EPS:
            break
        best = (float(c[ends[g]]), bound)
    return best


def variants(args: argparse.Namespace) -> dict:
    """Name -> function ``(crit, loss, r) -> (threshold, bound)``."""
    d = args.delta
    out = {
        "sgr": lambda c, l, r: sgr_selector_threshold(c, l, r, d),
        "sgr_last_tested": lambda c, l, r: sgr_last_tested(c, l, r, d),
        f"binary_grid{args.grid}": lambda c, l, r: binary_grid(c, l, r, d, args.grid),
    }
    for s in args.starts:
        out[f"fixed_seq_{s:g}"] = lambda c, l, r, s=s: fixed_sequence(c, l, r, d, s)
    for dd in args.deltas:
        out[f"sgr_delta_{dd:g}"] = lambda c, l, r, dd=dd: sgr_selector_threshold(c, l, r, dd)
    return out


def evaluate(data: dict, rows: list[str], fns: dict, args: argparse.Namespace) -> dict[tuple, list[float]]:
    """Values per ``(criterion, variant, r*, column)`` over seeds x splits."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    labels = data["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    for c in rows:
        for crit, pred in data["units"][c]:
            crit = np.asarray(crit, dtype=np.float64)
            loss = (pred != labels).astype(np.float64)
            for perm in splits:
                sel, test = perm[:half], perm[half:]
                for r, *_ in PAPER:
                    for name, fn in fns.items():
                        thr, bound = fn(crit[sel], loss[sel], r)
                        certified = thr != -np.inf
                        sel_risk, sel_cov = apply_threshold(crit[sel], loss[sel], thr)
                        risk, cov = apply_threshold(crit[test], loss[test], thr)
                        key = (c, name, r)
                        acc[*key, "certified"].append(float(certified))
                        acc[*key, "sel_cov"].append(sel_cov)
                        acc[*key, "test_cov"].append(cov)
                        acc[*key, "sel_risk"].append(sel_risk if certified else np.nan)
                        acc[*key, "test_risk"].append(risk if certified else np.nan)
                        acc[*key, "bound"].append(bound if certified else np.nan)
                        acc[*key, "viol"].append(float(certified and risk > r + EPS))
    return acc


def mean(values: list[float]) -> float:
    """Mean over the finite values (NaN if there are none)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


COLUMNS = [
    ("certified", "certified", "pct"),
    ("sel_risk", "train risk", "f"),
    ("sel_cov", "train cov", "f"),
    ("test_risk", "test risk", "f"),
    ("test_cov", "test cov", "f"),
    ("bound", "bound", "f"),
    ("viol", "violations", "pct"),
]


def paper_check() -> list[str]:
    """Markdown table: the bound that the paper's own train numbers give with m = 5000 and delta 0.001 / 13."""
    m, steps = 5000, math.ceil(math.log2(5000))
    lines = [
        "## Check of the paper's Table 1",
        "",
        f"B* (Lemma 3.1) of the paper's train risk and train coverage on m = {m} selection points, delta 0.001 / {steps} "
        "as in Algorithm 1, and with the full delta 0.001.",
        "",
        "| r* | train risk | train cov | accepted | errors | paper bound | B*, delta/13 | B*, delta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r, tr, tc, _, _, b in PAPER_TABLE:
        n = round(tc * m)
        e = round(tr * n)
        lines.append(
            f"| {r:.2f} | {tr:.4f} | {tc:.4f} | {n} | {e} | {b:.4f} | {risk_bound(e, n, 0.001 / steps):.4f} | {risk_bound(e, n, 0.001):.4f} |"
        )
    return [*lines, ""]


def write(out: Path, acc: dict, rows: list[str], fns: dict, args: argparse.Namespace) -> None:
    """``table.md`` (per criterion one table, variants x r*) and ``table.csv``."""
    lines = [
        "# SGR variants (CIFAR-10, Table 1 protocol)",
        "",
        f"Seeds {' '.join(map(str, args.seeds))} x {args.n_splits} selection/test splits (5k/5k), delta {args.delta} "
        "unless the variant name says otherwise. Risk, bound and violations over certified splits, coverage over all "
        "splits (0 when nothing is certified). violations: share of certified splits with test risk > r*.",
        "",
        *paper_check(),
    ]
    for c in rows:
        lines += [f"## {CRITERIA[c]['label']} (`{c}`)", ""]
        for r, _, _, p_risk, p_cov, _ in PAPER_TABLE:
            lines += [
                f"### r* = {r:.2f} (paper: test risk {p_risk:.4f}, test cov {p_cov:.4f})",
                "",
                "| variant | " + " | ".join(h for _, h, _ in COLUMNS) + " |",
                "|" + "---|" * (len(COLUMNS) + 1),
            ]
            for name in fns:
                cells = []
                for k, _, kind in COLUMNS:
                    m = mean(acc[c, name, r, k])
                    cells.append("n/a" if not np.isfinite(m) else (f"{m * 100:.0f}%" if kind == "pct" else f"{m:.4f}"))
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["criterion", "variant", "r_star", *(f"{k}_{s}" for k, _, _ in COLUMNS for s in ("mean", "std"))])
        for c in rows:
            for name in fns:
                for r, *_ in PAPER:
                    vals = []
                    for k, _, _ in COLUMNS:
                        v = np.asarray(acc[c, name, r, k], dtype=float)
                        v = v[np.isfinite(v)]
                        vals += [f"{v.mean():.6g}", f"{v.std():.6g}"] if v.size else ["", ""]
                    w.writerow([c, name, r, *vals])


def plot(path: Path, acc: dict, rows: list[str], fns: dict) -> None:
    """Test coverage per variant against r*, one panel per criterion, with the paper's coverage."""
    has_fira = setup_fonts()
    rs = [r for r, *_ in PAPER]
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4), sharey=True, squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, c in zip(axes[0], rows, strict=True):
        ax.plot(rs, [row[4] for row in PAPER_TABLE], color="black", marker="o", lw=1.5, label="Paper")
        for i, name in enumerate(fns):
            ax.plot(rs, [mean(acc[c, name, r, "test_cov"]) for r in rs], color=colors(i), marker=".", lw=1.2, label=name)
        ax.set_title(CRITERIA[c]["label"], fontsize=10)
        ax.set_xlabel("Desired risk r*", fontweight="semibold")
        ax.grid(alpha=0.3)
        if has_fira:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontfamily(THIN_FONT)
    axes[0][0].set_ylabel("Test coverage", fontweight="semibold")
    axes[0][-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Evaluate the variants and write ``table.md``, ``table.csv`` and ``coverage.png``."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.runs, args.seeds, "cifar10")
    if data is None:
        msg = f"no cifar10 dumps for seeds {args.seeds} in {args.runs}/seed*/shift"
        raise SystemExit(msg)
    rows = [c for c in ROWS if data["units"].get(c)]
    fns = variants(args)
    acc = evaluate(data, rows, fns, args)
    write(args.out, acc, rows, fns, args)
    plot(args.out / "coverage.png", acc, rows, fns)
    print((args.out / "table.md").read_text())


if __name__ == "__main__":
    main()
