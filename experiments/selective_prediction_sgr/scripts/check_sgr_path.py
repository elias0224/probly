"""Trace the SGR binary search at r* to see why a criterion ends uncertified.

Uses the clean CIFAR-10 dumps and the same seed x 5k/5k split protocol as ``evaluate_shift.py``. Per unit (seed, or
the one ensemble) and criterion it prints:

- how many splits end uncertified;
- the errors among the k most confident predictions of the whole test set, and the selective risk at a few coverages;
- the size of the largest tie block among the 1000 most confident points (SGR accepts a tie block as a whole);
- the smallest Clopper-Pearson bound over all prefixes of the selection half at the search level ``delta / steps``,
  averaged over the splits: if it is above r*, no threshold can be certified, whatever the search does;
- averaged over the splits, the accepted count, errors and bound of the first tested threshold (the median).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_shift import load_dataset
import numpy as np
from scipy.stats import beta

from probly.selective_prediction._common import _binomial_upper_bound
from sgr_experiment.utils import EXPERIMENT_DIR

TOP_K = (50, 100, 250, 500, 1000, 2500)
COVERAGES = (0.1, 0.2, 0.3, 0.5, 0.7)
DEFAULT_CRITERIA = (
    "sr_base",
    "ddu_maxprob",
    "ddu_density",
    "sngp_maxprob",
    "sngp_ds",
    "sngp_long_maxprob",
    "sngp_long_ds",
    "sngp_scratch_maxprob",
    "sngp_scratch_ds",
    "dropout_scratch_sr",
    "dropout_scratch_maxprob",
    "dropout_scratch_epistemic",
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--criteria", nargs="+", default=list(DEFAULT_CRITERIA))
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to use (default: all with clean dumps).")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--delta", type=float, default=0.001)
    return p.parse_args()


def min_prefix_bound(scores: np.ndarray, errors: np.ndarray, level: float) -> float:
    """Smallest Clopper-Pearson upper bound over all acceptable prefixes (ends of tie blocks) of the ranking."""
    order = np.argsort(scores, kind="stable")
    ascending = scores[order]
    cum = np.cumsum(errors[order])
    ends = np.flatnonzero(np.r_[ascending[1:] != ascending[:-1], True])
    m, e = ends + 1, cum[ends]
    bounds = np.where(e >= m, 1.0, beta.ppf(1 - level, e + 1, m - e))
    return float(bounds.min())


def trace(scores: np.ndarray, errors: np.ndarray, risk: float, delta: float) -> list[tuple[int, int, float, bool]]:
    """Steps of the SGR binary search as (accepted, errors, bound, certified), mirroring ``SGRSelector.calibrate``."""
    n = scores.shape[0]
    order = np.argsort(scores, kind="stable")
    ascending = scores[order]
    cum = np.concatenate([[0.0], np.cumsum(errors[order])])
    steps = (n - 1).bit_length()
    level = delta / steps
    z_min, z_max = 1, n
    out = []
    for _ in range(steps):
        z = (z_min + z_max + 1) // 2
        accepted = int(np.searchsorted(ascending, ascending[n - z], side="right"))
        bound = _binomial_upper_bound(float(cum[accepted]), accepted, level)
        ok = bound < risk
        out.append((accepted, int(cum[accepted]), bound, ok))
        if ok:
            z_max = z
        else:
            z_min = z
    return out


def main() -> None:
    """Print the search statistics per unit and criterion on the clean test set."""
    args = parse_args()
    seeds = args.seeds or sorted(int(f.parent.parent.name[4:]) for f in args.runs.glob("seed*/shift/cifar10.npz"))
    data = load_dataset(args.runs, seeds, "cifar10")
    if data is None:
        msg = f"No complete clean dumps (shift/cifar10.npz) under {args.runs}."
        raise SystemExit(msg)
    labels = data["labels"]
    n = len(labels)
    half = n // 2
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    level = args.delta / (half - 1).bit_length()
    print(f"r* = {args.risk}, delta = {args.delta}, search level {level:.2e}; top-k errors on all {n} clean test points")
    for c in args.criteria:
        units = data["units"].get(c)
        if not units:
            print(f"{c}: not in the dumps, skipping.")
            continue
        for u, (crit, pred) in enumerate(units):
            loss = (pred != labels).astype(np.float64)
            order = np.argsort(crit, kind="stable")
            ranked = loss[order]
            top = " ".join(f"{k}:{int(ranked[:k].sum())}" for k in TOP_K)
            risks = " ".join(f"{cov:.0%}:{ranked[: int(cov * n)].mean():.4f}" for cov in COVERAGES)
            _, counts = np.unique(crit[order[:1000]], return_counts=True)
            paths = [trace(crit[p[:half]], loss[p[:half]], args.risk, args.delta) for p in splits]
            uncert = sum(not any(s[3] for s in path) for path in paths)
            best = np.mean([min_prefix_bound(crit[p[:half]], loss[p[:half]], level) for p in splits])
            first = np.array([path[0][:3] for path in paths], dtype=float).mean(0)
            print(
                f"{c} unit {u}: uncertified {uncert}/{len(splits)} | top-k errors {top} | risk at coverage {risks} | "
                f"largest tie in top 1000: {counts.max()} | min bound {best:.4f} | "
                f"median step: accepted {first[0]:.0f}, errors {first[1]:.1f}, bound {first[2]:.4f}"
            )


if __name__ == "__main__":
    main()
