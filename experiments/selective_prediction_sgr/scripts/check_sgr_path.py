"""Trace the SGR binary search at r* = 0.01 to see why a criterion ends uncertified.

Uses the same seeds and 5k/5k splits as ``evaluate.py``. Per seed and criterion it prints how many splits end
uncertified, the errors among the most confident predictions of the whole test set, and, averaged over the splits,
the accepted count, errors and Clopper-Pearson bound of the first tested threshold (the median of the selection half).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probly.selective_prediction._common import _binomial_upper_bound
from sgr_experiment.uncertainty import one_minus_max
from sgr_experiment.utils import EXPERIMENT_DIR

TOP_K = (50, 100, 250, 500, 1000, 2500)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--delta", type=float, default=0.001)
    return p.parse_args()


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
    """Print the search statistics for sr_base and sr_dropout per seed."""
    args = parse_args()
    dirs = sorted(
        (f.parent for f in args.runs.glob("seed*/predictions_dropout.npz") if (f.parent / "predictions_base.npz").exists()),
        key=lambda p: int(p.name[4:]),
    )
    n = len(np.load(dirs[0] / "predictions_dropout.npz")["labels"])
    rng = np.random.default_rng(args.split_seed)
    splits = [rng.permutation(n) for _ in range(args.n_splits)]
    half = n // 2
    print(f"r* = {args.risk}, delta = {args.delta}; top-k errors on all {n} test points")
    for seed_dir in dirs:
        for name, crit_name in (("predictions_base.npz", "sr_base"), ("predictions_dropout.npz", "sr_dropout")):
            d = np.load(seed_dir / name)
            crit = one_minus_max(d["softmax"])
            loss = (d["softmax"].argmax(1) != d["labels"]).astype(np.float64)
            ranked = loss[np.argsort(crit, kind="stable")]
            top = " ".join(f"{k}:{int(ranked[:k].sum())}" for k in TOP_K)
            paths = [trace(crit[p[:half]], loss[p[:half]], args.risk, args.delta) for p in splits]
            uncert = sum(not any(s[3] for s in path) for path in paths)
            first = np.array([path[0][:3] for path in paths], dtype=float).mean(0)
            print(
                f"{seed_dir.name} {crit_name:10s} uncertified {uncert}/{len(splits)} | top-k errors {top} | "
                f"median step: accepted {first[0]:.0f}, errors {first[1]:.1f}, bound {first[2]:.4f}"
            )


if __name__ == "__main__":
    main()
