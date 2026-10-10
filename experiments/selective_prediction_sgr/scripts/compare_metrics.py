"""Cross-check the experiment's selective prediction metrics against ``probly.metrics.selective_prediction``.

For every criterion and seed of the clean CIFAR-10 dumps, compute AURC, AUGRC and the coverage at a few risks
with both implementations, on the full 10k test set. The risk-coverage curves and ``coverage_at_risk`` must agree
exactly. The areas differ by construction:

- The experiment's areas are the right Riemann mean over the ``n`` acceptance steps, and every instance of a
  tie run gets the run's point (a step).
- probly integrates the curve with the trapezoid rule from an endpoint at coverage 0. Without ties, this is the
  experiment's area minus ``(r_n - r_1) / (2n)`` (AURC) or ``r_n / (2n)`` (AUGRC). A tie run is interpolated
  linearly, so large tie runs (e.g. a saturated softmax) widen the gap.

The table reports the fraction of tied instances next to the gaps, and the pairs of criteria the two
implementations order differently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_shift import load_dataset
import numpy as np
from sgr_experiment import metrics as ex
from sgr_experiment.utils import EXPERIMENT_DIR

from probly.metrics import selective_prediction as pm

RISKS = (0.01, 0.03, 0.05)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=EXPERIMENT_DIR / "runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return p.parse_args()


def tie_fraction(criterion: np.ndarray) -> float:
    """Fraction of instances that share their criterion value with another instance."""
    _, counts = np.unique(criterion, return_counts=True)
    return float(counts[counts > 1].sum() / criterion.size)


def compare(criterion: np.ndarray, loss: np.ndarray) -> dict[str, float]:
    """Both implementations' metrics on one unit, plus the largest curve and coverage mismatch."""
    c_ex, r_ex = ex.risk_coverage_curve(criterion, loss)
    c_pm, r_pm, _ = pm.risk_coverage_curve(criterion, loss)
    points = np.unique(np.stack([c_pm[1:], r_pm[1:]]), axis=1)
    curve_gap = float(np.abs(points - np.stack([c_ex, r_ex])).max()) if points.shape == (2, c_ex.size) else np.inf
    cov_gap = max(abs(ex.coverage_at_risk(criterion, loss, r) - pm.coverage_at_risk(criterion, loss, r)) for r in RISKS)
    return {
        "aurc_ex": ex.aurc(criterion, loss),
        "aurc_pm": pm.aurc(criterion, loss),
        "augrc_ex": ex.augrc(criterion, loss),
        "augrc_pm": pm.augrc(criterion, loss),
        "ties": tie_fraction(criterion),
        "curve_gap": curve_gap,
        "cov_gap": cov_gap,
    }


def order_swaps(a: dict[str, float], b: dict[str, float], tol: float = 1e-6) -> list[str]:
    """Pairs of criteria that ``a`` and ``b`` order differently; near-equal pairs count as ties, not swaps."""
    names = list(a)
    swaps = []
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            da, db = a[x] - a[y], b[x] - b[y]
            if abs(da) > tol and abs(db) > tol and np.sign(da) != np.sign(db):
                swaps.append(f"{x} / {y}")
    return swaps


def main() -> None:
    """Print one row per criterion and the ranking agreement."""
    args = parse_args()
    data = load_dataset(args.runs, args.seeds, "cifar10")
    if data is None:
        msg = f"no cifar10 dumps for seeds {args.seeds} in {args.runs}/seed*/shift"
        raise SystemExit(msg)
    labels = data["labels"]
    rows: dict[str, dict[str, float]] = {}
    for c, units in data["units"].items():
        if not units:
            continue
        per_unit = [compare(crit, (pred != labels).astype(np.float64)) for crit, pred in units]
        rows[c] = {k: float(np.mean([u[k] for u in per_unit])) for k in per_unit[0]}
        rows[c]["curve_gap"] = max(u["curve_gap"] for u in per_unit)
        rows[c]["cov_gap"] = max(u["cov_gap"] for u in per_unit)

    cols = ["ties", "AURC ex", "AURC pm", "gap", "AUGRC ex", "AUGRC pm", "gap", "curve", "cov@r"]
    widths = [6, 8, 8, 7, 8, 8, 7, 7, 7]
    head = f"{'criterion':28s} " + " ".join(f"{c:>{w}s}" for c, w in zip(cols, widths, strict=True))
    print("x1000 for the areas; gap = ex - pm; curve / cov@r = largest mismatch over seeds")
    print(head)
    print("-" * len(head))
    for c, v in rows.items():
        print(
            f"{c:28s} {v['ties']:6.1%} {v['aurc_ex'] * 1e3:8.3f} {v['aurc_pm'] * 1e3:8.3f} "
            f"{(v['aurc_ex'] - v['aurc_pm']) * 1e3:7.3f} {v['augrc_ex'] * 1e3:8.3f} {v['augrc_pm'] * 1e3:8.3f} "
            f"{(v['augrc_ex'] - v['augrc_pm']) * 1e3:7.3f} {v['curve_gap']:7.1e} {v['cov_gap']:7.1e}"
        )
    for metric in ("aurc", "augrc"):
        ours = {c: v[f"{metric}_ex"] for c, v in rows.items()}
        swaps = order_swaps(ours, {c: v[f"{metric}_pm"] for c, v in rows.items()})
        print(f"{metric.upper()} ranking: {'identical' if not swaps else 'swapped pairs: ' + ', '.join(swaps)}")


if __name__ == "__main__":
    main()
