"""Exact risk-coverage helpers (no binning). Low criterion values are accepted first."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def risk_coverage_curve(criterion: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact risk-coverage curve.

    Instances are accepted in ascending order of the criterion. Instances with equal criterion are accepted
    together, so the curve has one point per distinct criterion value.

    Args:
        criterion: Uncertainty per instance, shape ``(n,)``; lower means more confident.
        losses: Loss per instance, shape ``(n,)``.

    Returns:
        Tuple ``(coverages, risks)``, each of shape ``(m,)`` with ``m`` the number of distinct criterion values.
    """
    criterion = np.asarray(criterion, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    order = np.argsort(criterion, kind="stable")
    c = criterion[order]
    cum = np.cumsum(losses[order])
    last_of_group = np.flatnonzero(np.append(c[1:] != c[:-1], True))
    accepted = last_of_group + 1
    return accepted / len(c), cum[last_of_group] / accepted


def _best_point(criterion: np.ndarray, losses: np.ndarray, risk: float) -> tuple[float, float]:
    """Return (coverage, threshold) of the largest feasible point, or (0, -inf)."""
    criterion = np.asarray(criterion, dtype=np.float64)
    cov, rsk = risk_coverage_curve(criterion, losses)
    feasible = np.flatnonzero(rsk <= risk + _EPS)
    if feasible.size == 0:
        return 0.0, -np.inf
    j = feasible[-1]  # coverage increases with the index
    return float(cov[j]), float(np.unique(criterion)[j])


def coverage_at_risk(criterion: np.ndarray, losses: np.ndarray, risk: float) -> float:
    """Largest coverage whose selective risk is at most ``risk``; 0 if there is none.

    Every threshold is checked because the risk-coverage curve is not monotone.
    """
    return _best_point(criterion, losses, risk)[0]


def threshold_for_risk(criterion: np.ndarray, losses: np.ndarray, risk: float) -> float:
    """Criterion threshold (accept if ``criterion <= threshold``) with the largest coverage at risk <= ``risk``.

    Returns ``-inf`` (accept nothing) if no threshold attains the risk.
    """
    return _best_point(criterion, losses, risk)[1]


def apply_threshold(criterion: np.ndarray, losses: np.ndarray, threshold: float) -> tuple[float, float]:
    """Selective risk and coverage obtained by accepting ``criterion <= threshold``.

    Returns:
        Tuple ``(risk, coverage)``; the risk is NaN if nothing is accepted.
    """
    accepted = np.asarray(criterion) <= threshold
    coverage = float(accepted.mean())
    risk = float(np.asarray(losses)[accepted].mean()) if accepted.any() else float("nan")
    return risk, coverage
