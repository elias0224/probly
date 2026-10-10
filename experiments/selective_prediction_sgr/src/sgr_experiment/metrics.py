"""Selective prediction helpers on top of probly's metrics. Low criterion values are accepted first."""

from __future__ import annotations

import math

import numpy as np
from probly.metrics.selective_prediction import aurc, risk_coverage_curve
from scipy import stats

_EPS = 1e-12


def threshold_for_risk(criterion: np.ndarray, losses: np.ndarray, risk: float) -> float:
    """Criterion threshold (accept if ``criterion <= threshold``) with the largest coverage at risk <= ``risk``.

    Uses the curve of :func:`probly.metrics.selective_prediction.risk_coverage_curve`, so the coverage of the returned
    threshold equals :func:`probly.metrics.selective_prediction.coverage_at_risk`.

    Returns ``-inf`` (accept nothing) if no threshold attains the risk.
    """
    _, curve_risk, thresholds = risk_coverage_curve(np.asarray(criterion, dtype=np.float64), np.asarray(losses))
    feasible = np.flatnonzero(curve_risk[1:] <= risk)  # skip the endpoint at coverage 0
    if feasible.size == 0:
        return -np.inf
    return float(thresholds[1:][feasible[-1]])  # coverage is non-decreasing in the index


def apply_threshold(criterion: np.ndarray, losses: np.ndarray, threshold: float) -> tuple[float, float]:
    """Selective risk and coverage obtained by accepting ``criterion <= threshold``.

    Returns:
        Tuple ``(risk, coverage)``; the risk is NaN if nothing is accepted.
    """
    accepted = np.asarray(criterion) <= threshold
    coverage = float(accepted.mean())
    risk = float(np.asarray(losses)[accepted].mean()) if accepted.any() else float("nan")
    return risk, coverage


def e_aurc(criterion: np.ndarray, losses: np.ndarray) -> float:
    """Excess AURC: the AURC minus the AURC of the optimal ranking (lowest losses first) for the same losses."""
    losses = np.asarray(losses, dtype=np.float64)
    optimal = np.argsort(np.argsort(losses, kind="stable"), kind="stable")
    return aurc(criterion, losses) - aurc(optimal, losses)


def risk_bound(num_errors: int, num_accepted: int, delta: float) -> float:
    """Upper confidence bound ``B*`` on the true risk: ``P(Bin(num_accepted, b) <= num_errors) = delta``.

    This is the Clopper-Pearson style bound of Geifman and El-Yaniv (2017), ``beta.ppf(1 - delta, k + 1, m - k)``.
    """
    if num_errors >= num_accepted:
        return 1.0
    return float(stats.beta.ppf(1 - delta, num_errors + 1, num_accepted - num_errors))


# probly is getting an SGRSelector / CoverageSelector in an upcoming PR; this reference implementation is meant to
# cross-check it.
def sgr_threshold(
    criterion: np.ndarray, losses: np.ndarray, r_star: float, delta: float = 0.001
) -> tuple[float, float]:
    """Threshold of Algorithm 1 (SGR) of Geifman and El-Yaniv (2017), arXiv 1705.08500.

    Binary search over the size of the accepted set (instances sorted by ascending criterion, tie groups accepted
    together) for ``ceil(log2 m)`` steps. Each step tests the bound ``B*`` with ``delta' = delta / ceil(log2 m)``, so
    with probability at least ``1 - delta`` the true risk of the returned set is at most ``r_star``.

    Args:
        criterion: Uncertainty per instance, shape ``(m,)``; lower means more confident.
        losses: Zero-one loss per instance, shape ``(m,)``.
        r_star: Desired risk.
        delta: Confidence parameter of the guarantee.

    Returns:
        Tuple ``(threshold, bound)``: accept if ``criterion <= threshold``, and ``B*`` of the returned accepted set.
        The threshold is ``-inf`` (accept nothing) and the bound 1 if even the smallest set fails.
    """
    criterion = np.asarray(criterion, dtype=np.float64)
    order = np.argsort(criterion, kind="stable")
    c = criterion[order]
    cum_err = np.cumsum(np.asarray(losses, dtype=np.float64)[order])
    ends = np.flatnonzero(np.append(c[1:] != c[:-1], True))  # index of the last instance of each tie group
    steps = max(1, math.ceil(math.log2(len(c))))
    delta_step = delta / steps
    lo, hi = -1, len(ends)  # ends[lo] is feasible (or the empty set), ends[hi] is not (or past the end)
    best_bound = 1.0
    for _ in range(steps):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        m_acc = int(ends[mid]) + 1
        bound = risk_bound(round(cum_err[ends[mid]]), m_acc, delta_step)
        if bound <= r_star + _EPS:
            lo, best_bound = mid, bound
        else:
            hi = mid
    if lo < 0:
        return -np.inf, 1.0
    return float(c[ends[lo]]), best_bound


def auroc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a ``positive`` score exceeds a ``negative`` one (ties count one half), via ranks."""
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    ranks = stats.rankdata(np.concatenate([negative, positive]))
    n_neg, n_pos = len(negative), len(positive)
    return float((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_neg * n_pos))
