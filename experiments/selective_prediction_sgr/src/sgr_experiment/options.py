"""Alternative accept/abstain rules for selective prediction: Learn-then-Test thresholds and conformal singletons.

Everything here is pure numpy (plus scipy for the binomial tail) and works on one criterion where low values are
accepted first, as in :mod:`sgr_experiment.metrics`.

References:
    Angelopoulos, Bates, Candes, Jordan, Lei (2021), Learn then Test: Calibrating predictive algorithms to achieve
    risk control. Sadinle, Lei, Wasserman (2019), Least ambiguous set-valued classifiers with bounded error levels.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

_EPS = 1e-12


def candidate_thresholds(crit: np.ndarray, grid_size: int = 100) -> np.ndarray:
    """Label-free candidate thresholds: quantiles of the criterion at coverage levels ``1/G, 2/G, ..., 1``.

    The ``j``-th candidate (1-based) is the criterion value at rank ``ceil(j * n / G)`` of the sorted criterion, so
    accepting ``crit <= threshold`` covers about ``j / G`` of the data (more if there are ties).

    Args:
        crit: Criterion per instance, shape ``(n,)``; lower means more confident.
        grid_size: Number of candidates ``G``.

    Returns:
        Non-decreasing array of shape ``(G,)``; the last entry is the maximum of ``crit``.
    """
    c = np.sort(np.asarray(crit, dtype=np.float64))
    n = len(c)
    ranks = np.ceil(np.arange(1, grid_size + 1) * n / grid_size - 1e-9).astype(np.int64)
    return c[np.clip(ranks - 1, 0, n - 1)]


def binomial_p_values(k: np.ndarray, m: np.ndarray, risk: float) -> np.ndarray:
    """P-values of ``H0: selective risk >= risk`` for ``k`` errors among ``m`` accepted instances.

    The p-value is the lower binomial tail ``P(Bin(m, risk) <= k)``; it is 1 if ``m = 0``.

    Args:
        k: Number of errors among the accepted instances, any shape.
        m: Number of accepted instances, same shape as ``k``.
        risk: Desired risk ``r*``.

    Returns:
        Array of p-values with the shape of ``k``.
    """
    return np.asarray(stats.binom.cdf(np.asarray(k), np.asarray(m), risk), dtype=np.float64)


def _candidate_p_values(
    crit: np.ndarray, loss: np.ndarray, risk: float, grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate thresholds and the binomial p-value of each, vectorized with one sort."""
    crit = np.asarray(crit, dtype=np.float64)
    order = np.argsort(crit, kind="stable")
    sorted_crit = crit[order]
    cum_err = np.concatenate([[0.0], np.cumsum(np.asarray(loss, dtype=np.float64)[order])])
    thresholds = candidate_thresholds(sorted_crit, grid_size)
    m = np.searchsorted(sorted_crit, thresholds, side="right")  # accepted count, ties included
    k = np.rint(cum_err[m])
    return thresholds, binomial_p_values(k, m, risk)


def ltt_bonferroni_threshold(
    crit: np.ndarray, loss: np.ndarray, risk: float, delta: float, grid_size: int = 100
) -> float:
    """Learn-then-Test threshold with Bonferroni correction (Angelopoulos et al., 2021).

    Each candidate of :func:`candidate_thresholds` tests ``H0: selective risk >= risk`` with the binomial p-value;
    it is certified if ``p <= delta / grid_size``. No monotonicity is assumed, so the certified candidate with the
    largest coverage is returned. With probability at least ``1 - delta`` the true risk of the returned rule is at
    most ``risk`` (the candidates are label-free, so the family is fixed before the labels are seen).

    Args:
        crit: Criterion on the calibration data, shape ``(n,)``; lower means more confident.
        loss: Zero-one loss per instance, shape ``(n,)``.
        risk: Desired risk ``r*``.
        delta: Confidence parameter of the guarantee.
        grid_size: Number of candidates ``G``.

    Returns:
        The threshold (accept if ``crit <= threshold``), or ``-inf`` if no candidate is certified.
    """
    thresholds, p = _candidate_p_values(crit, loss, risk, grid_size)
    certified = np.flatnonzero(p <= delta / grid_size)
    if certified.size == 0:
        return -math.inf
    return float(thresholds[certified[-1]])  # thresholds are non-decreasing, so coverage is too


def ltt_fixed_sequence_threshold(
    crit: np.ndarray, loss: np.ndarray, risk: float, delta: float, grid_size: int = 100, starts: int = 10
) -> float:
    """Learn-then-Test threshold with multi-start fixed-sequence testing (Angelopoulos et al., 2021).

    ``starts`` sequences start at evenly spaced candidates (coverage ``1/K, 2/K, ..., 1`` for ``K`` starts), each
    walks toward higher coverage at level ``delta / K`` and stops at the first candidate that is not certified.
    The union of the certified candidates is controlled at level ``delta`` (Bonferroni over the sequences); the
    one with the largest coverage is returned.

    Args:
        crit: Criterion on the calibration data, shape ``(n,)``; lower means more confident.
        loss: Zero-one loss per instance, shape ``(n,)``.
        risk: Desired risk ``r*``.
        delta: Confidence parameter of the guarantee.
        grid_size: Number of candidates ``G``.
        starts: Number of sequences ``K``.

    Returns:
        The threshold (accept if ``crit <= threshold``), or ``-inf`` if no candidate is certified.
    """
    thresholds, p = _candidate_p_values(crit, loss, risk, grid_size)
    passed = p <= delta / starts
    # first_fail[i]: index of the first uncertified candidate at or after i (grid_size if none)
    idx = np.where(passed, grid_size, np.arange(grid_size))
    first_fail = np.minimum.accumulate(idx[::-1])[::-1]
    start_idx = (np.arange(1, starts + 1) * grid_size) // starts - 1
    start_idx = np.clip(start_idx, 0, grid_size - 1)
    last = first_fail[start_idx] - 1  # last certified candidate of each sequence (< start if the first fails)
    valid = last >= start_idx
    if not valid.any():
        return -math.inf
    return float(thresholds[last[valid].max()])


def _prefix_errors(crit: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted criterion, cumulative errors and the 1-based accepted count at the end of each tie group."""
    crit = np.asarray(crit, dtype=np.float64)
    order = np.argsort(crit, kind="stable")
    c = crit[order]
    cum = np.cumsum(np.asarray(loss, dtype=np.float64)[order])
    ends = np.flatnonzero(np.append(c[1:] != c[:-1], True))
    return c[ends], cum[ends], ends + 1


def _bounds(cum: np.ndarray, m: np.ndarray, delta: float) -> np.ndarray:
    """Clopper-Pearson bound ``risk_bound`` for ``cum`` errors among ``m`` accepted, vectorized."""
    k = np.rint(cum)
    out = np.ones(len(k))
    ok = k < m
    out[ok] = stats.beta.ppf(1 - delta, k[ok] + 1, m[ok] - k[ok])
    return out


def _groups_at_coverages(m: np.ndarray, n: int, coverages: np.ndarray) -> np.ndarray:
    """Smallest tie group whose accepted set covers at least each coverage (at least one point)."""
    need = np.maximum(1, np.ceil(coverages * n - 1e-9))
    return np.minimum(np.searchsorted(m, need), len(m) - 1)


def fixed_sequence_split_threshold(
    crit: np.ndarray,
    loss: np.ndarray,
    risk: float,
    delta: float,
    rng: np.random.Generator,
    pilot: float = 0.2,
    margin: float = 0.1,
    step: float = 0.01,
) -> float:
    """Fixed-sequence threshold whose start is chosen on an independent pilot split.

    The calibration data is split at random into a pilot part (fraction ``pilot``) and a test part. On the pilot
    part the plug-in start is the largest coverage ``c`` whose pilot risk would be certified by the test part, i.e.
    ``risk_bound(round(pilot_risk(c) * c * n_test), round(c * n_test), delta) <= risk``; the start is
    ``max(step, c - margin)``, or ``step`` if no ``c`` qualifies. The test part alone then runs fixed-sequence
    testing at the full ``delta`` over the coverages ``start, start + step, ...`` (at most 1) and stops at the first
    coverage whose bound exceeds ``risk``.

    The order of the hypotheses depends only on the pilot part, which is disjoint from the test part, so
    fixed-sequence testing on the test part controls the family-wise error at ``delta``: with probability at least
    ``1 - delta`` the true selective risk of the returned threshold is at most ``risk``. ``pilot``, ``margin`` and
    ``step`` must be fixed a priori, not tuned on the test part.

    Args:
        crit: Criterion on the calibration data, shape ``(n,)``; lower means more confident.
        loss: Zero-one loss per instance, shape ``(n,)``.
        risk: Desired risk ``r*``.
        delta: Confidence parameter of the guarantee.
        rng: Generator drawing the pilot/test split.
        pilot: Fraction of the data used for the pilot part.
        margin: Coverage subtracted from the plug-in start to make the start conservative.
        step: Coverage step of the sequence.

    Returns:
        The threshold (accept if ``crit <= threshold``) of the last passed coverage, or ``-inf`` if the start fails.
    """
    crit = np.asarray(crit, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    perm = rng.permutation(len(crit))
    n_pilot = max(1, round(pilot * len(crit)))
    pi, ti = perm[:n_pilot], perm[n_pilot:]
    n_test = len(ti)
    pc, pcum, pm = _prefix_errors(crit[pi], loss[pi])
    p_cov = pm / n_pilot
    certified = _bounds(pcum / pm * p_cov * n_test, np.rint(p_cov * n_test), delta) <= risk + _EPS
    start = max(step, float(p_cov[np.flatnonzero(certified)[-1]]) - margin) if certified.any() else step
    tc, tcum, tm = _prefix_errors(crit[ti], loss[ti])
    covs = np.arange(start, 1 + 1e-9, step)
    groups = _groups_at_coverages(tm, n_test, covs)
    failed = np.flatnonzero(_bounds(tcum[groups], tm[groups], delta) > risk + _EPS)
    last = (failed[0] if failed.size else len(covs)) - 1
    return float(tc[groups[last]]) if last >= 0 else -math.inf


def conformal_qhat(cal_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile ``ceil((n + 1) (1 - alpha)) / n`` of the calibration scores.

    Args:
        cal_scores: Nonconformity scores of the calibration data, shape ``(n,)``.
        alpha: Miscoverage level.

    Returns:
        The ``ceil((n + 1) (1 - alpha))``-th smallest score, or ``inf`` if that rank exceeds ``n``.
    """
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    rank = math.ceil((n + 1) * (1 - alpha) - 1e-9)
    if rank > n:
        return math.inf
    return float(s[max(rank, 1) - 1])


def top_two(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Largest and second largest probability per row of ``(n, classes)`` probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    part = np.partition(p, -2, axis=-1)
    return part[..., -1], part[..., -2]


def lac_singleton_from_top_two(top1: np.ndarray, top2: np.ndarray, qhat: float) -> np.ndarray:
    """Singleton mask of the LAC sets ``{k : 1 - p_k <= qhat}`` from the two largest probabilities."""
    return (1.0 - top1 <= qhat) & (1.0 - top2 > qhat)


def lac_singleton_mask(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Accept exactly the instances whose LAC prediction set ``{k : 1 - p_k <= qhat}`` is a singleton.

    Args:
        probs: Probabilities ``(n, classes)`` with at least two classes.
        qhat: Conformal quantile of the LAC scores ``1 - p_y``.

    Returns:
        Boolean array of shape ``(n,)``; the prediction of an accepted instance is its argmax.
    """
    return lac_singleton_from_top_two(*top_two(probs), qhat)


def aps_singleton_from_top_two(top1: np.ndarray, top2: np.ndarray, qhat: float) -> np.ndarray:
    """Singleton mask of the non-randomized APS sets from the two largest probabilities.

    The APS score of the top class is ``p_(1)`` and of the second class ``p_(1) + p_(2)``, so the set is a singleton
    iff ``p_(1) <= qhat < p_(1) + p_(2)``.
    """
    return (top1 <= qhat) & (top1 + top2 > qhat)


def aps_singleton_mask(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Accept exactly the instances whose non-randomized APS prediction set is a singleton.

    Args:
        probs: Probabilities ``(n, classes)`` with at least two classes.
        qhat: Conformal quantile of the non-randomized APS scores.

    Returns:
        Boolean array of shape ``(n,)``.
    """
    return aps_singleton_from_top_two(*top_two(probs), qhat)
