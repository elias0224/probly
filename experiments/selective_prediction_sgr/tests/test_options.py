"""Tests of the Learn-then-Test thresholds and the conformal singleton rules on synthetic data."""

from __future__ import annotations

import math

import numpy as np

from sgr_experiment.options import (
    aps_singleton_mask,
    binomial_p_values,
    candidate_thresholds,
    conformal_qhat,
    lac_singleton_mask,
    ltt_bonferroni_threshold,
    ltt_fixed_sequence_threshold,
)


def test_candidate_thresholds_are_quantiles() -> None:
    crit = np.arange(1000, dtype=float)
    t = candidate_thresholds(crit, 100)
    assert t.shape == (100,)
    np.testing.assert_allclose(t[[0, 9, 99]], [9, 99, 999])
    assert np.all(np.diff(t) >= 0)


def test_binomial_p_values() -> None:
    p = binomial_p_values(np.array([0, 1, 0]), np.array([10, 10, 0]), 0.1)
    np.testing.assert_allclose(p, [0.9**10, 0.9**10 + 10 * 0.1 * 0.9**9, 1.0])


def test_tiny_and_all_error_data_are_uncertified() -> None:
    crit = np.linspace(0, 1, 20)
    for loss in (np.zeros(20), np.ones(20)):
        assert ltt_bonferroni_threshold(crit, loss, 0.01, 0.001) == -math.inf
        assert ltt_fixed_sequence_threshold(crit, loss, 0.01, 0.001) == -math.inf
    crit = np.linspace(0, 1, 5000)
    ones = np.ones(5000)
    assert ltt_bonferroni_threshold(crit, ones, 0.05, 0.1) == -math.inf
    assert ltt_fixed_sequence_threshold(crit, ones, 0.05, 0.1) == -math.inf


def test_perfect_data_certifies_nearly_full_coverage() -> None:
    n = 20000
    crit = np.random.default_rng(0).random(n)
    loss = np.zeros(n)
    for thr in (
        ltt_bonferroni_threshold(crit, loss, 0.01, 0.001),
        ltt_fixed_sequence_threshold(crit, loss, 0.01, 0.001),
    ):
        assert (crit <= thr).mean() >= 0.99


def _risk_levels(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted criterion, true error rate per rank (increasing in the rank) and a sampled loss."""
    crit = np.sort(rng.random(n))
    rate = 0.002 + 0.12 * crit**2
    return crit, rate, (rng.random(n) < rate).astype(float)


def test_risk_control_simulation() -> None:
    rng = np.random.default_rng(1)
    n, risk, delta = 2000, 0.03, 0.01
    viol = {"bonf": 0, "fs": 0}
    certified = {"bonf": 0, "fs": 0}
    reps = 200
    for _ in range(reps):
        crit, rate, loss = _risk_levels(rng, n)
        cum_rate = np.cumsum(rate) / np.arange(1, n + 1)  # true selective risk of the first i accepted
        for name, thr in (
            ("bonf", ltt_bonferroni_threshold(crit, loss, risk, delta)),
            ("fs", ltt_fixed_sequence_threshold(crit, loss, risk, delta)),
        ):
            m = int((crit <= thr).sum())
            if m:
                certified[name] += 1
                viol[name] += cum_rate[m - 1] > risk
    for name in viol:
        assert certified[name] > reps // 2
        assert viol[name] / reps <= 0.05


def _naive_fixed_sequence(crit: np.ndarray, loss: np.ndarray, risk: float, delta: float, g: int, k: int) -> float:
    """Loop reference: every sequence walks up from its start and stops at the first uncertified candidate."""
    thresholds = candidate_thresholds(crit, g)
    best = -math.inf
    for j in range(k):
        for i in range((j + 1) * g // k - 1, g):
            accepted = crit <= thresholds[i]
            p = binomial_p_values(loss[accepted].sum(), accepted.sum(), risk)
            if p > delta / k:
                break
            best = max(best, float(thresholds[i]))
    return best


def test_fixed_sequence_stops_at_first_failure() -> None:
    n = 10000
    crit = np.arange(n, dtype=float)
    loss = np.zeros(n)
    loss[3000:6000] = 1.0  # candidates from coverage 0.32 on are far above r*, and stay so
    thr_fs = ltt_fixed_sequence_threshold(crit, loss, 0.05, 0.1, grid_size=100, starts=10)
    thr_bonf = ltt_bonferroni_threshold(crit, loss, 0.05, 0.1, grid_size=100)
    # coverage 0.31 (100 errors among 3100) is certified, 0.32 (200 among 3200) is not
    assert thr_fs == 3099.0
    assert thr_bonf == 3099.0


def test_fixed_sequence_matches_loop_reference() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 3000
        crit = np.sort(rng.random(n))
        rate = 0.01 + 0.2 * crit**2
        rate[rng.integers(0, n, 30)] = 0.9  # error spikes make single candidates fail inside a run
        loss = (rng.random(n) < rate).astype(float)
        got = ltt_fixed_sequence_threshold(crit, loss, 0.05, 0.05, grid_size=50, starts=5)
        assert got == _naive_fixed_sequence(crit, loss, 0.05, 0.05, 50, 5)


def test_conformal_qhat() -> None:
    scores = np.arange(1.0, 10.0)  # n = 9
    assert conformal_qhat(scores, 0.5) == 5.0  # ceil(10 * 0.5) = 5
    assert conformal_qhat(scores, 0.1) == 9.0  # ceil(9) = 9
    assert conformal_qhat(scores, 0.05) == math.inf  # ceil(9.5) = 10 > 9
    assert conformal_qhat(scores[::-1], 0.5) == 5.0


def test_lac_singleton_mask() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.45, 0.05], [0.4, 0.3, 0.3], [0.7, 0.2, 0.1]])
    # qhat 0.31: set = {k : p_k >= 0.69}
    np.testing.assert_array_equal(lac_singleton_mask(probs, 0.31), [True, False, False, True])
    # qhat 0.65: set = {k : p_k >= 0.35}
    np.testing.assert_array_equal(lac_singleton_mask(probs, 0.65), [True, False, True, True])
    assert not lac_singleton_mask(probs, math.inf).any()


def test_aps_singleton_mask() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.5, 0.45, 0.05], [0.4, 0.3, 0.3]])
    # top-1 score is p1, second class score is p1 + p2
    np.testing.assert_array_equal(aps_singleton_mask(probs, 0.95), [True, False, False])
    np.testing.assert_array_equal(aps_singleton_mask(probs, 0.5), [False, True, True])
    assert not aps_singleton_mask(probs, math.inf).any()
