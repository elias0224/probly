"""Hand-checked tests of the exact risk-coverage helpers."""

from __future__ import annotations

import numpy as np

from probly.metrics.selective_prediction import aurc, coverage_at_risk, risk_coverage_curve
from sgr_experiment.metrics import (
    apply_threshold,
    auroc,
    e_aurc,
    risk_bound,
    sgr_threshold,
    threshold_for_risk,
)

# Sorted by criterion: 0.1(0), 0.2(1), 0.3(0), 0.4(0) -> risks 0, 1/2, 1/3, 1/4.
C = np.array([0.4, 0.1, 0.3, 0.2])
L = np.array([0.0, 0.0, 0.0, 1.0])


def test_probly_curve_has_coverage_zero_endpoint() -> None:
    cov, rsk, thr = risk_coverage_curve(C, L)
    np.testing.assert_allclose(cov, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(rsk, [0.0, 0.0, 1 / 2, 1 / 3, 1 / 4])
    assert thr[0] == -np.inf


def test_coverage_at_risk_non_monotone() -> None:
    # Risks are 0, 1/2, 1/3, 1/4: the risk drops again after coverage 0.5.
    assert coverage_at_risk(C, L, 0.25) == 1.0
    assert coverage_at_risk(C, L, 0.3) == 1.0
    assert coverage_at_risk(C, L, 0.4) == 1.0
    assert coverage_at_risk(C, L, 0.0) == 0.25
    assert coverage_at_risk(C, L, 1.0) == 1.0


def test_threshold_for_risk_and_apply() -> None:
    thr = threshold_for_risk(C, L, 0.0)
    assert thr == 0.1
    risk, cov = apply_threshold(C, L, thr)
    assert (risk, cov) == (0.0, 0.25)
    assert threshold_for_risk(C, L, 0.3) == 0.4
    assert threshold_for_risk(np.array([0.1]), np.array([1.0]), 0.0) == -np.inf
    risk, cov = apply_threshold(np.array([0.1]), np.array([1.0]), -np.inf)
    assert np.isnan(risk)
    assert cov == 0.0


def test_threshold_with_ties() -> None:
    c = np.array([0.1, 0.2, 0.2, 0.3])
    loss = np.array([0.0, 1.0, 0.0, 0.0])
    # Accepting the tied pair costs risk 1/3, so r = 0.2 only allows the first instance.
    assert threshold_for_risk(c, loss, 0.2) == 0.1
    assert coverage_at_risk(c, loss, 0.2) == 0.25


def test_e_aurc_hand_computed() -> None:
    # Trapezoid AURC over the curve with the coverage-0 endpoint; the optimal ranking has risks 0, 0, 0, 0, 1/4.
    trapz = lambda r: float(np.trapezoid(r, [0.0, 0.25, 0.5, 0.75, 1.0]))  # noqa: E731
    assert np.isclose(aurc(C, L), trapz([0, 0, 1 / 2, 1 / 3, 1 / 4]))
    assert np.isclose(e_aurc(C, L), trapz([0, 0, 1 / 2, 1 / 3, 1 / 4]) - trapz([0, 0, 0, 0, 1 / 4]))


def test_threshold_agrees_with_coverage_at_risk() -> None:
    rng = np.random.default_rng(1)
    crit = np.round(rng.random(500), 2)
    loss = (rng.random(500) < 0.2 * crit).astype(np.float64)
    for r in (0.0, 0.02, 0.05, 0.1, 0.3):
        thr = threshold_for_risk(crit, loss, r)
        assert apply_threshold(crit, loss, thr)[1] == coverage_at_risk(crit, loss, r) or thr == -np.inf


def test_e_aurc_zero_for_optimal_ranking() -> None:
    loss = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    assert np.isclose(e_aurc(loss + 0.01 * np.arange(5), loss), 0.0)


def test_risk_bound_hand_checked() -> None:
    # No errors: 1 - delta^(1/m).
    assert np.isclose(risk_bound(0, 10, 0.05), 1 - 0.05**0.1)
    assert risk_bound(3, 3, 0.05) == 1.0
    assert risk_bound(1, 10, 0.05) > 0.1  # above the empirical risk


def test_risk_bound_decreases_with_samples() -> None:
    bounds = [risk_bound(round(0.02 * m), m, 0.001) for m in (100, 500, 2500, 12500)]
    assert all(a > b for a, b in zip(bounds, bounds[1:], strict=False))
    assert all(b >= 0.02 for b in bounds)


def _random_problem(m: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    crit = rng.random(m)
    return crit, (rng.random(m) < 0.1 * crit**3).astype(np.float64)


def test_sgr_threshold_set_satisfies_bound() -> None:
    crit, loss = _random_problem()
    m = len(crit)
    delta_step = 0.001 / np.ceil(np.log2(m))
    for r_star in (0.01, 0.02, 0.05):
        thr, bound = sgr_threshold(crit, loss, r_star)
        assert np.isfinite(thr)
        accepted = crit <= thr
        k = int(loss[accepted].sum())
        assert bound <= r_star
        assert np.isclose(bound, risk_bound(k, int(accepted.sum()), delta_step))
        assert bound >= loss[accepted].mean()
        # The guaranteed threshold is more conservative than the empirical one.
        assert thr <= threshold_for_risk(crit, loss, r_star)


def test_sgr_threshold_hand_checked() -> None:
    crit = np.arange(16, dtype=np.float64)
    loss = np.zeros(16)
    # 4 steps, delta' = 0.001; without errors B* = 1 - 0.001^(1/m_acc): 0.578 for 8 instances, 0.499 for 10.
    # r* = 0.6: the search tests the sizes 8, 12, 14, 15, all feasible, and ends with 15 accepted instances.
    thr, bound = sgr_threshold(crit, loss, 0.6, delta=0.004)
    assert thr == 14.0
    assert np.isclose(bound, 1 - 0.001 ** (1 / 15))
    # r* = 0.5: 8 instances already fail, so the search keeps shrinking until nothing is accepted.
    assert sgr_threshold(crit, loss, 0.5, delta=0.004) == (-np.inf, 1.0)


def test_sgr_threshold_nothing_accepted() -> None:
    thr, bound = sgr_threshold(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0, 1.0]), 0.05)
    assert thr == -np.inf
    assert bound == 1.0


def test_sgr_threshold_accepts_whole_tie_groups() -> None:
    crit = np.repeat(np.arange(10, dtype=np.float64), 100)
    loss = np.zeros(1000)
    thr, _ = sgr_threshold(crit, loss, 0.05)
    assert thr in set(range(10))
    assert (crit <= thr).sum() % 100 == 0


def test_auroc() -> None:
    assert auroc(np.array([0.1, 0.2]), np.array([0.3, 0.4])) == 1.0
    assert auroc(np.array([0.3, 0.4]), np.array([0.1, 0.2])) == 0.0
    assert auroc(np.array([0.1, 0.3]), np.array([0.2, 0.4])) == 0.75
    assert auroc(np.array([0.5, 0.5]), np.array([0.5])) == 0.5


def _selector_threshold(crit: np.ndarray, loss: np.ndarray, r_star: float) -> float:
    import warnings  # noqa: PLC0415

    from probly.selective_prediction import SGRSelector  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(SGRSelector(r_star, 0.001).calibrate(crit, loss).threshold)


def test_probly_sgr_selector_matches_reference_on_continuous_scores() -> None:
    for seed in range(5):
        rng = np.random.default_rng(seed)
        crit = rng.random(5000)
        loss = (rng.random(5000) < 0.1 * crit**3).astype(np.float64)
        # Risks below the risk of the whole set: probly never tests the largest score, ours can, so they would differ
        # by one instance if almost everything was certifiable.
        for r_star in (0.005, 0.01, 0.02):
            assert _selector_threshold(crit, loss, r_star) == sgr_threshold(crit, loss, r_star)[0]


def test_sgr_thresholds_with_ties_are_both_certified() -> None:
    # With ties the two searches can end at different thresholds, because ours bisects over tie groups and probly's over
    # instances, so the probe order differs. Both returned sets must still be certified.
    rng = np.random.default_rng(0)
    crit = np.round(rng.random(5000), 2)
    loss = (rng.random(5000) < 0.1 * crit**3).astype(np.float64)
    delta_step = 0.001 / np.ceil(np.log2(len(crit)))
    for r_star in (0.01, 0.02, 0.05):
        for thr in (sgr_threshold(crit, loss, r_star)[0], _selector_threshold(crit, loss, r_star)):
            accepted = crit <= thr
            assert accepted.any()
            assert risk_bound(int(loss[accepted].sum()), int(accepted.sum()), delta_step) <= r_star
