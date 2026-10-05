"""Hand-checked tests of the exact risk-coverage helpers."""

from __future__ import annotations

import numpy as np

from sgr_experiment.metrics import apply_threshold, coverage_at_risk, risk_coverage_curve, threshold_for_risk

# Sorted by criterion: 0.1(0), 0.2(1), 0.3(0), 0.4(0) -> risks 0, 1/2, 1/3, 1/4.
C = np.array([0.4, 0.1, 0.3, 0.2])
L = np.array([0.0, 0.0, 0.0, 1.0])


def test_curve_no_ties() -> None:
    cov, rsk = risk_coverage_curve(C, L)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(rsk, [0.0, 1 / 2, 1 / 3, 1 / 4])


def test_curve_ties_accepted_together() -> None:
    c = np.array([0.1, 0.2, 0.2, 0.3])
    loss = np.array([0.0, 1.0, 0.0, 0.0])
    cov, rsk = risk_coverage_curve(c, loss)
    np.testing.assert_allclose(cov, [0.25, 0.75, 1.0])
    np.testing.assert_allclose(rsk, [0.0, 1 / 3, 1 / 4])


def test_coverage_at_risk_non_monotone() -> None:
    # Risks are 0, 1/2, 1/3, 1/4: the risk drops again after coverage 0.5.
    assert coverage_at_risk(C, L, 0.25) == 1.0
    assert coverage_at_risk(C, L, 0.3) == 1.0
    assert coverage_at_risk(C, L, 0.4) == 1.0
    assert coverage_at_risk(C, L, 0.0) == 0.25
    assert coverage_at_risk(C, L, 1.0) == 1.0


def test_coverage_at_risk_none_feasible() -> None:
    assert coverage_at_risk(np.array([0.1, 0.2]), np.array([1.0, 1.0]), 0.5) == 0.0


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
