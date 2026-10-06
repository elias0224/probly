"""Hand-checked tests of the calibration helpers."""

from __future__ import annotations

import numpy as np

from sgr_experiment.calibration import brier_score, brier_uncertainty, entropy_nats, ranking_scores


def test_brier_score_hand_checked() -> None:
    p = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 0])
    # row 1: 0.3^2 + 0.3^2 = 0.18; row 2: 0.8^2 + 0.8^2 = 1.28
    assert np.isclose(brier_score(p, labels), (0.18 + 1.28) / 2)


def test_brier_score_perfect_and_worst() -> None:
    p = np.eye(3)
    assert brier_score(p, np.arange(3)) == 0.0
    assert np.isclose(brier_score(p, (np.arange(3) + 1) % 3), 2.0)


def test_entropy_nats_zero_log_zero() -> None:
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    h = entropy_nats(p)
    assert h[0] == 0.0
    assert np.isclose(h[1], np.log(2))
    assert np.isclose(h[2], -(2 * 0.25 * np.log(0.25) + 0.5 * np.log(0.5)))
    assert np.all(np.isfinite(h))


def test_brier_uncertainty() -> None:
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.8, 0.2]])
    np.testing.assert_allclose(brier_uncertainty(p), [0.0, 0.5, 1 - 0.68])


def test_ranking_scores_keys_and_order() -> None:
    p = np.array([[0.9, 0.1], [0.6, 0.4]])
    s = ranking_scores(p)
    assert set(s) == {"msr", "tu_log", "tu_brier"}
    for v in s.values():
        assert v[0] < v[1]
    np.testing.assert_allclose(s["msr"], [0.1, 0.4])
