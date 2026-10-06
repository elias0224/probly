"""Calibration measures and uncertainty scores computed from class probabilities."""

from __future__ import annotations

import numpy as np

from sgr_experiment.uncertainty import one_minus_max


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: the mean over samples of ``sum_k (p_k - onehot_k)^2``.

    Args:
        probs: Probabilities ``(n, classes)``.
        labels: Integer labels ``(n,)``.

    Returns:
        The score in ``[0, 2]``, lower is better.
    """
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((p - onehot) ** 2).sum(1).mean())


def entropy_nats(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats per row (float64, with ``0 log 0 = 0``).

    Args:
        probs: Probabilities ``(n, classes)``.

    Returns:
        Array ``(n,)``.
    """
    p = np.asarray(probs, dtype=np.float64)
    safe = np.where(p > 0, p, 1.0)
    return -(p * np.log(safe)).sum(1)


def brier_uncertainty(probs: np.ndarray) -> np.ndarray:
    """Brier (Gini) uncertainty ``1 - sum_k p_k^2`` per row (float64).

    Args:
        probs: Probabilities ``(n, classes)``.

    Returns:
        Array ``(n,)``.
    """
    p = np.asarray(probs, dtype=np.float64)
    return 1.0 - (p**2).sum(1)


def ranking_scores(probs: np.ndarray) -> dict[str, np.ndarray]:
    """The three selective-ranking scores ``msr``, ``tu_log`` and ``tu_brier`` (higher means more uncertain)."""
    return {"msr": one_minus_max(probs), "tu_log": entropy_nats(probs), "tu_brier": brier_uncertainty(probs)}
