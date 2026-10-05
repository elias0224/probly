"""Cross-check probly's entropy decomposition against plain numpy formulas."""

from __future__ import annotations

import numpy as np
import torch

from sgr_experiment.uncertainty import decompose, member_representation, predicted_class_variance


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(p)).sum(-1)


def test_decomposition_matches_numpy() -> None:
    probs = torch.softmax(torch.randn(6, 9, 10, generator=torch.Generator().manual_seed(0)) * 2, dim=-1).double()
    out = decompose(member_representation(probs))
    p = probs.numpy()
    total = _entropy(p.mean(0))
    aleatoric = _entropy(p).mean(0)
    np.testing.assert_allclose(out["total"], total, atol=1e-6)
    np.testing.assert_allclose(out["aleatoric"], aleatoric, atol=1e-6)
    np.testing.assert_allclose(out["epistemic"], total - aleatoric, atol=1e-6)


def test_identical_members_have_no_epistemic_uncertainty() -> None:
    p = torch.softmax(torch.randn(1, 5, 10), dim=-1).repeat(4, 1, 1)
    out = decompose(member_representation(p))
    np.testing.assert_allclose(out["epistemic"], 0.0, atol=1e-6)


def test_predicted_class_variance() -> None:
    probs = torch.softmax(torch.randn(7, 5, 10), dim=-1)
    got = predicted_class_variance(probs).numpy()
    pred = probs.mean(0).argmax(-1).numpy()
    expected = np.array([probs[:, i, pred[i]].numpy().var() for i in range(5)])
    np.testing.assert_allclose(got, expected, atol=1e-6)
