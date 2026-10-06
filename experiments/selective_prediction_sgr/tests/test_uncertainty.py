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


def test_summarize_samples_with_sample_axis_one() -> None:
    from probly.representation.distribution import create_categorical_distribution  # noqa: PLC0415
    from probly.representation.distribution.torch_categorical import TorchCategoricalDistributionSample  # noqa: PLC0415
    from sgr_experiment.uncertainty import summarize_samples  # noqa: PLC0415

    probs = torch.softmax(torch.randn(5, 8, 10, generator=torch.Generator().manual_seed(1)) * 2, dim=-1)  # (S, n, C)
    as_batch_first = torch.movedim(probs, 0, 1)  # (n, S, C) like the Laplace, SWAG and VBLL representers
    rep = TorchCategoricalDistributionSample(tensor=create_categorical_distribution(as_batch_first), sample_dim=1)
    out = summarize_samples(rep)
    p = probs.double().numpy()
    total = _entropy(p.mean(0))
    aleatoric = _entropy(p).mean(0)
    np.testing.assert_allclose(out["mean_probs"], p.mean(0), atol=1e-6)
    np.testing.assert_allclose(out["maxprob"], 1 - p.mean(0).max(-1), atol=1e-6)
    np.testing.assert_allclose(out["total"], total, atol=1e-5)
    np.testing.assert_allclose(out["aleatoric"], aleatoric, atol=1e-5)
    np.testing.assert_allclose(out["epistemic"], total - aleatoric, atol=1e-5)


def test_one_minus_max_breaks_float32_saturation_ties() -> None:
    from sgr_experiment.uncertainty import one_minus_max, torch_one_minus_max  # noqa: PLC0415

    logits = torch.zeros(3, 10)
    logits[:, 0] = torch.tensor([20.0, 25.0, 30.0])  # max softmax rounds to 1.0 in float32
    p = torch.softmax(logits, dim=-1)
    assert (1 - p.max(-1).values == 0).all()
    crit = one_minus_max(p.numpy())
    assert (np.diff(crit) < 0).all()  # strictly ordered by confidence, no ties
    np.testing.assert_allclose(crit, torch_one_minus_max(p).numpy())
    q = torch.softmax(torch.randn(50, 10, generator=torch.Generator().manual_seed(2)), dim=-1).double().numpy()
    np.testing.assert_allclose(one_minus_max(q), 1 - q.max(-1), atol=1e-6)
