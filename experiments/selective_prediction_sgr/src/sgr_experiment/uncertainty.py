"""Entropy decomposition of sampled predictions through probly (total = H(mean), aleatoric = mean H, epistemic = MI)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from probly.quantification import quantify
from probly.representation.distribution.torch_categorical import TorchProbabilityCategoricalDistribution
from probly.representation.sample.torch import TorchSample

if TYPE_CHECKING:
    from probly.representation.representation import Representation


def member_representation(probs: torch.Tensor) -> TorchSample:
    """Wrap member probabilities ``(members, n, classes)``, e.g. of a deep ensemble, as a probly sample representation."""
    return TorchSample(tensor=TorchProbabilityCategoricalDistribution(probs), sample_dim=0)


def decompose(representation: Representation) -> dict[str, np.ndarray]:
    """Total, aleatoric and epistemic uncertainty (nats, float64 arrays of shape ``(n,)``) via ``quantify``."""
    uq = quantify(representation)
    return {k: getattr(uq, k).detach().cpu().double().numpy() for k in ("total", "aleatoric", "epistemic")}


def predicted_class_variance(probs: torch.Tensor) -> torch.Tensor:
    """Variance over the samples of the probability of the predicted class (the paper's MC criterion).

    Args:
        probs: Sample probabilities ``(samples, n, classes)``.

    Returns:
        Tensor ``(n,)``; the predicted class is the argmax of the sample mean.
    """
    pred = probs.mean(0).argmax(-1)
    pred_probs = probs.gather(-1, pred.expand(probs.shape[0], -1).unsqueeze(-1)).squeeze(-1)
    return pred_probs.var(0, unbiased=False)
