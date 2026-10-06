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


def one_minus_max(probs: np.ndarray) -> np.ndarray:
    """Softmax response criterion ``1 - max p`` without ties from float saturation.

    Computed as the float64 sum of all non-max probabilities. This equals ``1 - max p`` in exact
    arithmetic, but keeps the ranking of confident predictions: float32 values just below 1.0
    are spaced 2**-24 apart, so ``1 - max`` collapses them into a few large tie blocks that the
    rank-based SGR search cannot split.

    Args:
        probs: Probabilities ``(n, classes)``.

    Returns:
        Float64 array ``(n,)``.
    """
    return np.sort(np.asarray(probs, dtype=np.float64), axis=-1)[..., :-1].sum(-1)


def torch_one_minus_max(probs: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`one_minus_max` (float64 sum of the non-max probabilities on the CPU, since MPS has no float64; shape ``(n,)``)."""
    return probs.detach().cpu().double().sort(-1).values[..., :-1].sum(-1)


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


def sample_probabilities(representation: Representation) -> torch.Tensor:
    """Sample probabilities of a representation as ``(samples, n, classes)``, whatever its sample axis is."""
    probs = representation.tensor.probabilities  # ty: ignore[unresolved-attribute]
    return torch.movedim(probs, representation.sample_dim, 0)  # ty: ignore[unresolved-attribute]


def summarize_samples(representation: Representation) -> dict[str, np.ndarray]:
    """Mean probabilities and the criteria ``maxprob``, ``total``, ``aleatoric``, ``epistemic`` as float32 arrays.

    ``maxprob`` is 1 - max mean probability (tie-free, see :func:`torch_one_minus_max`); the three entropies come from probly's ``quantify``.
    """
    mean = sample_probabilities(representation).detach().float().mean(0)
    out = {
        "mean_probs": mean.cpu().numpy(),
        "maxprob": torch_one_minus_max(mean).cpu().numpy(),
    }
    out.update(decompose(representation))
    return {k: v.astype(np.float32) for k, v in out.items()}
