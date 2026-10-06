"""Reload the trained models of the two stages, e.g. for future experiments.

Stage 1 ("base") is the plain VGG; stage 2 ("dropout") is the same network with probly's MC dropout applied and
fine-tuned.

Example:
    >>> base = load_base("runs/seed0/base.pt")
    >>> mc_model = load_dropout("runs/seed0/dropout.pt", p=0.5)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sgr_experiment.model import build_plain, to_ddu, to_mc_dropout, to_sngp, to_swag, to_vbll


def _load_state(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_base(path: str | Path, arch: str = "vgg16") -> nn.Module:
    """Load the plain network of ``arch`` (no dropout before the Linear layers) from ``base.pt``, in eval mode on the CPU."""
    model = build_plain(arch)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_dropout(path: str | Path, p: float = 0.5, arch: str = "vgg16") -> nn.Module:
    """Load the MC-dropout model from ``dropout.pt``, in eval mode on the CPU.

    The architecture is rebuilt, the probly dropout transformation is applied with probability ``p``, and then the
    state dict is loaded. ``p`` must match the value used for training.
    """
    model = to_mc_dropout(build_plain(arch), p=p)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_finetune(path: str | Path, arch: str = "vgg16") -> nn.Module:
    """Load the fine-tuned plain VGG from ``finetune.pt`` (same architecture as the base), in eval mode on the CPU."""
    return load_base(path, arch)


def load_swag(path: str | Path, max_rank: int = 20, scale: float = 0.5, arch: str = "vgg16") -> nn.Module:
    """Load the SWAG predictor from ``swag.pt`` (weights and collected statistics), in eval mode on the CPU."""
    model = to_swag(build_plain(arch), max_rank=max_rank, scale=scale)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_ddu(path: str | Path, sn_coeff: float = 3.0, arch: str = "vgg16") -> nn.Module:
    """Load the DDU predictor from ``ddu.pt`` in eval mode on the CPU; the density head is not fitted yet."""
    model = to_ddu(build_plain(arch), sn_coeff=sn_coeff)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_sngp(
    path: str | Path,
    norm_multiplier: float = 6.0,
    random_feature_init_std: float = 0.05,
    momentum: float = -1.0,
    num_random_features: int = 1024,
    arch: str = "vgg16",
) -> nn.Module:
    """Load the SNGP model from ``sngp.pt`` in eval mode on the CPU (the model returns ``(logits, variance)``)."""
    model = to_sngp(
        build_plain(arch),
        norm_multiplier=norm_multiplier,
        random_feature_init_std=random_feature_init_std,
        momentum=momentum,
        num_random_features=num_random_features,
    )
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_vbll(path: str | Path, parameterization: str = "dense", arch: str = "vgg16") -> nn.Module:
    """Load the VBLL model from ``vbll.pt``, in eval mode on the CPU."""
    model = to_vbll(build_plain(arch), parameterization=parameterization)
    model.load_state_dict(_load_state(path))
    return model.eval()
