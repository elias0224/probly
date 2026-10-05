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

from sgr_experiment.model import build_plain_vgg, to_ddu, to_mc_dropout, to_swag, to_vbll


def _load_state(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_base(path: str | Path) -> nn.Sequential:
    """Load the plain VGG (no dropout before the Linear layers) from ``base.pt``, in eval mode on the CPU."""
    model = build_plain_vgg()
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_dropout(path: str | Path, p: float = 0.5) -> nn.Module:
    """Load the MC-dropout model from ``dropout.pt``, in eval mode on the CPU.

    The architecture is rebuilt, the probly dropout transformation is applied with probability ``p``, and then the
    state dict is loaded. ``p`` must match the value used for training.
    """
    model = to_mc_dropout(build_plain_vgg(), p=p)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_finetune(path: str | Path) -> nn.Sequential:
    """Load the fine-tuned plain VGG from ``finetune.pt`` (same architecture as the base), in eval mode on the CPU."""
    return load_base(path)


def load_swag(path: str | Path, max_rank: int = 20, scale: float = 0.5) -> nn.Module:
    """Load the SWAG predictor from ``swag.pt`` (weights and collected statistics), in eval mode on the CPU."""
    model = to_swag(build_plain_vgg(), max_rank=max_rank, scale=scale)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_ddu(path: str | Path, sn_coeff: float = 3.0) -> nn.Module:
    """Load the DDU predictor from ``ddu.pt`` in eval mode on the CPU; the density head is not fitted yet."""
    model = to_ddu(build_plain_vgg(), sn_coeff=sn_coeff)
    model.load_state_dict(_load_state(path))
    return model.eval()


def load_vbll(path: str | Path, parameterization: str = "dense") -> nn.Module:
    """Load the VBLL model from ``vbll.pt``, in eval mode on the CPU."""
    model = to_vbll(build_plain_vgg(), parameterization=parameterization)
    model.load_state_dict(_load_state(path))
    return model.eval()
