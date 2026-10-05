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

from sgr_experiment.model import build_plain_vgg, to_mc_dropout


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
