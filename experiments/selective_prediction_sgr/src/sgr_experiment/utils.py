"""Shared helpers: device selection, seeding, and default paths."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    """Select the best available device in the order cuda, mps, cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dir(runs: Path, seed: int) -> Path:
    """Directory that holds all files of one seed."""
    return Path(runs) / f"seed{seed}"
