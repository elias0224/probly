"""CIFAR-10 loading with the augmentation of Geifman and El-Yaniv (2017)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

if TYPE_CHECKING:
    from pathlib import Path

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def train_transform() -> transforms.Compose:
    """Random flip, shifts up to 10 percent, rotations up to 15 degrees, normalization."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def test_transform() -> transforms.Compose:
    """Normalization only."""
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_loader(
    data_dir: Path,
    *,
    train: bool,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    subset: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """Build a CIFAR-10 DataLoader, downloading the data into ``data_dir`` if needed.

    Args:
        data_dir: Directory that contains (or will contain) ``cifar-10-batches-py``.
        train: Whether to load the training set with augmentation (shuffled) or the test set.
        batch_size: Batch size.
        workers: Number of worker processes.
        pin_memory: Whether to pin host memory.
        subset: If given, use only this many randomly chosen (fixed by ``seed``) instances.
        seed: Seed for the subset choice.

    Returns:
        The DataLoader.
    """
    ds = datasets.CIFAR10(
        str(data_dir),
        train=train,
        download=True,
        transform=train_transform() if train else test_transform(),
    )
    if subset is not None and subset < len(ds):
        idx = np.random.default_rng(seed).permutation(len(ds))[:subset]
        ds = Subset(ds, idx.tolist())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def load_test_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the full normalized CIFAR-10 test set as tensors ``(10000, 3, 32, 32)`` and labels ``(10000,)``."""
    ds = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_transform())
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y
