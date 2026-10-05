"""CIFAR-10 held on the device, with the augmentation of Geifman and El-Yaniv (2017) applied there in batches.

The whole dataset (150 MB as uint8) lives on the GPU and is augmented with batched affine warps, so training needs no
DataLoader workers. Per-image CPU augmentation with PIL was the bottleneck on Windows.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
MAX_ROTATION_DEG = 15.0
MAX_SHIFT = 0.1  # fraction of the image size


def load_cifar10(data_dir: Path, *, train: bool, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Load a CIFAR-10 split as uint8 images ``(n, 3, 32, 32)`` and labels ``(n,)`` on ``device``.

    The data is downloaded into ``data_dir`` if it is not there yet.
    """
    ds = datasets.CIFAR10(str(data_dir), train=train, download=True)
    x = torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Convert uint8 images to float in ``[0, 1]`` and apply the CIFAR-10 normalization."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255 - mean) / std


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random horizontal flip, shifts up to 10 percent and rotations up to 15 degrees for a float batch.

    Pixels moved in from outside the image repeat the border, like the default ``fill_mode="nearest"`` of the Keras
    ``ImageDataGenerator`` the paper used.
    """
    n = x.shape[0]
    dev = x.device

    def uniform(low: float, high: float) -> torch.Tensor:
        return torch.rand(n, generator=generator, device=dev) * (high - low) + low

    angle = uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG) * (math.pi / 180)
    # affine_grid works in coordinates from -1 to 1, so a shift by a fraction f of the image is 2 f.
    tx = uniform(-MAX_SHIFT, MAX_SHIFT) * 2
    ty = uniform(-MAX_SHIFT, MAX_SHIFT) * 2
    flip = torch.where(torch.rand(n, generator=generator, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack(
        [torch.stack([cos * flip, -sin, tx], dim=1), torch.stack([sin * flip, cos, ty], dim=1)],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def train_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield shuffled, augmented and normalized batches for one epoch.

    Args:
        x: uint8 images ``(n, 3, 32, 32)`` on the training device.
        y: Labels ``(n,)`` on the same device.
        batch_size: Batch size; the last batch may be smaller.
        generator: Generator on the same device that drives shuffling and augmentation.
    """
    perm = torch.randperm(len(y), generator=generator, device=y.device)
    for start in range(0, len(y), batch_size):
        idx = perm[start : start + batch_size]
        xb = x[idx].float() / 255
        xb = augment(xb, generator)
        mean = torch.tensor(MEAN, device=xb.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=xb.device).view(1, 3, 1, 1)
        yield (xb - mean) / std, y[idx]


def load_test_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the full normalized CIFAR-10 test set on the CPU as ``(10000, 3, 32, 32)`` floats and ``(10000,)`` labels."""
    x, y = load_cifar10(data_dir, train=False)
    return normalize(x), y
