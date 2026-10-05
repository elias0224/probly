"""OOD test sets (SVHN, CIFAR-100) and GPU-friendly CIFAR-10-C style corruptions of the CIFAR-10 test set.

The corruption parameters are those of Hendrycks and Dietterich (2019), ``make_cifar_c.py``, for severities 1 to 5.
Corrupted images are clipped and re-quantized to uint8 so that they look like real images, and are then normalized
with the CIFAR-10 statistics (the OOD sets as well, since that is what the model expects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets

if TYPE_CHECKING:
    from pathlib import Path

NOISE_SIGMA = [0.04, 0.06, 0.08, 0.09, 0.10]
BLUR_SIGMA = [0.4, 0.6, 0.7, 0.8, 1.0]
CONTRAST_C = [0.75, 0.5, 0.4, 0.3, 0.15]
PIXELATE_C = [0.95, 0.9, 0.85, 0.75, 0.65]
CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "contrast", "pixelate"]
SEVERITIES = [1, 3, 5]
OOD_DATASETS = ["svhn", "cifar100"]


def load_svhn_test(data_dir: Path) -> torch.Tensor:
    """Load the SVHN test split (26032 images) as uint8 ``(n, 3, 32, 32)`` on the CPU, downloading it if needed."""
    ds = datasets.SVHN(str(data_dir), split="test", download=True)
    return torch.from_numpy(np.ascontiguousarray(ds.data)).contiguous()


def load_cifar100_test(data_dir: Path) -> torch.Tensor:
    """Load the CIFAR-100 test split (10000 images) as uint8 ``(n, 3, 32, 32)`` on the CPU, downloading it if needed."""
    ds = datasets.CIFAR100(str(data_dir), train=False, download=True)
    return torch.from_numpy(np.ascontiguousarray(ds.data)).permute(0, 3, 1, 2).contiguous()


def corrupted_name(corruption: str, severity: int) -> str:
    """Dataset name of a corrupted CIFAR-10 test set, e.g. ``c10_gaussian_noise_s3``."""
    return f"c10_{corruption}_s{severity}"


def _gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = int(4 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(t**2) / (2 * sigma**2))
    return k / k.sum()


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gaussian_kernel(sigma).to(x.device)
    r = len(k) // 2
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def _contrast(x: torch.Tensor, c: float) -> torch.Tensor:
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * c + mean


def _pixelate(x: torch.Tensor, c: float) -> torch.Tensor:
    size = x.shape[-1]
    small = F.interpolate(x, size=round(size * c), mode="area")
    return F.interpolate(small, size=size, mode="nearest")


def corrupt(x: torch.Tensor, corruption: str, severity: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply a corruption to uint8 images ``(n, 3, 32, 32)``; the result is uint8 again.

    Args:
        x: uint8 images.
        corruption: One of :data:`CORRUPTIONS`.
        severity: 1 to 5.
        generator: Generator on the device of ``x``; only ``gaussian_noise`` is random and needs it.

    Returns:
        The corrupted images, clipped to ``[0, 255]`` and rounded.
    """
    if not 1 <= severity <= 5:
        msg = f"severity must be in 1..5, got {severity}."
        raise ValueError(msg)
    i = severity - 1
    f = x.float() / 255
    if corruption == "gaussian_noise":
        if generator is None:
            msg = "gaussian_noise needs a generator."
            raise ValueError(msg)
        f = f + torch.randn(f.shape, generator=generator, device=f.device) * NOISE_SIGMA[i]
    elif corruption == "gaussian_blur":
        f = _blur(f, BLUR_SIGMA[i])
    elif corruption == "contrast":
        f = _contrast(f, CONTRAST_C[i])
    elif corruption == "pixelate":
        f = _pixelate(f, PIXELATE_C[i])
    else:
        msg = f"unknown corruption {corruption!r}."
        raise ValueError(msg)
    return (f.clamp(0, 1) * 255).round().to(torch.uint8)
