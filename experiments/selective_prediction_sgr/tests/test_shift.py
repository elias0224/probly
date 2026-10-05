"""Tests of the corruptions."""

from __future__ import annotations

import pytest
import torch

from sgr_experiment.shift import CORRUPTIONS, _contrast, corrupt, corrupted_name


def _images(n: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, 3, 32, 32), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_shape_and_dtype(name: str) -> None:
    x = _images()
    out = corrupt(x, name, 3, torch.Generator().manual_seed(1))
    assert out.shape == x.shape
    assert out.dtype == torch.uint8


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_deterministic(name: str) -> None:
    x = _images()
    a = corrupt(x, name, 5, torch.Generator().manual_seed(7))
    b = corrupt(x, name, 5, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_noise_depends_on_seed() -> None:
    x = _images()
    a = corrupt(x, "gaussian_noise", 5, torch.Generator().manual_seed(1))
    b = corrupt(x, "gaussian_noise", 5, torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_severity_ordering(name: str) -> None:
    x = _images(32)
    diffs = [
        (corrupt(x, name, s, torch.Generator().manual_seed(3)).float() - x.float()).abs().mean().item() for s in (1, 3, 5)
    ]
    assert diffs[0] < diffs[1] < diffs[2]


def test_contrast_one_is_identity() -> None:
    f = _images().float() / 255
    assert torch.allclose(_contrast(f, 1.0), f, atol=1e-6)


def test_noise_needs_generator() -> None:
    with pytest.raises(ValueError, match="generator"):
        corrupt(_images(), "gaussian_noise", 1)


def test_name() -> None:
    assert corrupted_name("gaussian_noise", 3) == "c10_gaussian_noise_s3"
