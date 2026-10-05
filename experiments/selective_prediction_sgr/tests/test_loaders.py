"""Round-trip tests of the model loaders and the dropout placement."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sgr_experiment.loaders import load_base, load_dropout
from sgr_experiment.model import build_plain_vgg, to_mc_dropout


def test_dropout_inserted_before_each_linear_only() -> None:
    model = to_mc_dropout(build_plain_vgg(), p=0.5)
    layers = list(model.children())
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            assert isinstance(layers[i - 1], nn.Dropout)
            assert layers[i - 1].p == 0.5
    assert sum(isinstance(layer, nn.Linear) for layer in layers) == 2
    plain_dropouts = sum(isinstance(layer, nn.Dropout) for layer in build_plain_vgg().children())
    assert sum(isinstance(layer, nn.Dropout) for layer in layers) == plain_dropouts + 2


def test_round_trip(tmp_path: Path) -> None:
    base = build_plain_vgg().eval()
    torch.save(base.state_dict(), tmp_path / "base.pt")
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(load_base(tmp_path / "base.pt")(x), base(x))
    mc = to_mc_dropout(base, p=0.5).eval()
    torch.save(mc.state_dict(), tmp_path / "dropout.pt")
    with torch.no_grad():
        assert torch.allclose(load_dropout(tmp_path / "dropout.pt")(x), mc(x))
