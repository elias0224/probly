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


def test_swag_round_trip_keeps_statistics(tmp_path: Path) -> None:
    from probly.method.swag import collect_swag  # noqa: PLC0415
    from sgr_experiment.loaders import load_swag  # noqa: PLC0415
    from sgr_experiment.model import to_swag  # noqa: PLC0415

    model = to_swag(build_plain_vgg(), max_rank=3)
    for _ in range(2):
        collect_swag(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.01 * torch.randn_like(p))
    torch.save(model.state_dict(), tmp_path / "swag.pt")
    loaded = load_swag(tmp_path / "swag.pt", max_rank=3)
    assert int(loaded.num_collected) == 2
    assert torch.equal(loaded.mean, model.mean)
    assert torch.equal(loaded.deviations, model.deviations)


def test_ddu_and_vbll_round_trip(tmp_path: Path) -> None:
    from sgr_experiment.loaders import load_ddu, load_finetune, load_vbll  # noqa: PLC0415
    from sgr_experiment.model import to_ddu, to_vbll  # noqa: PLC0415

    x = torch.randn(2, 3, 32, 32)
    ddu_model = to_ddu(build_plain_vgg()).eval()
    torch.save(ddu_model.state_dict(), tmp_path / "ddu.pt")
    with torch.no_grad():
        assert torch.allclose(load_ddu(tmp_path / "ddu.pt")(x)[0], ddu_model(x)[0])
    vbll_model = to_vbll(build_plain_vgg()).eval()
    torch.save(vbll_model.state_dict(), tmp_path / "vbll.pt")
    with torch.no_grad():
        assert torch.allclose(load_vbll(tmp_path / "vbll.pt")(x)[0], vbll_model(x)[0])
    torch.save(build_plain_vgg().state_dict(), tmp_path / "finetune.pt")
    assert isinstance(load_finetune(tmp_path / "finetune.pt"), nn.Sequential)


def test_sngp_round_trip(tmp_path: Path) -> None:
    from sgr_experiment.loaders import load_sngp  # noqa: PLC0415
    from sgr_experiment.model import to_sngp  # noqa: PLC0415

    x = torch.randn(4, 3, 32, 32)
    model = to_sngp(build_plain_vgg())
    model.train()
    model(x)  # fills the spectral norm buffers and accumulates the precision matrix
    model.eval()
    with torch.no_grad():
        logits, variance = model(x)  # refreshes the covariance
    torch.save(model.state_dict(), tmp_path / "sngp.pt")
    loaded = load_sngp(tmp_path / "sngp.pt")
    with torch.no_grad():
        logits2, variance2 = loaded(x)
    assert torch.allclose(logits, logits2, atol=1e-5)
    assert torch.allclose(variance, variance2, atol=1e-5)
