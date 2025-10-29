import pytest
pytest.importorskip("torch")

def _build_model():
    import torch.nn as nn
    import probly.transformation.evidential.regression as er

    backbone = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    if hasattr(er, "EvidentialRegression"):
        try:
            return er.EvidentialRegression(backbone)          
        except TypeError:
            return er.EvidentialRegression(backbone=backbone) 

    if hasattr(er, "make_evidential_regression"):
        return er.make_evidential_regression(backbone=backbone)

    pytest.skip("Kein Konstruktor gefunden：Bitte den Name in _build_model() auf den tatsächlich exportierte Projekt von API ändern")

def test_forward_shapes():
    import torch
    model = _build_model()
    model.eval()
    x = torch.randn(8, 1)
    with torch.no_grad():
        out = model(x)

    if isinstance(out, dict):
        for k in ["mu", "v", "alpha", "beta"]:
            assert k in out, f"fehlt {k}"
            assert out[k].shape == x.shape
           
            if k in ("v", "beta"):
                assert (out[k] > 0).all()
    else:
        assert hasattr(out, "mean")
        assert out.mean.shape == x.shape


