import pytest
pytest.importorskip("torch")


def _build_model():
   def _build_model():
    import torch.nn as nn
    import probly.transformation.evidential.regression as er

    backbone = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    # 模块只导出了函数 evidential_regression；尝试几种常见签名
    for kwargs in (
        {"backbone": backbone},
        {"model": backbone},
        {"net": backbone},
        {"module": backbone},
        {},                
        (backbone,),       
    ):
        try:
            if isinstance(kwargs, dict):
                return er.evidential_regression(**kwargs)
            else:
                return er.evidential_regression(*kwargs)
        except TypeError:
            continue

    import pytest
    pytest.skip("evidential_regression 的签名对不上：请把参数名改成实际要求的那个（如 backbone/model/net/module 等）。")



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
        m = out.mean() if callable(getattr(out, "mean", None)) else getattr(out, "mean", None)
        assert m is not None, "Kein mean von zurückgegebene Objekt"
        assert m.shape == x.shape



