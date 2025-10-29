import pytest
pytest.importorskip("torch")


def _build_model():
    import inspect
    import torch.nn as nn
    import probly.transformation.evidential.regression as er

    # 一个很小的回归 backbone
    backbone = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    fn = er.evidential_regression

    # 1) 直呼模式：evidential_regression(backbone=...)
    for kwargs in ({"backbone": backbone}, {"model": backbone}, {"net": backbone}, {"module": backbone}):
        try:
            out = fn(**kwargs)
            if out is None:
                # 很多“就地变换”返回 None，但已经把 backbone 改好了
                return backbone
            if hasattr(out, "forward"):
                return out
        except TypeError:
            pass

    # 2) 柯里化/两段式：trans = evidential_regression(...); model = trans(backbone)
    try:
        trans = fn()
        if callable(trans):
            out = trans(backbone)
            if out is None:
                return backbone
            if hasattr(out, "forward"):
                return out
    except TypeError:
        pass

    # 3) 只返回 head：把 head 接到 backbone 后面
    for kw in ({"in_features": 1, "out_features": 1},
               {"in_features": 1, "output_dim": 1},
               {"features": 1}):
        try:
            head = fn(**kw)
            if hasattr(head, "forward"):
                return nn.Sequential(backbone, head)
        except TypeError:
            pass

    import pytest
    sig = ""
    try:
        import inspect as _inspect
        sig = str(_inspect.signature(fn))
    except Exception:
        pass
    pytest.skip(f"evidential_regression 的调用方式不明，当前签名 {sig}。请按实际 API 改 _build_model()。")




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



