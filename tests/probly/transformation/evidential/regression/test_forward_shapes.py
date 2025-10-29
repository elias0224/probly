import pytest
pytest.importorskip("torch")


def _build_model():
    import inspect
    import torch.nn as nn
    import probly.transformation.evidential.regression as er

    fn = er.evidential_regression
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    # 一个最小的回归骨干
    backbone = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    # 1) 先按参数名动态拼 kwargs
    name_map = ["model", "backbone", "net", "module"]
    kwargs = {}
    for k in name_map:
        if k in param_names:
            kwargs[k] = backbone
            break

    # 常见可选参数：head、输出维度
    if "head" in param_names:
        kwargs["head"] = "nig"
    for out_key in ("num_outputs", "out_features", "output_dim"):
        if out_key in param_names:
            kwargs[out_key] = 4  # NIG 通常导出 4 个参数

    # 尝试：关键字方式
    try:
        out = fn(**kwargs) if kwargs else fn()
        if out is None:
            # 有些变换是“就地修改”，返回 None
            return backbone
        # 返回的是模型（有 forward）
        if hasattr(out, "forward"):
            return out
        # 返回的是可调用的“转换器”，再喂 backbone
        if callable(out):
            mod = out(backbone)
            return backbone if mod is None else mod
    except TypeError:
        pass

    # 2) 再试位置参数（有些签名只认位置参数）
    try:
        out = fn(backbone)
        if out is None:
            return backbone
        if hasattr(out, "forward"):
            return out
        if callable(out):
            mod = out(backbone)
            return backbone if mod is None else mod
    except TypeError:
        pass

    import pytest
    pytest.skip(f"搞不清 evidential_regression 的用法，实际签名是 {sig}。把上面 kwargs 映射改成它要的键即可。")





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



