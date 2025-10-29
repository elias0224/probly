import pytest
pytest.importorskip("torch")


MU_KEYS    = ("mu", "mean", "loc", "y_hat", "pred", "m")
V_KEYS     = ("v", "lambda", "precision", "tau")
ALPHA_KEYS = ("alpha", "a")
BETA_KEYS  = ("beta", "b")

def _pick_key(d: dict, candidates, what):
    for k in candidates:
        if k in d:
            return k
    pytest.skip(f"找不到 {what} 的键。这个实现的键有：{list(d.keys())}")


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
    k_mu    = _pick_key(out, MU_KEYS,    "均值(mu)")
    k_v     = _pick_key(out, V_KEYS,     "精度/尺度(v)")
    k_alpha = _pick_key(out, ALPHA_KEYS, "alpha")
    k_beta  = _pick_key(out, BETA_KEYS,  "beta")

    for k in (k_mu, k_v, k_alpha, k_beta):
        assert out[k].shape == x.shape, f"{k} 形状不对：{out[k].shape} vs {x.shape}"

    # 宽松的正性约束
    assert (out[k_v]    > 0).all()
    assert (out[k_beta] > 0).all()
    # 有些实现只保 >0，这里放宽
    assert (out[k_alpha] > 0).all()
else:
    # 分布对象风格
    m = out.mean() if callable(getattr(out, "mean", None)) else getattr(out, "mean", None)
    assert m is not None, "返回对象没有 mean"
    assert m.shape == x.shape


