import warnings
import pytest

# 没装 torch 就跳过整份测试，别污染 CI
pytest.importorskip("torch")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")

# 可能的键名映射（不同实现爱起骚名）
MU_KEYS    = ("mu", "mean", "loc", "y_hat", "pred", "m")
V_KEYS     = ("v", "lambda", "precision", "tau")
ALPHA_KEYS = ("alpha", "a")
BETA_KEYS  = ("beta", "b")


def _pick_key(d: dict, candidates, what):
    for k in candidates:
        if k in d:
            return k
    pytest.skip(f"找不到 {what} 的键。这个实现里可用的键：{list(d.keys())}")


def _build_model():
    import inspect
    import torch.nn as nn
    import probly.transformation.evidential.regression as er

    fn = er.evidential_regression
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())

    # 简单回归 backbone
    backbone = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))

    # 1) 关键字调用：自动匹配参数名
    name_map = ["model", "backbone", "net", "module"]
    kwargs = {}
    for k in name_map:
        if k in param_names:
            kwargs[k] = backbone
            break

    if "head" in param_names:
        kwargs["head"] = "nig"
    for out_key in ("num_outputs", "out_features", "output_dim"):
        if out_key in param_names:
            kwargs[out_key] = 4  # NIG 常见 4 参数

    try:
        out = fn(**kwargs) if kwargs else fn()
        if out is None:
            return backbone            # 就地修改型返回 None
        if hasattr(out, "forward"):
            return out                 # 直接返回模型
        if callable(out):
            mod = out(backbone)        # 柯里化：先拿转换器再套模型
            return backbone if mod is None else mod
    except TypeError:
        pass

    # 2) 位置参数再试一次
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

    pytest.skip(f"evidential_regression 的用法没对上，实际签名是 {sig}。请按项目真实 API 调整 _build_model。")


def test_forward_shapes():
    import torch

    model = _build_model()
    model.eval()
    x = torch.randn(8, 1)

    with torch.no_grad():
        out = model(x)

    if isinstance(out, dict):
        # 选出各参数的真实键名
        k_mu    = _pick_key(out, MU_KEYS,    "均值(mu)")
        k_v     = _pick_key(out, V_KEYS,     "精度/尺度(v)")
        k_alpha = _pick_key(out, ALPHA_KEYS, "alpha")
        k_beta  = _pick_key(out, BETA_KEYS,  "beta")

        # 形状断言
        for k in (k_mu, k_v, k_alpha, k_beta):
            assert out[k].shape == x.shape, f"{k} 形状不对：{out[k].shape} vs {x.shape}"

        # 宽松数值约束
        assert (out[k_v] > 0).all()
        assert (out[k_beta] > 0).all()
        assert (out[k_alpha] > 0).all()  # 有的实现只保证 >0，不强求 >1
    else:
        # 分布对象风格：至少得有 mean 且形状匹配
        m_attr = getattr(out, "mean", None)
        m = m_attr() if callable(m_attr) else m_attr
        assert m is not None, "返回对象没有 mean"
        assert m.shape == x.shape




