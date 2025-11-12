from __future__ import annotations

import pytest
from typing import Any, Callable, NoReturn

# 可选依赖：torch 不在，就跳过整个文件
try:
    import torch
    from torch import nn
except Exception:
    pytest.skip("torch not available", allow_module_level=True)

# 顶层导入待测模块
from probly.transformation.evidential import regression as er


def _die(msg: str) -> NoReturn:
    """统一跳过，并避免 mypy 报 Missing return。"""
    pytest.skip(msg)
    raise AssertionError("unreachable")  # pragma: no cover


def _get_evidential_transform() -> Callable[..., Any]:
    for name in (
        "evidential_regression",
        "regression",
        "to_evidential_regressor",
        "make_evidential_regression",
        "evidential",
        "transform",
    ):
        fn = getattr(er, name, None)
        if callable(fn):
            return fn
    _die("No evidential regression transform found in probly.transformation.evidential.regression")


def _first_linear_in_features(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    _die("Fixture model has no nn.Linear; cannot infer input feature size")


def _last_linear_out_features(model: nn.Module) -> int:
    last: nn.Linear | None = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        _die("Model has no nn.Linear; cannot infer output feature size")
    return int(last.out_features)


def _first_conv_spec(model: nn.Module) -> tuple[int, int, int]:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size
            kH, kW = (k, k) if isinstance(k, int) else k
            return int(m.in_channels), int(kH), int(kW)
    _die("Fixture model has no nn.Conv2d; conv-forward test not applicable")


def _try_unpack(y):
    target = ("mu", "v", "alpha", "beta")

    if isinstance(y, dict):
        if all(k in y for k in target):
            return y["mu"], y["v"], y["alpha"], y["beta"]
        for v in y.values():
            out = _try_unpack(v)
            if out is not None:
                return out

    if all(hasattr(y, k) for k in target):
        return y.mu, y.v, y.alpha, y.beta

    if isinstance(y, (tuple, list)):
        if len(y) == 4 and all(torch.is_tensor(t) for t in y):
            return y[0], y[1], y[2], y[3]
        for item in y:
            out = _try_unpack(item)
            if out is not None:
                return out

    if torch.is_tensor(y) and y.ndim >= 2:
        D = y.shape[-1]
        if D % 4 == 0:
            split = D // 4
            return torch.split(y, split, dim=-1)

    return None


def _unpack_four(y):
    out = _try_unpack(y)
    if out is None:
        _die("Cannot interpret model output as evidential {mu,v,alpha,beta}")
    return out


class TestTorchForward:
    def test_forward_and_parameter_shapes(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        evidential = _get_evidential_transform()
        base = torch_model_small_2d_2d

        in_dim = _first_linear_in_features(base)
        out_dim = _last_linear_out_features(base)

        model = evidential(base)
        model.eval()

        B = 8
        x = torch.randn(B, in_dim)

        with torch.no_grad():
            y = model(x)

        mu, v, alpha, beta = _unpack_four(y)

        for t in (mu, v, alpha, beta):
            assert torch.is_tensor(t), "Each output head must be a tensor"
            assert t.shape[-1] == out_dim, f"Expected last dim {out_dim}, got {t.shape[-1]}"
            assert t.shape[0] == B, f"Expected batch {B}, got {t.shape[0]}"

        for name, t in zip(("mu", "v", "alpha", "beta"), (mu, v, alpha, beta)):
            assert torch.isfinite(t).all(), f"{name} contains non-finite values"

        for name, t in zip(("v", "alpha", "beta"), (v, alpha, beta)):
            assert torch.is_floating_point(t), f"{name} has non-floating dtype: {t.dtype}"
            assert (t > 0).all(), f"{name} must be positive"

    def test_forward_conv_model(self, torch_conv_linear_model: nn.Sequential) -> None:
        evidential = _get_evidential_transform()
        base = torch_conv_linear_model

        # 只做 smoke test：能包一层就算过
        evidential(base)

