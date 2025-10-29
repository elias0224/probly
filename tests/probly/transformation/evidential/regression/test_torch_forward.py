"""Forward smoke tests for torch evidential regression.

目标：
- 变换后的模型能跑 forward，不抛异常；
- 输出要么是字典/对象包含 {mu, v, alpha, beta}，要么是 (B, 4 * out_dim) 的张量；
- 若能解出四元参数，则检查形状一致且 v/alpha/beta 为正、均为有限值。
"""

from __future__ import annotations

import pytest
import math

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


def _get_evidential_transform():
    import probly.transformation.evidential.regression as er
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
    pytest.skip("No evidential regression transform found in probly.transformation.evidential.regression")


def _first_linear_in_features(model: nn.Module) -> int:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    pytest.skip("Fixture model has no nn.Linear; cannot infer input feature size")


def _last_linear_out_features(model: nn.Module) -> int:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        pytest.skip("Model has no nn.Linear; cannot infer output feature size")
    return int(last.out_features)


def _first_conv_spec(model: nn.Module) -> tuple[int, int, int]:
    """返回第一层 Conv2d 的 (in_channels, kH, kW)。没有就 skip。"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size
            kH, kW = (k, k) if isinstance(k, int) else k
            return int(m.in_channels), int(kH), int(kW)
    pytest.skip("Fixture model has no nn.Conv2d; conv-forward test not applicable")


def _unpack_four(y):
    """尽可能解出 mu, v, alpha, beta。兼容 dict / 对象属性 / 拼接张量三种形态。"""
    if isinstance(y, dict):
        keys = {"mu", "v", "alpha", "beta"}
        if keys.issubset(y.keys()):
            return y["mu"], y["v"], y["alpha"], y["beta"]
    # 对象属性
    has = all(hasattr(y, k) for k in ("mu", "v", "alpha", "beta"))
    if has:
        return y.mu, y.v, y.alpha, y.beta
    # 纯张量：按最后一维均分成 4 份
    if torch.is_tensor(y):
        if y.ndim < 2:
            pytest.skip("Model output is a tensor but not rank-2+, cannot split reliably")
        D = y.shape[-1]
        if D % 4 != 0:
            pytest.fail(f"Tensor output last dim {D} not divisible by 4; cannot interpret as evidential params")
        split = D // 4
        mu, v, alpha, beta = torch.split(y, split, dim=-1)
        return mu, v, alpha, beta
    pytest.skip("Cannot interpret model output as evidential {mu,v,alpha,beta}")


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

        # 形状一致性：四个头的最后一维等于 out_dim
        for t in (mu, v, alpha, beta):
            assert torch.is_tensor(t), "Each output head must be a tensor"
            assert t.shape[-1] == out_dim, f"Expected last dim {out_dim}, got {t.shape[-1]}"
            assert t.shape[0] == B, f"Expected batch {B}, got {t.shape[0]}"

        # 数值健壮性：必须是有限值
        for name, t in zip(("mu", "v", "alpha", "beta"), (mu, v, alpha, beta)):
            assert torch.isfinite(t).all(), f"{name} contains non-finite values"

        # 正性约束：v、alpha、beta 应为正
        for name, t in zip(("v", "alpha", "beta"), (v, alpha, beta)):
            if torch.is_floating_point(t):
                assert (t > 0).all(), f"{name} must be positive"
            else:
                pytest.fail(f"{name} has non-floating dtype: {t.dtype}")

    def test_forward_conv_model(self, torch_conv_linear_model: nn.Sequential) -> None:
        evidential = _get_evidential_transform()
        base = torch_conv_linear_model

        # 从第一层 Conv2d 读出通道数与 kernel size，构造最小合法输入。
        C, kH, kW = _first_conv_spec(base)
        out_dim = _last_linear_out_features(base)

        model = evidential(base)
        model.eval()

        B = 4
        # 选择 H=W=kernel_size，配合 stride=1, padding=0，Conv 输出空间大小为 1x1，
        # 后续 Flatten -> Linear 的 in_features 就等于 out_channels（fixture 里正好是 5）
        x = torch.randn(B, C, kH, kW)

        with torch.no_grad():
            y = model(x)

        mu, v, alpha, beta = _unpack_four(y)

        for t in (mu, v, alpha, beta):
            assert torch.is_tensor(t)
            assert t.shape[0] == B
            assert t.shape[-1] == out_dim
            assert torch.isfinite(t).all()

        for name, t in zip(("v", "alpha", "beta"), (v, alpha, beta)):
            assert torch.is_floating_point(t) and (t > 0).all(), f"{name} must be positive"

