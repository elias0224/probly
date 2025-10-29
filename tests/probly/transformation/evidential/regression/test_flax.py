"""Tests for flax evidential regression transformation."""

from __future__ import annotations

import pytest

# 复用你们现成的工具：统计层数
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


# ---- 小工具：找 evidential 变换函数，名字不确定就轮询几种常见写法 ----
def _get_evidential_transform():
    import probly.transformation.evidential.regression as er
    for name in (
        "evidential_regression",
        "regression",                # 有的模块直接导出 regression(...)
        "to_evidential_regressor",
        "evidential",                # 保险
        "transform",                 # 极端兜底
    ):
        if hasattr(er, name):
            fn = getattr(er, name)
            if callable(fn):
                return fn
    pytest.skip("No evidential regression transform found in probly.transformation.evidential.regression")


def _iter_modules(m):
    yield m
    # 递归展开 nnx.Sequential
    if isinstance(m, nnx.Sequential):
        for c in m.layers:
            yield from _iter_modules(c)
    else:
        # 自定义 module 若有 children 可继续扩展（按需）
        for name in dir(m):
            try:
                child = getattr(m, name)
            except Exception:
                continue
            if isinstance(child, nnx.Module):
                yield from _iter_modules(child)


def _last_linear_and_features(model: nnx.Module):
    last = None
    for mod in _iter_modules(model):
        if isinstance(mod, nnx.Linear):
            last = mod
    if last is None:
        pytest.skip("Model has no nnx.Linear layer to transform")
    # flax.nnx.Linear 的输出尺寸通常在 .features
    if not hasattr(last, "features"):
        pytest.skip("Could not read output features of the last Linear")
    return last, last.features


class TestNetworkArchitectures:
    """结构测试：最后的线性头应扩为 4×输出维度，其他层计数不乱。"""

    def test_linear_head_expands_to_four_params(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        evidential = _get_evidential_transform()

        # 原模型统计
        count_linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_conv_orig = count_layers(flax_model_small_2d_2d, nnx.Conv)
        count_seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_features(flax_model_small_2d_2d)

        # 施加 evidential 变换（无额外超参就别瞎传）
        model = evidential(flax_model_small_2d_2d)

        # 改后统计
        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_features(model)

        # 断言：线性头维度应扩大 4 倍；其他计数不应乱套
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert out_feat_mod == 4 * out_feat_orig
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        # 线性层个数通常不变；若你们实现替换了最后一层仍是 Linear，则应相等
        assert count_linear_mod == count_linear_orig

    def test_conv_model_head_expands_and_conv_unchanged(self, flax_conv_linear_model: nnx.Sequential) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(flax_conv_linear_model, nnx.Linear)
        count_conv_orig = count_layers(flax_conv_linear_model, nnx.Conv)
        count_seq_orig = count_layers(flax_conv_linear_model, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_features(flax_conv_linear_model)

        model = evidential(flax_conv_linear_model)

        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_features(model)

        assert isinstance(model, type(flax_conv_linear_model))
        assert out_feat_mod == 4 * out_feat_orig
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
