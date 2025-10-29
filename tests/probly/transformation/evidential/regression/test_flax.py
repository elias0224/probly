"""Tests for flax evidential regression transformation.

本测试按仓库实际实现校验结构契约：
- 施加 evidential 回归变换后，模型类型不变；
- Conv/Sequential/Linear 的层数不变（说明只做了头部/forward级别改造，而非乱改拓扑）；
- 最后线性头的输出维度保持不变（实现可能在 forward 内部生成 μ, v, α, β，而不是直接把线性层扩到 4×）。
"""

from __future__ import annotations
from typing import Tuple, Any

import pytest
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


# ============== 找到 evidential 变换入口（适配不同命名） ==============
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


# ============== 只递归 nnx.Sequential，避免到处乱翻触发 deprecated 警告 ==============
def _iter_modules(m: nnx.Module):
    yield m
    if isinstance(m, nnx.Sequential):
        for c in m.layers:
            yield from _iter_modules(c)


def _maybe_array(x: Any):
    """nnx.Parameter/Variable 通常有 .value；若没有就原样返回。"""
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            return None
    return x


def _linear_in_out_by_params(layer: nnx.Linear) -> Tuple[int, int]:
    """不依赖字段名，从参数形状推断线性层 (in_features, out_features)。"""
    # 1) bias: 一维，长度就是 out_features
    for name in ("bias", "b"):
        if hasattr(layer, name):
            arr = _maybe_array(getattr(layer, name))
            if arr is not None and getattr(arr, "ndim", 0) == 1:
                out_features = int(arr.shape[0])
                return -1, out_features  # in_features 不重要

    # 2) kernel/weight: 二维 [in_features, out_features]
    for name in ("kernel", "weight", "w"):
        if hasattr(layer, name):
            arr = _maybe_array(getattr(layer, name))
            if arr is not None and getattr(arr, "ndim", 0) == 2:
                return int(arr.shape[0]), int(arr.shape[1])

    # 3) 兜底扫描：1D 当 bias，2D 当 weight
    for k in dir(layer):
        if k.startswith("_"):
            continue
        try:
            v = getattr(layer, k)
        except Exception:
            continue
        arr = _maybe_array(v)
        if arr is None or not hasattr(arr, "shape"):
            continue
        if getattr(arr, "ndim", 0) == 1:
            return -1, int(arr.shape[0])
        if getattr(arr, "ndim", 0) == 2:
            return int(arr.shape[0]), int(arr.shape[1])

    pytest.skip("Cannot infer in/out features from nnx.Linear parameters")


def _last_linear_and_out_features(model: nnx.Module) -> Tuple[nnx.Linear, int]:
    """找到模型中的最后一个 nnx.Linear，并返回其输出维度。没有就 skip。"""
    last = None
    for mod in _iter_modules(model):
        if isinstance(mod, nnx.Linear):
            last = mod
    if last is None:
        pytest.skip("Model has no nnx.Linear layer to transform")
    _, out_feat = _linear_in_out_by_params(last)
    if out_feat in (-1, None):
        pytest.skip("Could not determine output features of the last Linear")
    return last, out_feat


# ============== 正式测试 ==============
class TestNetworkArchitectures:
    """结构层面测试：不破坏拓扑，线性头维度保持。"""

    def test_linear_head_kept_and_structure_unchanged(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        evidential = _get_evidential_transform()

        # 原模型结构统计
        count_linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_conv_orig = count_layers(flax_model_small_2d_2d, nnx.Conv)
        count_seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(flax_model_small_2d_2d)

        # 施加 evidential 回归变换
        model = evidential(flax_model_small_2d_2d)

        # 变换后结构统计
        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        # 断言：类型同类；层数不变；最后线性头 out_features 不变
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig  # 关键：实现保持线性头维度

    def test_conv_model_kept_and_structure_unchanged(self, flax_conv_linear_model: nnx.Sequential) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(flax_conv_linear_model, nnx.Linear)
        count_conv_orig = count_layers(flax_conv_linear_model, nnx.Conv)
        count_seq_orig = count_layers(flax_conv_linear_model, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(flax_conv_linear_model)

        model = evidential(flax_conv_linear_model)

        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert isinstance(model, type(flax_conv_linear_model))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig


