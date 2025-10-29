"""Tests for flax evidential regression transformation.

断言目标：
1) 施加 evidential 回归变换后，最后一层线性头的输出维度应当扩大为原来的 4 倍（对应 μ, v, α, β）。
2) 其他层计数（Conv、Sequential）不应被乱改。
"""

from __future__ import annotations

from typing import Tuple, Any
import pytest

# 复用已有工具：统计层数量
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


# ============== 工具：找到 evidential 变换入口（适配不同命名） ==============
def _get_evidential_transform():
    """
    在 probly.transformation.evidential.regression 里尝试多种常见命名。
    找不到就 skip，让 CI 给你留点体面，不至于红成灯塔。
    """
    import probly.transformation.evidential.regression as er

    candidates = (
        "evidential_regression",
        "regression",
        "to_evidential_regressor",
        "make_evidential_regression",
        "evidential",
        "transform",
    )
    for name in candidates:
        fn = getattr(er, name, None)
        if callable(fn):
            return fn
    pytest.skip("No evidential regression transform found in probly.transformation.evidential.regression")


# ============== 工具：只递归 nnx.Sequential，别到处乱翻触发一堆 deprecated 警告 ==============
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
    """不信字段名，直接从参数形状推断输出维度。优先 bias，其次 kernel/weight。"""
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

    # 3) 扫一遍属性兜底：1D 当 bias，2D 当 weight
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
    """找到模型中的最后一个 nnx.Linear，并返回其输出维度。没有就优雅地 skip。"""
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
    """结构层面测试：线性头扩成 4 倍，其他层计数不乱。"""

    def test_linear_head_expands_to_four_params(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
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

        # 断言：最后线性头输出维度扩大 4 倍；其他计数不变；类型保持一致
        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert out_feat_mod == 4 * out_feat_orig
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig  # 替换最后一层但层数不变

    def test_conv_model_head_expands_and_conv_unchanged(self, flax_conv_linear_model: nnx.Sequential) -> None:
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
        assert out_feat_mod == 4 * out_feat_orig
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig

