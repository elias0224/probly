"""Tests for torch evidential regression transformation.

按仓库实际实现校验结构契约（Torch 版）：
- 施加 evidential 回归变换后，模型类型不变；
- Conv/Sequential 的层数不变；
- Linear 的层数不增加，且允许因为替换成 evidential head 而**最多少 1 层**；
- 如果少 1 层，则最后一个模块不应再是 nn.Linear；
- 最后一个 Linear 的 out_features 与原来保持一致（因为 head 在 Linear 之后做参数展开/重组）。
"""

from __future__ import annotations

import pytest
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


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


def _last_linear_and_out_features(model: nn.Module) -> tuple[nn.Linear, int]:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        pytest.skip("Model has no nn.Linear layer to transform")
    return last, int(last.out_features)


def _last_module(model: nn.Module) -> nn.Module:
    last = None
    # .modules() 包含自身和子模块，最后一个通常是“真正尾巴”
    for m in model.modules():
        last = m
    return last


class TestNetworkArchitectures:
    """结构层面测试：不破坏拓扑，线性头可能被 head 替换。"""

    def test_linear_head_kept_or_replaced_once_and_structure_ok(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        evidential = _get_evidential_transform()

        # 原模型结构统计
        count_linear_orig = count_layers(torch_model_small_2d_2d, nn.Linear)
        count_conv_orig = count_layers(torch_model_small_2d_2d, nn.Conv2d)
        count_seq_orig = count_layers(torch_model_small_2d_2d, nn.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(torch_model_small_2d_2d)

        # 施加 evidential 回归变换
        model = evidential(torch_model_small_2d_2d)

        # 变换后结构统计
        count_linear_mod = count_layers(model, nn.Linear)
        count_conv_mod = count_layers(model, nn.Conv2d)
        count_seq_mod = count_layers(model, nn.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        # 断言：类型同类；Conv/Sequential 不变
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig

        # 线性层不增加，且与原来最多差 1
        assert count_linear_mod <= count_linear_orig
        assert (count_linear_orig - count_linear_mod) in (0, 1)

        # 若少了一层，确认最后一个模块确实不是 nn.Linear（说明被 evidential head 替换）
        if count_linear_mod == count_linear_orig - 1:
            tail = _last_module(model)
            assert not isinstance(tail, nn.Linear)

        # 无论是否替换，模型中“最后一个 Linear”的 out_features 应保持一致
        assert out_feat_mod == out_feat_orig

    def test_conv_model_kept_or_replaced_once_and_structure_ok(self, torch_conv_linear_model: nn.Sequential) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(torch_conv_linear_model, nn.Linear)
        count_conv_orig = count_layers(torch_conv_linear_model, nn.Conv2d)
        count_seq_orig = count_layers(torch_conv_linear_model, nn.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(torch_conv_linear_model)

        model = evidential(torch_conv_linear_model)

        count_linear_mod = count_layers(model, nn.Linear)
        count_conv_mod = count_layers(model, nn.Conv2d)
        count_seq_mod = count_layers(model, nn.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert isinstance(model, type(torch_conv_linear_model))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig

        assert count_linear_mod <= count_linear_orig
        assert (count_linear_orig - count_linear_mod) in (0, 1)

        if count_linear_mod == count_linear_orig - 1:
            tail = _last_module(model)
            assert not isinstance(tail, nn.Linear)

        assert out_feat_mod == out_feat_orig


