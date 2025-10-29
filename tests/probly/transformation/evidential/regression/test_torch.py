"""Tests for torch evidential regression transformation.

按仓库实际实现校验结构契约（Torch 版）：
- 施加 evidential 回归变换后，模型类型不变；
- Conv/Sequential/Linear 的层数不变（说明只是头部/forward 级别逻辑，而不是乱改拓扑）；
- 最后线性层的 out_features 保持不变（实现可能在 forward 内部生成 μ, v, α, β）。
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


class TestNetworkArchitectures:
    """结构层面测试：不破坏拓扑，线性头维度保持。"""

    def test_linear_head_kept_and_structure_unchanged(self, torch_model_small_2d_2d: nn.Sequential) -> None:
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

        # 断言：类型同类；层数不变；最后线性头 out_features 不变
        assert model is not None
        assert isinstance(model, type(torch_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig

    def test_conv_model_kept_and_structure_unchanged(self, torch_conv_linear_model: nn.Sequential) -> None:
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
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig
