from __future__ import annotations

import pytest
from typing import Tuple, Any, NoReturn

# 依赖可选
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402

from tests.probly.flax_utils import count_layers
from probly.transformation.evidential import regression as er


def _die(msg: str) -> NoReturn:
    pytest.skip(msg)
    raise AssertionError("unreachable")  # pragma: no cover


def _get_evidential_transform():
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


def _iter_modules(m: nnx.Module):
    yield m
    if isinstance(m, nnx.Sequential):
        for c in m.layers:
            yield from _iter_modules(c)


def _maybe_array(x: Any):
    if hasattr(x, "value"):
        try:
            return x.value
        except Exception:
            return None
    return x


def _linear_in_out_by_params(layer: nnx.Linear) -> Tuple[int, int]:
    # 先尝试 bias 推断 out_features
    for name in ("bias", "b"):
        if hasattr(layer, name):
            arr = _maybe_array(getattr(layer, name))
            if arr is not None and getattr(arr, "ndim", 0) == 1:
                out_features = int(arr.shape[0])
                return -1, out_features  # in_features 未知，用 -1 占位

    # 再尝试 kernel/weight
    for name in ("kernel", "weight", "w"):
        if hasattr(layer, name):
            arr = _maybe_array(getattr(layer, name))
            if arr is not None and getattr(arr, "ndim", 0) == 2:
                return int(arr.shape[0]), int(arr.shape[1])

    # 最后穷举属性里可能的数组
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

    _die("Cannot infer in/out features from nnx.Linear parameters")


def _last_linear_and_out_features(model: nnx.Module) -> Tuple[nnx.Linear, int]:
    last: nnx.Linear | None = None
    for mod in _iter_modules(model):
        if isinstance(mod, nnx.Linear):
            last = mod
    if last is None:
        _die("Model has no nnx.Linear layer to transform")
    _, out_feat = _linear_in_out_by_params(last)
    if out_feat in (-1, None):
        _die("Could not determine output features of the last Linear")
    return last, out_feat


class TestNetworkArchitectures:
    def test_linear_head_kept_and_structure_unchanged(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        evidential = _get_evidential_transform()

        count_linear_orig = count_layers(flax_model_small_2d_2d, nnx.Linear)
        count_conv_orig = count_layers(flax_model_small_2d_2d, nnx.Conv)
        count_seq_orig = count_layers(flax_model_small_2d_2d, nnx.Sequential)
        _, out_feat_orig = _last_linear_and_out_features(flax_model_small_2d_2d)

        model = evidential(flax_model_small_2d_2d)

        count_linear_mod = count_layers(model, nnx.Linear)
        count_conv_mod = count_layers(model, nnx.Conv)
        count_seq_mod = count_layers(model, nnx.Sequential)
        _, out_feat_mod = _last_linear_and_out_features(model)

        assert model is not None
        assert isinstance(model, type(flax_model_small_2d_2d))
        assert count_conv_mod == count_conv_orig
        assert count_seq_mod == count_seq_orig
        assert count_linear_mod == count_linear_orig
        assert out_feat_mod == out_feat_orig

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
