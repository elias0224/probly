"""Test for torch ensemble models."""

from __future__ import annotations

import pytest
from torch import nn

from probly.transformation import ensemble
from tests.probly.torch_utils import count_layers

torch = pytest.importorskip("torch")


def test_linear_network_with_first_linear(torch_model_small_2d_2d: nn.Sequential) -> None:
    n_members = 5
    model = ensemble(torch_model_small_2d_2d, n_members=n_members)

    # count
    count_linear_original = count_layers(torch_model_small_2d_2d, nn.Linear)
    count_sequential_original = count_layers(torch_model_small_2d_2d, nn.Sequential)
    count_linear_modified = count_layers(model, nn.Linear)
    count_sequential_modified = count_layers(model, nn.Sequential)

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert (count_linear_original * n_members) == count_linear_modified
    assert (count_sequential_original * n_members) == count_sequential_modified


def test_convolutional_network(torch_conv_linear_model: nn.Sequential) -> None:
    n_members = 5
    model = ensemble(torch_conv_linear_model, n_members=n_members)

    # count
    count_linear_original = count_layers(torch_conv_linear_model, nn.Linear)
    count_sequential_original = count_layers(torch_conv_linear_model, nn.Sequential)
    count_linear_modified = count_layers(model, nn.Linear)
    count_sequential_modified = count_layers(model, nn.Sequential)

    # check that the model is not modified except for the dropout layer
    assert model is not None
    assert (count_linear_original * n_members) == count_linear_modified
    assert (count_sequential_original * n_members) == count_sequential_modified
