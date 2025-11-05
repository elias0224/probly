"""Test for flax ensemble transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probly.transformation import ensemble
from tests.probly.flax_utils import count_layers

flax = pytest.importorskip("flax")
from flax import nnx  # noqa: E402


class TestNetworkStructure:
    """Test class for network structure tests."""

    def test_linear_network_no_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the linear model ensemble without resetting parameters."""
        num_members = 4
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=False)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_model_small_2d_2d, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)

            assert member is not None
            assert isinstance(member, type(flax_model_small_2d_2d))
            assert count_linear_modified == count_linear_original
            assert count_dropout_modified == count_dropout_original
            assert count_sequential_modified == count_sequential_original

    def test_linear_network_with_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests the linear model ensemble with resetting parameters."""
        num_members = 4
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_model_small_2d_2d, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_model_small_2d_2d, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_model_small_2d_2d, nnx.Sequential)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)

            assert member is not None
            assert isinstance(member, type(flax_model_small_2d_2d))
            assert count_linear_modified == count_linear_original
            assert count_dropout_modified == count_dropout_original
            assert count_sequential_modified == count_sequential_original

    def test_conv_linear_network_no_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the conv-linear model ensemble without resetting parameters."""
        model = ensemble(flax_conv_linear_model, num_members=3, reset_params=False)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_conv_linear_model, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)
            # count number of nnx.Conv2d layers in modified model
            count_conv_modified = count_layers(member, nnx.Conv)

            # check that the model is not modified except for the dropout layer
            assert member is not None
            assert isinstance(member, type(flax_conv_linear_model))
            assert count_dropout_original == count_dropout_modified
            assert count_linear_original == count_linear_modified
            assert count_sequential_original == count_sequential_modified
            assert count_conv_original == count_conv_modified

    def test_conv_linear_network_with_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests the conv-linear model ensemble with resetting parameters."""
        model = ensemble(flax_conv_linear_model, num_members=3, reset_params=True)

        # count number of nnx.Linear layers in original model
        count_linear_original = count_layers(flax_conv_linear_model, nnx.Linear)
        # count number of nnx.Dropout layers in original model
        count_dropout_original = count_layers(flax_conv_linear_model, nnx.Dropout)
        # count number of nnx.Sequential layers in original model
        count_sequential_original = count_layers(flax_conv_linear_model, nnx.Sequential)
        # count number of nnx.Conv2d layers in original model
        count_conv_original = count_layers(flax_conv_linear_model, nnx.Conv)

        for member in model:
            # count number of nnx.Linear layers in modified model
            count_linear_modified = count_layers(member, nnx.Linear)
            # count number of nnx.Dropout layers in modified model
            count_dropout_modified = count_layers(member, nnx.Dropout)
            # count number of nnx.Sequential layers in modified model
            count_sequential_modified = count_layers(member, nnx.Sequential)
            # count number of nnx.Conv2d layers in modified model
            count_conv_modified = count_layers(member, nnx.Conv)

            # check that the model is not modified except for the dropout layer
            assert member is not None
            assert isinstance(member, type(flax_conv_linear_model))
            assert count_dropout_original == count_dropout_modified
            assert count_linear_original == count_linear_modified
            assert count_sequential_original == count_sequential_modified
            assert count_conv_original == count_conv_modified

    def test_custom_network_with_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model essemble with resetting parameters."""
        num_members = 5
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=True)

        # check if model type is correct
        for member in model:
            assert isinstance(member, type(flax_custom_model))
            assert not isinstance(member, nnx.Sequential)

    def test_custom_network_no_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests the custom model essemble without resetting parameters."""
        num_members = 5
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        for member in model:
            assert isinstance(member, type(flax_custom_model))
            assert not isinstance(member, nnx.Sequential)


class TestParameters:
    """Test class for network parameter tests."""

    def test_parameters_linear_network_no_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=False)

        _, original_params = nnx.split(flax_model_small_2d_2d, nnx.Param)

        for member in model:
            _, member_params = nnx.split(member, nnx.Param)
            for orig, memb in zip(
                jax.tree_util.tree_leaves(original_params),
                jax.tree_util.tree_leaves(member_params),
                strict=False,
            ):
                assert jnp.array_equal(orig, memb)

    def test_parameters_linear_network_with_reset(self, flax_model_small_2d_2d: nnx.Sequential) -> None:
        """Tests that parameters are the the same reseeded when reset_params is True."""
        num_members = 3
        model = ensemble(flax_model_small_2d_2d, num_members=num_members, reset_params=True)

        dummy = jnp.ones((1, 2))

        for i in range(num_members - 1):
            nnx.reseed(flax_model_small_2d_2d, policy="match_shape", rng=jax.random.PRNGKey(i))
            reseeded_flax_model_small_2d_2d = flax_model_small_2d_2d(dummy)
            reseed_model = model[i](dummy)

            _, random_params = nnx.split(reseeded_flax_model_small_2d_2d)
            _, member_params = nnx.split(reseed_model)

            for reseed, memb in zip(
                jax.tree_util.tree_leaves(random_params),
                jax.tree_util.tree_leaves(member_params),
                strict=False,
            ):
                assert jnp.array_equal(reseed, memb)

    def test_parameters_conv_linear_network_no_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_conv_linear_model, num_members=num_members, reset_params=False)

        _, original_params = nnx.split(flax_conv_linear_model, nnx.Param)

        for member in model:
            _, member_params = nnx.split(member, nnx.Param)
            for orig, memb in zip(
                jax.tree_util.tree_leaves(original_params),
                jax.tree_util.tree_leaves(member_params),
                strict=False,
            ):
                assert jnp.array_equal(orig, memb)

    """
    def test_parameters_conv_linear_network_with_reset(self, flax_conv_linear_model: nnx.Sequential) -> None:

        num_members = 3
        model= ensemble(flax_conv_linear_model, num_members=num_members, reset_params=True)

        dummy = jnp.zeros((1, 5, 5, 3), dtype=jnp.float32)  # type: ignore

        for i in range(num_members - 1):

            nnx.reseed(flax_conv_linear_model, policy='match_shape', rng=jax.random.PRNGKey(i))

            reseeded_flax_conv_model = flax_conv_linear_model(dummy)  # type: ignore
            reseed_model = model[i](dummy)  # type: ignore

            _, reseed_params = nnx.split(reseeded_flax_conv_model)
            _, member_params= nnx.split(reseed_model)

            for reseed, memb in zip(jax.tree_util.tree_leaves(reseed_params), jax.tree_util.tree_leaves(member_params)):
                assert jnp.array_equal(reseed, memb)
    """

    def test_parameters_custom_network_no_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests that parameters are the same when reset_params is False."""
        num_members = 3
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=False)

        _, original_params = nnx.split(flax_custom_model, nnx.Param)

        for member in model:
            _, member_params = nnx.split(member, nnx.Param)
            for orig, memb in zip(
                jax.tree_util.tree_leaves(original_params),
                jax.tree_util.tree_leaves(member_params),
                strict=False,
            ):
                assert jnp.array_equal(orig, memb)

    def test_parameters_custom_network_with_reset(self, flax_custom_model: nnx.Module) -> None:
        """Tests that parameters are the the same reseeded when reset_params is True."""
        num_members = 3
        model = ensemble(flax_custom_model, num_members=num_members, reset_params=True)

        dummy = jnp.ones((1, 10))

        for i in range(num_members - 1):
            nnx.reseed(flax_custom_model, policy="match_shape", rng=jax.random.PRNGKey(i))
            reseeded_flax_custom_model = flax_custom_model(dummy)
            reseed_model = model[i](dummy)

            _, reseed_params = nnx.split(reseeded_flax_custom_model)
            _, member_params = nnx.split(reseed_model)
            for reseed, memb in zip(
                jax.tree_util.tree_leaves(reseed_params),
                jax.tree_util.tree_leaves(member_params),
                strict=False,
            ):
                assert jnp.array_equal(reseed, memb)
