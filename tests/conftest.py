"""Test fixtures for probly."""

# no-op edit to benchmark warm incremental docs build (not docs-relevant)

from __future__ import annotations

pytest_plugins = [
    "tests.probly.fixtures.common",
    "tests.probly.fixtures.torch_models",
    "tests.probly.fixtures.torch_data",
    "tests.probly.fixtures.flax_models",
    "tests.probly.fixtures.sklearn_models",
    "tests.probly.fixtures.samples",
    "tests.probly.fixtures.torch_samples",
    "tests.probly.fixtures.jax_samples",
]
