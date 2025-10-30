"""Test for ensemble models."""

from __future__ import annotations

import pytest

from probly.predictor import Predictor
from probly.transformation import dropout


def test_invalid_type() -> None:
    """Test that an invalid type raises NotImplementedError."""

    class InvalidPredictor(Predictor[int, None, int]):
        def predict(self, inputs: int) -> int:
            return inputs

    invalid_model = InvalidPredictor()

    with pytest.raises(NotImplementedError, match="No ensemble generator is registered for type"):
        dropout.ensemble(invalid_model, n_members=5)
