import pytest
torch = pytest.importorskip("torch")

def test_bad_hparams_raise(base_backbone):
  try:
        from probly.transformation.evidential.regression import EvidentialRegression
        with pytest.raises((ValueError, AssertionError)):
            EvidentialRegression(base_backbone, evidence_reg_weight=-1.0)
    except ImportError:
        from probly.transformation.evidential.regression import make_evidential_regression
        with pytest.raises((ValueError, AssertionError)):
            make_evidential_regression(base_backbone, evidence_reg_weight=-1.0)
