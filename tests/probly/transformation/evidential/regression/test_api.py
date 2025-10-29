import pytest
torch = pytest.importorskip("torch")

def test_transform_constructs(base_backbone):
    try:
        from probly.transformation.evidential.regression import EvidentialRegression
        model = EvidentialRegression(base_backbone)
    except ImportError:
        from probly.transformation.evidential.regression import make_evidential_regression
        model = make_evidential_regression(base_backbone)

    assert hasattr(model, "forward")
    assert callable(model.forward)
    model.train()
    model.eval()
