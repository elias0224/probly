import pytest
torch = pytest.importorskip("torch")

def _build_model(base_backbone):
    try:
        from probly.transformation.evidential.regression import EvidentialRegression
        return EvidentialRegression(base_backbone)
    except ImportError:
        from probly.transformation.evidential.regression import make_evidential_regression
        return make_evidential_regression(base_backbone)

def test_forward_output_and_constraints(base_backbone, toy_regression_data, device):
    x, _ = toy_regression_data
    model = _build_model(base_backbone).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    if isinstance(out, dict):
        for k in ["mu", "v", "alpha", "beta"]:
            assert k in out
            assert out[k].shape == x.shape
    
        assert torch.all(out["v"] > 0)
        assert torch.all(out["alpha"] > 1) or torch.all(out["alpha"] > 0)
        assert torch.all(out["beta"] > 0)
    else:
        assert hasattr(out, "mean")
        m = out.mean
        assert m.shape == x.shape



