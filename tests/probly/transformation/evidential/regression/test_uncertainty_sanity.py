import pytest
torch = pytest.importorskip("torch")

def _variance_from_nig(v, alpha, beta):
    return beta / torch.clamp(alpha - 1, min=1e-3)

def _build_model(base_backbone):
    try:
        from probly.transformation.evidential.regression import EvidentialRegression
        return EvidentialRegression(base_backbone)
    except ImportError:
        from probly.transformation.evidential.regression import make_evidential_regression
        return make_evidential_regression(base_backbone)

def test_uncertainty_correlates_with_residual(base_backbone, toy_regression_data):
    x, y = toy_regression_data
    model = _build_model(base_backbone)
    model.eval()
    with torch.no_grad():
        out = model(x)
        if not isinstance(out, dict):
            pytest.skip("Distribution object returned; adapt this test accordingly.")
        mu = out["mu"]
        var = _variance_from_nig(out["v"], out["alpha"], out["beta"])
        resid = (y - mu).abs()
        q = resid.quantile(0.75)
        high_mask = resid >= q
        low_mask = resid <= resid.quantile(0.25)
        assert var[high_mask].mean() > var[low_mask].mean()
