import pytest
torch = pytest.importorskip("torch")

def _build_model(base_backbone):
    try:
        from probly.transformation.evidential.regression import EvidentialRegression, evidential_regression_loss
        return EvidentialRegression(base_backbone), evidential_regression_loss
    except ImportError:
        from probly.transformation.evidential.regression import make_evidential_regression, evidential_regression_loss
        return make_evidential_regression(base_backbone), evidential_regression_loss

@pytest.mark.timeout(15)
def test_toy_training_decreases_loss(base_backbone, toy_regression_data, device):
    x, y = toy_regression_data
    model, loss_fn = _build_model(base_backbone)
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        out = model(x)
        try:
            loss = loss_fn(out, y)
        except TypeError:
            if isinstance(out, dict):
                loss = loss_fn(out["mu"], out["v"], out["alpha"], out["beta"], y)
            else:
                raise
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    assert losses[0] > losses[-1] * 1.05  
