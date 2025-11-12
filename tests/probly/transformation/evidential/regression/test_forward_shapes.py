# tests/probly/transformation/evidential/regression/test_forward_shapes.py
import pytest
import jax
import jax.numpy as jnp

from probly.transformation.evidential.regression import evid


def _unpack(out):
    if isinstance(out, dict):
        mu = out["mu"]; v = out["v"]; alpha = out["alpha"]; beta = out["beta"]
    else:
        mu = getattr(out, "mu"); v = getattr(out, "v")
        alpha = getattr(out, "alpha"); beta = getattr(out, "beta")
    return mu, v, alpha, beta


@pytest.mark.parametrize("batch, in_dim, out_dim", [(1, 3, 1), (5, 4, 2)])
def test_forward_shapes(batch, in_dim, out_dim):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, in_dim))

    params = evid.init(key, x, out_dim=out_dim)
    out = evid.apply(params, x, train=False)
    mu, v, alpha, beta = _unpack(out)

    assert mu.shape == (batch, out_dim)
    assert v.shape == (batch, out_dim)
    assert alpha.shape == (batch, out_dim)
    assert beta.shape == (batch, out_dim)

    assert jnp.all(v > 0), "v must be positive"
    assert jnp.all(alpha > 1), "alpha must be > 1"
    assert jnp.all(beta > 0), "beta must be positive"

    y = jax.random.normal(key, (batch, out_dim))
    nll = evid.loss_nll(y, out)
    assert nll.shape == (), "loss_nll must return a scalar"

    total = evid.loss_total(y, out, lambda_reg=1e-3)
    assert total.shape == ()




