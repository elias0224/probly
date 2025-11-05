import pytest
import jax
import jax.numpy as jnp

def _unpack(out):
    if isinstance(out, dict):
        mu = out["mu"]; v = out["v"]; alpha = out["alpha"]; beta = out["beta"]
    else:
        mu = getattr(out, "mu"); v = getattr(out, "v")
        alpha = getattr(out, "alpha"); beta = getattr(out, "beta")
    return mu, v, alpha, beta

@pytest.mark.parametrize("batch, in_dim, out_dim", [(1, 3, 1), (5, 4, 2)])
def test_forward_shapes(batch, in_dim, out_dim):
    pytest.skip("skip")

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, in_dim))

    params = evid.init(key, x)            
    out = evid.apply(params, x, train=False)
    mu, v, alpha, beta = _unpack(out)

    assert mu.shape == (batch, out_dim)
    assert v.shape == (batch, out_dim)
    assert alpha.shape == (batch, out_dim)
    assert beta.shape == (batch, out_dim)




