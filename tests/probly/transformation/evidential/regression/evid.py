# probly/transformation/evidential/regression/evid.py
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.logaddexp(x, 0.0)


def init(
    key: jax.Array,
    x: jnp.ndarray,
    out_dim: int | None = None,
    *,
    scale: float = 0.05,
) -> dict:
    in_dim = int(x.shape[-1])
    if out_dim is None:
        out_dim = 1

    k_mu, k_v, k_a, k_b, k_bmu, k_bv, k_ba, k_bb = jax.random.split(key, 8)

    def _w(k):
        return jax.random.normal(k, (in_dim, out_dim)) * scale

    def _b(k):
        return jax.random.normal(k, (out_dim,)) * scale

    params = {
        "W_mu": _w(k_mu),
        "W_v": _w(k_v),
        "W_alpha": _w(k_a),
        "W_beta": _w(k_b),
        "b_mu": _b(k_bmu),
        "b_v": _b(k_bv),
        "b_alpha": _b(k_ba),
        "b_beta": _b(k_bb),
        "out_dim": out_dim,
    }
    return params


def apply(params: dict, x: jnp.ndarray, *, train: bool = False) -> dict:
    W_mu, b_mu = params["W_mu"], params["b_mu"]
    W_v, b_v = params["W_v"], params["b_v"]
    W_a, b_a = params["W_alpha"], params["b_alpha"]
    W_b, b_b = params["W_beta"], params["b_beta"]

    logits_mu = x @ W_mu + b_mu
    logits_v = x @ W_v + b_v
    logits_a = x @ W_a + b_a
    logits_b = x @ W_b + b_b

    eps = 1e-6
    mu = logits_mu
    v = _softplus(logits_v) + eps
    alpha = _softplus(logits_a) + 1.01  # 严格大于1
    beta = _softplus(logits_b) + eps

    return {"mu": mu, "v": v, "alpha": alpha, "beta": beta}


def loss_nll(y: jnp.ndarray, out: dict) -> jnp.ndarray:
    mu, v, alpha, beta = out["mu"], out["v"], out["alpha"], out["beta"]
    y = jnp.asarray(y)

    y = jnp.broadcast_to(y, mu.shape)

    term1 = 0.5 * jnp.log(jnp.pi / v)
    term2 = -alpha * jnp.log(2.0 * beta)
    term3 = (alpha + 0.5) * jnp.log((y - mu) ** 2 * v + 2.0 * beta)
    term4 = gammaln(alpha) - gammaln(alpha + 0.5)

    nll = term1 + term2 + term3 + term4
    return nll.mean()


def loss_total(
    y: jnp.ndarray,
    out: dict,
    *,
    lambda_reg: float = 1e-2,
) -> jnp.ndarray:
    mu, v, alpha, _beta = out["mu"], out["v"], out["alpha"], out["beta"]
    nll = loss_nll(y, out)
    reg = jnp.mean(jnp.abs(jnp.broadcast_to(y, mu.shape) - mu) * (2.0 * v + alpha))
    return nll + lambda_reg * reg


def predict_mean_var(out: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    mu, v, alpha, beta = out["mu"], out["v"], out["alpha"], out["beta"]
    var = beta * (1.0 + v) / (v * (alpha - 1.0))
    return mu, var
