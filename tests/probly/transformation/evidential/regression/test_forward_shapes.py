import pytest
import jax
import jax.numpy as jnp

# TODO: 按仓库实际 API 修改下面这些 import
# from probly.transformation.evidential.regression import to_evidential_regressor
# from probly.models import TinyMLP  # 或你们用来做最小基模的东西

def _unpack(out):
    """兼容 dict 或具名属性的输出结构"""
    if isinstance(out, dict):
        mu = out["mu"]; v = out["v"]; alpha = out["alpha"]; beta = out["beta"]
    else:
        mu = getattr(out, "mu"); v = getattr(out, "v")
        alpha = getattr(out, "alpha"); beta = getattr(out, "beta")
    return mu, v, alpha, beta

@pytest.mark.parametrize("batch, in_dim, out_dim", [(1, 3, 1), (5, 4, 2)])
def test_forward_shapes(batch, in_dim, out_dim):
    # 先跳过，等你把 evidential 构造器接上再把下面两行删掉
    pytest.skip("TODO: 先搭好骨架；接上实际 evidential 模型后删除此 skip")

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, in_dim))

    # 参考 dropout 的测试怎么构造 base model，然后包一层 evidential
    # base = TinyMLP(in_dim=in_dim, out_dim=out_dim, hidden_sizes=[8])
    # evid = to_evidential_regressor(base)

    params = evid.init(key, x)             # 按你们框架的 init 签名
    out = evid.apply(params, x, train=False)
    mu, v, alpha, beta = _unpack(out)

    assert mu.shape == (batch, out_dim)
    assert v.shape == (batch, out_dim)
    assert alpha.shape == (batch, out_dim)
    assert beta.shape == (batch, out_dim)




