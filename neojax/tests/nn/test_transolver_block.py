import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.transolver_block import TransolverBlock
from neojax.tests.conftest import assert_filter_jittable


class TestTransolverBlock:
    @pytest.mark.parametrize(
        "ndim,shape", [(None, (16, 40)), (2, (16, 6, 7)), (3, (16, 4, 3, 3))]
    )
    def test_dimensions(self, ndim, shape):
        block = TransolverBlock(
            jr.key(0), hidden_channels=16, n_heads=2, slice_num=4, ndim=ndim
        )
        x = jr.normal(jr.key(1), shape)
        assert block(x).shape == shape
        assert_filter_jittable(block, x)

    def test_residual_structure(self):
        block = TransolverBlock(jr.key(0), hidden_channels=8, n_heads=2, slice_num=4)
        x = jr.normal(jr.key(1), (8, 12))
        h = x + block.attn(block.norm_1(x))
        expected = h + block.mlp(block.norm_2(h))
        np.testing.assert_allclose(block(x), expected, rtol=1e-6)

    def test_mlp_ratio(self):
        block = TransolverBlock(jr.key(0), hidden_channels=8, n_heads=2, mlp_ratio=2.5)
        assert block.mlp.weights[0].shape == (20, 8)
        assert block.mlp.weights[1].shape == (8, 20)

    def test_dropout_requires_key(self):
        block = TransolverBlock(
            jr.key(0), hidden_channels=8, n_heads=2, slice_num=4, dropout=0.3
        )
        x = jr.normal(jr.key(1), (8, 12))
        assert block(x, inference=True).shape == x.shape
        assert block(x, key=jr.key(2)).shape == x.shape
        with pytest.raises(ValueError):
            block(x)

    def test_grad_finite(self):
        block = TransolverBlock(jr.key(0), hidden_channels=8, n_heads=2, slice_num=4)
        x = jr.normal(jr.key(1), (8, 12))
        grads = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(block)
        for leaf in jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)):
            assert jnp.all(jnp.isfinite(leaf))

    def test_raises(self):
        with pytest.raises(ValueError):
            TransolverBlock(jr.key(0), hidden_channels=8, mlp_ratio=0.0)
        with pytest.raises(ValueError):
            TransolverBlock(jr.key(0), hidden_channels=10, n_heads=4)
