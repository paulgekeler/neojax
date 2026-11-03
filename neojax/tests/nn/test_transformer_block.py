import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.transformer_block import SwiGLU, TransformerBlock
from neojax.tests.conftest import assert_filter_jittable

C, T = 16, 9


def silu(x):
    return x / (1.0 + np.exp(-x))


class TestSwiGLU:
    def test_matches_manual(self):
        ffn = SwiGLU(jr.key(0), C, 24)
        x = jr.normal(jr.key(1), (C, T))
        w1, w3, w2 = (np.asarray(m.weights[0]) for m in (ffn.gate, ffn.up, ffn.down))
        expected = w2 @ (silu(w1 @ np.asarray(x)) * (w3 @ np.asarray(x)))
        np.testing.assert_allclose(ffn(x), expected, rtol=1e-4, atol=1e-5)
        # no biases
        assert all(b is None for m in (ffn.gate, ffn.up, ffn.down) for b in m.biases)

    def test_trailing_axes_and_jit(self):
        ffn = SwiGLU(jr.key(0), 4, 8)
        x = jr.normal(jr.key(1), (4, 3, 5))
        assert ffn(x).shape == (4, 3, 5)
        assert_filter_jittable(ffn, x)

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            SwiGLU(jr.key(0), 0, 4)
        with pytest.raises(ValueError):
            SwiGLU(jr.key(0), 4, 0)


class TestTransformerBlock:
    @pytest.mark.parametrize("use_attn_norm", [True, False])
    @pytest.mark.parametrize("use_ffn_norm", [True, False])
    def test_matches_reference_structure(self, use_attn_norm, use_ffn_norm):
        block = TransformerBlock(
            jr.key(0),
            C,
            n_heads=4,
            use_attn_norm=use_attn_norm,
            use_ffn_norm=use_ffn_norm,
        )
        x = jr.normal(jr.key(1), (C, T))

        h = x if not use_attn_norm else block.attn_norm(x)
        h = x + block.attn(h)  # residual of the un-normalized x
        h = h if not use_ffn_norm else block.ffn_norm(h)
        expected = h + block.ffn(h)  # residual taken after the norm
        np.testing.assert_allclose(block(x), expected, rtol=1e-5, atol=1e-6)
        assert (block.attn_norm is None) == (not use_attn_norm)
        assert (block.ffn_norm is None) == (not use_ffn_norm)
        assert_filter_jittable(block, x)

    def test_ffn_hidden_channels(self):
        block = TransformerBlock(jr.key(0), C, n_heads=4, ffn_multiplier=2)
        assert block.ffn.gate.weights[0].shape == (2 * C, C)
        assert block.ffn.down.weights[0].shape == (C, 2 * C)

    def test_skip_connection(self):
        block = TransformerBlock(jr.key(0), C, n_heads=4, skip_connection=True)
        plain = TransformerBlock(jr.key(0), C, n_heads=4)
        x = jr.normal(jr.key(1), (C, T))
        skip = jr.normal(jr.key(2), (C, T))

        assert block.skip_proj.weights[0].shape == (C, 2 * C)
        assert plain.skip_proj is None

        merged = block.skip_proj(jnp.concatenate([x, skip], axis=0))
        merged_block = eqx.tree_at(lambda b: b.skip_proj, block, None)
        np.testing.assert_allclose(
            block(x, skip), merged_block(merged), rtol=1e-5, atol=1e-6
        )
        # without a skip input the block is a plain block
        np.testing.assert_allclose(block(x), merged_block(x), rtol=1e-5, atol=1e-6)
        # a block without skip connection ignores the skip input
        np.testing.assert_allclose(plain(x, skip), plain(x), atol=1e-6)

    def test_dropout_vmap_and_grad(self):
        block = TransformerBlock(jr.key(0), C, n_heads=4, dropout=0.3, use_rope=True)
        xs = jr.normal(jr.key(1), (3, C, T))
        keys = jr.split(jr.key(2), 3)
        out = jax.vmap(lambda x, k: block(x, key=k))(xs, keys)
        assert out.shape == (3, C, T)
        with pytest.raises(ValueError):
            block(xs[0])
        np.testing.assert_allclose(
            block(xs[0], inference=True),
            block(xs[0], key=keys[0], inference=True),
            atol=1e-6,
        )
        grads = eqx.filter_grad(lambda m: jnp.sum(m(xs[0], key=keys[0]) ** 2))(block)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert bool(jnp.all(jnp.isfinite(leaf)))

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            TransformerBlock(jr.key(0), C, n_heads=4, ffn_multiplier=0)
        with pytest.raises(ValueError):
            TransformerBlock(jr.key(0), C, n_heads=3)
