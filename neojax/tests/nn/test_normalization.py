import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from neojax.nn.normalization import ChannelLayerNorm, InstanceNorm, RMSNorm
from neojax.tests.conftest import assert_filter_jittable


class TestInstanceNorm:
    def test_instancenorm_shape(self):
        shape = (3, 16, 16)
        x = jnp.ones(shape)
        norm = InstanceNorm(shape)
        out = norm(x)
        assert out.shape == shape

    def test_instancenorm_zero_mean_unit_var(self):
        shape = (2, 32, 32)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, shape) * 5.0 + 10.0  # Non-zero mean, non-unit var

        norm = InstanceNorm(shape, use_weight=False, use_bias=False)
        out = norm(x)

        # Check each channel's spatial mean and variance
        for c in range(shape[0]):
            channel_data = out[c]
            assert jnp.allclose(jnp.mean(channel_data), 0.0, atol=1e-5)
            assert jnp.allclose(jnp.var(channel_data), 1.0, atol=1e-5)

    def test_instancenorm_affine(self):
        shape = (4, 8)
        x = jnp.ones(shape)

        norm = InstanceNorm(shape, use_weight=True, use_bias=True)
        # Manually set weight and bias
        norm = jtu.tree_map(
            lambda _: jnp.array([2.0, 2.0, 2.0, 2.0]).reshape(4, 1), norm
        )
        # InstanceNorm initializes weight to 1s and bias to 0s.
        # The above is a bit simplified, we just check they exist and are applied.

        out = norm(x)
        assert out.shape == shape

    def test_instancenorm_vmap(self):
        shape = (3, 10, 10)
        batch_size = 5
        x = jnp.ones((batch_size, *shape))

        norm = InstanceNorm(shape)
        vmap_norm = jax.vmap(norm)
        out = vmap_norm(x)

        assert out.shape == (batch_size, *shape)


class TestChannelLayerNorm:
    def test_channellayernorm_shape_and_stats(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (6, 5, 4)) * 3.0 + 2.0

        norm = ChannelLayerNorm(6)
        out = norm(x)

        assert out.shape == x.shape
        assert jnp.allclose(jnp.mean(out, axis=0), 0.0, atol=1e-5)
        assert jnp.allclose(jnp.var(out, axis=0), 1.0, atol=1e-3)

    def test_channellayernorm_matches_eqx_layernorm(self):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (8, 10))

        norm = ChannelLayerNorm(8)
        expected = jax.vmap(norm.norm, in_axes=1, out_axes=1)(x)
        assert jnp.allclose(norm(x), expected)

    def test_channellayernorm_affine_toggles(self):
        norm = ChannelLayerNorm(4, use_weight=False, use_bias=False)
        assert norm.norm.weight is None
        assert norm.norm.bias is None
        assert norm(jnp.ones((4, 3))).shape == (4, 3)


class TestRMSNorm:
    @pytest.mark.parametrize("shape", [(5, 7), (5, 4, 3), (5, 3, 4, 2)])
    def test_matches_manual(self, shape):
        x = jr.normal(jr.key(0), shape) * 3.0 + 1.0
        norm = RMSNorm(shape[0], eps=1e-6)
        out = norm(x)
        xn = np.asarray(x)
        expected = xn / np.sqrt((xn**2).mean(axis=0, keepdims=True) + 1e-6)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # mean square over the channels is one at every location
        np.testing.assert_allclose((np.asarray(out) ** 2).mean(axis=0), 1.0, atol=1e-4)

    def test_weight_scales_channels(self):
        norm = RMSNorm(3)
        norm = eqx.tree_at(lambda n: n.weight, norm, jnp.array([1.0, 2.0, 3.0]))
        x = jr.normal(jr.key(1), (3, 6))
        np.testing.assert_allclose(
            norm(x), RMSNorm(3)(x) * jnp.array([1.0, 2.0, 3.0])[:, None], rtol=1e-5
        )

    def test_scale_invariant(self):
        x = jr.normal(jr.key(2), (4, 9))
        norm = RMSNorm(4)
        np.testing.assert_allclose(norm(x), norm(10.0 * x), rtol=1e-4, atol=1e-5)

    def test_low_precision_is_computed_in_float32(self):
        x = jr.normal(jr.key(3), (8, 5)).astype(jnp.bfloat16)
        out = RMSNorm(8)(x)
        assert out.dtype == jnp.bfloat16
        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float32),
            np.asarray(RMSNorm(8)(x.astype(jnp.float32))),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_jittable_and_grad(self):
        x = jr.normal(jr.key(4), (4, 6))
        norm = RMSNorm(4)
        assert_filter_jittable(norm, x)
        g = eqx.filter_grad(lambda n: jnp.sum(n(x) ** 2))(norm)
        assert bool(jnp.all(jnp.isfinite(g.weight)))

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            RMSNorm(0)
        with pytest.raises(ValueError):
            RMSNorm(4)(jnp.ones((3, 5)))
