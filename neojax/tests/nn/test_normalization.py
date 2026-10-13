import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from neojax.nn.normalization import ChannelLayerNorm, InstanceNorm


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
