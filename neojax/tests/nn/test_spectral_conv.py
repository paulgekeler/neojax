import jax.numpy as jnp
import jax.random as jr

from neojax.nn.spectral_conv import SpectralConvNd


class TestSpectralConv:
    def test_dimensions(self):
        key = jr.PRNGKey(0)
        in_c, out_c = 2, 4

        # 1D
        conv_1d = SpectralConvNd(key, in_c, out_c, (8,))
        assert conv_1d(jnp.ones((in_c, 32))).shape == (out_c, 32)

        # 2D
        conv_2d = SpectralConvNd(key, in_c, out_c, (8, 8))
        assert conv_2d(jnp.ones((in_c, 32, 32))).shape == (out_c, 32, 32)

        # 3D
        conv_3d = SpectralConvNd(key, in_c, out_c, (4, 4, 4))
        assert conv_3d(jnp.ones((in_c, 8, 8, 8))).shape == (out_c, 8, 8, 8)

        # 4D
        conv_4d = SpectralConvNd(key, in_c, out_c, (4, 4, 4, 4))
        assert conv_4d(jnp.ones((in_c, 8, 8, 8, 8))).shape == (out_c, 8, 8, 8, 8)
