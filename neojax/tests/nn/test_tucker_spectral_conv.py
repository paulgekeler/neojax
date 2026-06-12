import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.nn.tucker_spectral_conv import TuckerSpectralConvNd


class TestTuckerSpectralConv:
    def test_dimensions(self):
        key = jr.key(0)
        in_c, out_c = 2, 4

        # 1D
        tconv_1d = TuckerSpectralConvNd(
            key,
            in_c,
            out_c,
            (8,),
            2,
        )
        assert tconv_1d(jnp.ones((in_c, 32))).shape == (out_c, 32)

        # 2D
        tconv_2d = TuckerSpectralConvNd(key, in_c, out_c, (8, 8), 3)
        assert tconv_2d(jnp.ones((in_c, 32, 32))).shape == (out_c, 32, 32)

        # 3D
        tconv_3d = TuckerSpectralConvNd(key, in_c, out_c, (4, 4, 4), 2)
        assert tconv_3d(jnp.ones((in_c, 8, 8, 8))).shape == (out_c, 8, 8, 8)

        # 4D
        tconv_4d = TuckerSpectralConvNd(key, in_c, out_c, (4, 4, 4, 4), 2)
        assert tconv_4d(jnp.ones((in_c, 8, 8, 8, 8))).shape == (out_c, 8, 8, 8, 8)

    def test_ranks_type(self):
        key = jr.key(0)
        in_c, out_c = 2, 4

        tconv_2d = TuckerSpectralConvNd(key, in_c, out_c, (8, 8), (4, 4, 4, 4))
        assert tconv_2d(jnp.ones((in_c, 32, 32))).shape == (out_c, 32, 32)

        with pytest.raises(ValueError):
            TuckerSpectralConvNd(key, in_c, out_c, (8, 8), (4, 4, 4))

    def test_unshared_factor_matrices(self):
        key = jr.key(0)
        in_c, out_c = 2, 4

        tconv_3d = TuckerSpectralConvNd(
            key, in_c, out_c, (8, 8), 4, share_factor_matrices=False
        )
        assert tconv_3d(jnp.ones((in_c, 32, 32))).shape == (out_c, 32, 32)
