import jax.numpy as jnp
import jax.random as jr

from neojax.nn.geo_spectral_conv import GeoSpectralConvNd
from neojax.tests.conftest import assert_filter_jittable


class TestGeoSpectralConv:
    def test_geo_spectral_conv_fallback(self):
        key = jr.key(0)
        # 2D Grid Fallback Mode (x_in is None)
        conv = GeoSpectralConvNd(
            key=key,
            in_channels=2,
            out_channels=3,
            modes=(4, 4),
        )
        u = jr.normal(key, (2, 16, 16))
        out = conv(u)
        # Verify fallback outputs correct shape: (out_channels, grid...)
        assert out.shape == (3, 16, 16)
        assert_filter_jittable(conv, u)

    def test_geo_spectral_conv_continuous(self):
        key = jr.key(0)
        # 2D Continuous DFT Mode
        conv = GeoSpectralConvNd(
            key=key,
            in_channels=2,
            out_channels=3,
            modes=(4, 4),
        )
        # 20 input coordinates, 15 output coordinates
        x_in = jr.uniform(key, (20, 2))
        x_out = jr.uniform(key, (15, 2))
        u = jr.normal(key, (2, 20))

        out = conv(u, x_in=x_in, x_out=x_out)
        assert out.shape == (3, 15)
        assert_filter_jittable(conv, u, x_in=x_in, x_out=x_out)

        # Default x_out = x_in
        out_self = conv(u, x_in=x_in)
        assert out_self.shape == (3, 20)
        assert_filter_jittable(conv, u, x_in=x_in)

    def test_geo_spectral_conv_complex(self):
        key = jr.key(0)
        conv = GeoSpectralConvNd(
            key=key,
            in_channels=1,
            out_channels=1,
            modes=(4, 4),
            is_complex_data=True,
        )
        x_in = jr.uniform(key, (10, 2))
        u = jr.normal(key, (1, 10)) + 1j * jr.normal(key, (1, 10))
        out = conv(u, x_in=x_in)
        assert out.shape == (1, 10)
        assert jnp.iscomplexobj(out)
        assert_filter_jittable(conv, u, x_in=x_in)
