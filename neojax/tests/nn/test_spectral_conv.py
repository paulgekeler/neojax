import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.spectral_conv import SpectralConvNd
from neojax.tests.conftest import (
    create_assert_primitive_in_jaxpr,
    create_jaxpr_assert_equal,
)


class TestSpectralConv:
    def test_dimensions(self):
        key = jr.key(0)
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

    def test_complex_data(self):
        key = jr.key(0)
        in_c, out_c = 2, 4
        x = jr.normal(key=key, shape=(in_c, 32, 32), dtype=jnp.complex64)
        conv_2d = SpectralConvNd(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            modes=(8, 8),
            is_complex_data=True,
        )
        out = conv_2d(x)
        assert out.dtype == jnp.complex64
        assert out.shape == (out_c, 32, 32)

    def test_fft_norms(self):
        key = jr.key(0)
        in_c, out_c = 2, 4
        x = jr.normal(key=key, shape=(in_c, 32, 32), dtype=jnp.float32)
        with pytest.raises((ValueError, TypeCheckError)):
            conv = SpectralConvNd(
                key=key,
                in_channels=in_c,
                out_channels=out_c,
                modes=(8, 8),
                fft_norm="invalid",
            )
        for fft_norm in ["forward", "backward", "ortho", None]:
            conv = SpectralConvNd(
                key=key,
                in_channels=in_c,
                out_channels=out_c,
                modes=(8, 8),
                fft_norm=fft_norm,
            )
            assert conv(x).shape == (out_c, 32, 32)

    def test_hermitian_symmetry(self):
        key = jr.key(0)
        in_c, out_c = 2, 4
        x = jr.normal(key=key, shape=(in_c, 32, 32), dtype=jnp.float32)
        conv_nhs = SpectralConvNd(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            modes=(8, 8),
            enforce_hermitian_symmetry=False,
        )
        conv_hs = SpectralConvNd(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            modes=(8, 8),
            enforce_hermitian_symmetry=True,
        )
        # assert jaxpr are unequal
        with pytest.raises(AssertionError):
            create_jaxpr_assert_equal(conv_nhs, conv_hs, x)

        # assert there is no jnp.real call before irfftn in conv_nhs, but there is in conv_hs
        create_assert_primitive_in_jaxpr(conv_hs, jax.lax.real_p, x)
        with pytest.raises(AssertionError):
            create_assert_primitive_in_jaxpr(conv_nhs, jax.lax.real_p, x)

    def test_resolution_scaling(self):
        key = jr.key(0)
        in_c, out_c = 2, 4

        # Test 1D resolution scaling (up-scaling)
        conv_1d = SpectralConvNd(key, in_c, out_c, (8,), resolution_scaling_factor=1.5)
        assert conv_1d(jnp.ones((in_c, 32))).shape == (out_c, 48)

        # Test 2D resolution scaling (down-scaling)
        conv_2d = SpectralConvNd(
            key, in_c, out_c, (8, 8), resolution_scaling_factor=0.5
        )
        assert conv_2d(jnp.ones((in_c, 32, 32))).shape == (out_c, 16, 16)
