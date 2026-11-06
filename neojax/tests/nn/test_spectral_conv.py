# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
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

        # assert there is no conjugation before irfftn in conv_nhs, but there is in conv_hs
        create_assert_primitive_in_jaxpr(conv_hs, jax.lax.conj_p, x)
        with pytest.raises(AssertionError):
            create_assert_primitive_in_jaxpr(conv_nhs, jax.lax.conj_p, x)

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


def _band_limited_field(key, ndim, n_points, modes):
    """Real field of shape `(1, n, ..., n)` with content in the retained modes only."""
    axes = tuple(range(1, ndim + 1))
    ft = jnp.fft.rfftn(jr.normal(key, (1,) + (n_points,) * ndim), axes=axes)
    freqs = [jnp.abs(jnp.fft.fftfreq(n_points, 1 / n_points))] * (ndim - 1)
    freqs.append(jnp.fft.rfftfreq(n_points, 1 / n_points))
    grids = jnp.meshgrid(*freqs, indexing="ij")
    mask = jnp.all(jnp.stack(grids) < modes, axis=0)
    return jnp.fft.irfftn(ft * mask, s=(n_points,) * ndim, axes=axes)


def _identity_conv(ndim, modes):
    """Spectral convolution with all weights equal to one."""
    conv = SpectralConvNd(jr.key(1), 1, 1, modes=(modes,) * ndim)
    weights = tuple(
        conv.weights.to_real(jnp.ones(w.shape[:-1], dtype=jnp.complex64))
        for w in conv.weights.weights
    )
    return eqx.tree_at(lambda m: m.weights.weights, conv, weights)


class TestHermitianSymmetry:
    """See issue #62 for why these tests were added."""

    dummy_spectral_conv = _identity_conv(1, 4)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_unit_weights_are_identity(self, ndim):
        x = _band_limited_field(jr.key(0), ndim, n_points=16, modes=4)
        assert jnp.allclose(_identity_conv(ndim, modes=4)(x), x, atol=1e-5)

    def test_keeps_odd_profile_along_first_axis(self):
        # The spectrum of sin(x) lies in the k_y = 0 plane and is purely imaginary.
        n_points = 16
        grid = 2 * jnp.pi * jnp.arange(n_points) / n_points
        x = jnp.broadcast_to(jnp.sin(grid)[None, :, None], (1, n_points, n_points))
        assert jnp.allclose(_identity_conv(2, modes=4)(x), x, atol=1e-5)

    def test_projection_2d(self):
        key_re, key_im = jr.split(jr.key(0))
        plane = jr.normal(key_re, (3, 8)) + 1j * jr.normal(key_im, (3, 8))
        part = self.dummy_spectral_conv._hermitian_part(plane)
        mirrored = jnp.roll(jnp.flip(part, axis=1), 1, axis=1)
        assert jnp.allclose(part, jnp.conj(mirrored))
        # the projection is idempotent and only self-conjugate frequencies become real
        assert jnp.allclose(self.dummy_spectral_conv._hermitian_part(part), part)
        assert jnp.allclose(part[:, jnp.array([0, 4])].imag, 0.0)
        assert not jnp.allclose(part[:, 1].imag, 0.0)

    def test_projection_keeps_plane_of_real_signal(self):
        x = jr.normal(jr.key(0), (2, 8, 8))
        plane = jnp.fft.rfftn(x, axes=(1, 2))[..., 0]
        assert jnp.allclose(
            self.dummy_spectral_conv._hermitian_part(plane), plane, atol=1e-5
        )

    def test_projection_1d_is_real_part(self):
        plane = jnp.array([1.0 + 2.0j, 3.0 - 4.0j])
        assert jnp.allclose(self.dummy_spectral_conv._hermitian_part(plane), plane.real)


FACTORIZATIONS = [
    {},
    {"factorization": "tucker", "ranks": 8},
    {"factorization": "cp", "ranks": 16},
    {"factorization": "tt", "ranks": 8},
    {"factorization": "tucker", "ranks": 8, "implementation": "reconstructed"},
    {"factorization": "cp", "ranks": 16, "implementation": "reconstructed"},
]


class TestSpectralWeights:
    """See issue #61 for why these tests were added."""

    @pytest.mark.parametrize("kwargs", FACTORIZATIONS)
    def test_learnable_weights_are_real(self, kwargs):
        conv = SpectralConvNd(jr.key(0), 1, 1, modes=(4, 4), **kwargs)
        leaves = jax.tree.leaves(eqx.filter(conv, eqx.is_inexact_array))
        assert leaves
        assert not any(jnp.iscomplexobj(leaf) for leaf in leaves)

    @pytest.mark.parametrize("kwargs", FACTORIZATIONS)
    def test_plain_optax_learns_translation(self, kwargs):
        # A translation is a diagonal Fourier multiplier of rank one, so all weights can
        # represent it. Complex weights with plain Optax updates would diverge here.
        keys = jr.split(jr.key(0), 16)
        x = jax.vmap(lambda k: _band_limited_field(k, 2, n_points=32, modes=7))(keys)
        y = jnp.roll(x, (3, 5), axis=(-2, -1))
        conv = SpectralConvNd(jr.key(1), 1, 1, modes=(8, 8), **kwargs)
        optimizer = optax.adam(1e-2)
        state = optimizer.init(eqx.filter(conv, eqx.is_inexact_array))

        def loss_fn(model):
            return jnp.linalg.norm(jax.vmap(model)(x) - y) / jnp.linalg.norm(y)

        @eqx.filter_jit
        def step(model, state):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, state = optimizer.update(grads, state)
            return eqx.apply_updates(model, updates), state, loss

        for _ in range(400):
            conv, state, loss = step(conv, state)
        assert loss < 0.05

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_half_precision_weights_forward(self, dtype):
        conv = SpectralConvNd(jr.key(0), 1, 1, modes=(4, 4))
        half = jax.tree.map(
            lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, conv
        )
        x = jr.normal(jr.key(1), (1, 16, 16))
        assert jnp.allclose(half(x), conv(x), atol=1e-1)
