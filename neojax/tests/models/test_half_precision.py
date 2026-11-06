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
import pytest

from neojax.models import CNO, FNO, TFNO, UNO, GeoFNO, MgNO
from neojax.nn.geo_spectral_conv import GeoSpectralConvNd
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.nn.resample import Resampler
from neojax.nn.spectral_conv import SpectralConvNd
from neojax.tests.conftest import assert_reduced_precision_close_to_full

KEY = jr.key(0)
FNO_KWARGS = dict(key=KEY, in_channels=1, out_channels=1, hidden_channels=8, n_layers=2)
HALF_DTYPES = [jnp.float16, jnp.bfloat16]

MODELS = {
    "fno_1d": (
        lambda: FNO(
            modes=(4,), positional_embedding="grid", domain_padding=0.1, **FNO_KWARGS
        ),
        (1, 32),
    ),
    "fno_2d": (lambda: FNO(modes=(4, 4), **FNO_KWARGS), (1, 32, 32)),
    "fno_2d_grid": (
        lambda: FNO(modes=(4, 4), positional_embedding="grid", **FNO_KWARGS),
        (1, 32, 32),
    ),
    "fno_3d": (
        lambda: FNO(modes=(4, 4, 4), positional_embedding="grid", **FNO_KWARGS),
        (1, 16, 16, 16),
    ),
    "tfno": (lambda: TFNO(modes=(4, 4), ranks=4, **FNO_KWARGS), (1, 32, 32)),
    "uno_1d": (
        lambda: UNO(
            key=KEY,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 8],
            uno_modes=[[4], [4]],
            uno_scalings=[[0.5], [2]],
            n_fno_layers=2,
            positional_embedding="grid",
        ),
        (1, 32),
    ),
    "uno_2d": (
        lambda: UNO(
            key=KEY,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 8],
            uno_modes=[[4, 4], [4, 4]],
            uno_scalings=[[0.5, 0.5], [2, 2]],
            n_fno_layers=2,
            positional_embedding="grid",
        ),
        (1, 32, 32),
    ),
    "uno_3d": (
        lambda: UNO(
            key=KEY,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 8],
            uno_modes=[[4, 4, 4], [4, 4, 4]],
            uno_scalings=[[0.5, 0.5, 0.5], [2, 2, 2]],
            n_fno_layers=2,
            positional_embedding="grid",
        ),
        (1, 16, 16, 16),
    ),
    "cno": (
        lambda: CNO(
            key=KEY,
            in_channels=1,
            out_channels=1,
            ndim=2,
            n_layers=1,
            n_res=1,
            n_res_neck=1,
            channel_multiplier=8,
            latent_lift_proj_channels=8,
            positional_embedding="grid",
            batch_norm=False,
            pad_mode="wrap",
        ),
        (1, 32, 32),
    ),
    "mgno": (lambda: MgNO.make_mgno_type(KEY, "MgNO_NS", n_layers=2), (1, 64, 64)),
    "geo_fno_grid": (
        lambda: GeoFNO(
            key=KEY,
            in_channels=2,
            out_channels=1,
            hidden_channels=8,
            n_layers=2,
            modes=(4, 4),
            grid_resolution=(16, 16),
            use_coord_projection=True,
        ),
        (2, 16, 16),
    ),
}


@pytest.mark.parametrize("dtype", HALF_DTYPES)
@pytest.mark.parametrize("name", MODELS)
class TestModels:
    def test_forward_matches_full_precision(self, name, dtype):
        make_model, shape = MODELS[name]
        assert_reduced_precision_close_to_full(
            lambda model, x: model(x),
            filter_kwarg_names=(),
            reduced_dtype=dtype,
            rtol_factor=20.0,
            model=make_model(),
            x=jr.normal(jr.key(1), shape),
        )

    def test_astype_output_dtype_and_grads(self, name, dtype):
        make_model, shape = MODELS[name]
        model = make_model().astype(dtype)
        x = jr.normal(jr.key(1), shape).astype(dtype)
        assert eqx.filter_jit(model)(x).dtype == dtype

        def loss_fn(m):
            return jnp.sum(m(x).astype(jnp.float32) ** 2)

        grads = eqx.filter_grad(loss_fn)(model)
        for leaf in jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array)):
            assert leaf.dtype == dtype
            assert jnp.isfinite(leaf.astype(jnp.float32)).all()


@pytest.mark.parametrize("dtype", HALF_DTYPES)
class TestComponents:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"factorization": "tucker", "ranks": 4},
            {"resolution_scaling_factor": 2},
        ],
    )
    def test_spectral_conv(self, dtype, kwargs):
        conv = SpectralConvNd(KEY, 2, 3, modes=(4, 4), **kwargs)
        half = jax.tree.map(
            lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, conv
        )
        assert half(jr.normal(KEY, (2, 16, 16)).astype(dtype)).dtype == dtype

    def test_geo_spectral_conv_with_coordinates(self, dtype):
        conv = GeoSpectralConvNd(KEY, 2, 3, modes=(4, 4))
        half = jax.tree.map(
            lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, conv
        )
        u, x_in = jr.normal(KEY, (2, 20)), jr.uniform(KEY, (20, 2))
        out = half(u.astype(dtype), x_in=x_in.astype(dtype))
        assert out.dtype == dtype
        assert out.shape == (3, 20)

    def test_grid_embedding(self, dtype):
        embedding = GridEmbeddingNd(in_channels=1, ndim=2)
        assert embedding(jnp.ones((1, 8, 8), dtype=dtype)).dtype == dtype

    @pytest.mark.parametrize("shape", [(2, 16), (2, 16, 16), (2, 16, 16, 16)])
    def test_resampler(self, dtype, shape):
        resampler = Resampler(shape, 0.5, axes=tuple(range(1, len(shape))))
        out = resampler(jr.normal(KEY, shape).astype(dtype))
        assert out.dtype == dtype
        assert out.shape == (2, *[s // 2 for s in shape[1:]])


def test_spectral_conv_keeps_double_precision():
    with jax.enable_x64(True):
        conv = SpectralConvNd(KEY, 1, 1, modes=(4, 4))
        double = jax.tree.map(
            lambda a: a.astype(jnp.float64) if eqx.is_inexact_array(a) else a, conv
        )
        x = jr.normal(KEY, (1, 16, 16), dtype=jnp.float64)
        assert double(x).dtype == jnp.float64
