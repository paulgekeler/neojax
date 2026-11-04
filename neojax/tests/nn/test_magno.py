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
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.magno import MAGNOBlock
from neojax.nn.neighbor_search import radius_neighbor_search
from neojax.tests.conftest import assert_filter_jittable

C, N, M, RADIUS = 4, 70, 25, 0.3


@pytest.fixture
def make_data():
    def f(d=2):
        k1, k2, k3 = jr.split(jr.key(0), 3)
        y = jr.uniform(k1, (N, d), minval=-1.0, maxval=1.0)
        x = jr.uniform(k2, (M, d), minval=-1.0, maxval=1.0)
        return y, x, jr.normal(k3, (C, N))

    return f


@pytest.fixture
def make_block():
    def f(d=2, **kwargs):
        defaults = dict(
            key=jr.key(1),
            channels=C,
            coord_dim=d,
            radius=RADIUS,
            max_neighbors=20,
            hidden_channels=8,
            n_layers=2,
            attention_dim=8,
        )
        return MAGNOBlock(**(defaults | kwargs))

    return f


@pytest.mark.usefixtures("make_data", "make_block")
class TestMAGNOBlock:
    @pytest.mark.parametrize("d", [2, 3])
    @pytest.mark.parametrize("scales", [(1.0,), (0.5, 1.0, 2.0)])
    @pytest.mark.parametrize("use_geoembed", [True, False])
    def test_shapes_and_jit(self, d, scales, use_geoembed, make_data, make_block):
        y, x, f_y = make_data(d)
        block = make_block(d, scales=scales, use_geoembed=use_geoembed)
        out = block(y, x, f_y)
        assert out.shape == (C, M)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert_filter_jittable(block, y, x, f_y)

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    @pytest.mark.parametrize("embedding_method", ["statistical", "pointnet"])
    @pytest.mark.parametrize("node_embedding", [True, False])
    def test_variants_run(
        self, attention_type, embedding_method, node_embedding, make_data, make_block
    ):
        y, x, f_y = make_data()
        block = make_block(
            attention_type=attention_type,
            embedding_method=embedding_method,
            node_embedding=node_embedding,
            node_embedding_frequencies=2,
            use_scale_weights=True,
            scales=(1.0, 2.0),
        )
        out = block(y, x, f_y)
        assert out.shape == (C, M) and bool(jnp.all(jnp.isfinite(out)))

    def test_attention_flag(self, make_block):
        assert make_block().gno.integral_transform.attention is not None
        assert make_block(use_attention=False).gno.integral_transform.attention is None
        assert (
            make_block(use_attention=False).gno.integral_transform.reduction == "mean"
        )

    def test_single_scale_without_geoembed_is_the_gno(self, make_data, make_block):
        y, x, f_y = make_data()
        block = make_block(use_geoembed=False)
        neighbors = radius_neighbor_search(y, x, RADIUS, 20)
        np.testing.assert_allclose(
            block(y, x, f_y),
            block.gno(y, x, f_y, neighbors=neighbors),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_scales_are_averaged_with_shared_parameters(self, make_data, make_block):
        y, x, f_y = make_data()
        scales = (0.5, 1.0, 2.0)
        block = make_block(use_geoembed=False, scales=scales)
        expected = jnp.mean(
            jnp.stack(
                [
                    block.gno(
                        y,
                        x,
                        f_y,
                        neighbors=radius_neighbor_search(y, x, RADIUS * s, 20),
                    )
                    for s in scales
                ]
            ),
            axis=0,
        )
        np.testing.assert_allclose(block(y, x, f_y), expected, rtol=1e-5, atol=1e-6)

    def test_scale_weights(self, make_data, make_block):
        y, x, f_y = make_data()
        scales = (1.0, 2.0)
        block = make_block(use_geoembed=False, scales=scales, use_scale_weights=True)
        assert block.scale_weighting is not None
        per_scale = jnp.stack(
            [
                block.gno(
                    y, x, f_y, neighbors=radius_neighbor_search(y, x, RADIUS * s, 20)
                )
                for s in scales
            ]
        )
        weights = jax.nn.softmax(block.scale_weighting(x.T), axis=0)
        np.testing.assert_allclose(weights.sum(0), 1.0, atol=1e-6)
        expected = jnp.sum(weights[:, None, :] * per_scale, axis=0)
        np.testing.assert_allclose(block(y, x, f_y), expected, rtol=1e-5, atol=1e-6)

    def test_geometric_embedding_is_merged_by_recovery(self, make_data, make_block):
        y, x, f_y = make_data()
        block = make_block()
        neighbors = radius_neighbor_search(y, x, RADIUS, 20)
        gno_out = block.gno(y, x, f_y, neighbors=neighbors)
        geometry = block.geoembed(y, x, neighbors)
        expected = block.recovery(jnp.concatenate([gno_out, geometry], axis=0))
        np.testing.assert_allclose(block(y, x, f_y), expected, rtol=1e-5, atol=1e-6)

    def test_precomputed_neighbors(self, make_data, make_block):
        y, x, f_y = make_data()
        block = make_block(scales=(1.0, 2.0))
        neighbors = block.build_neighbors(y, x)
        assert len(neighbors) == 2
        np.testing.assert_allclose(
            block(y, x, f_y), block(y, x, f_y, neighbors=neighbors), atol=1e-6
        )
        with pytest.raises(ValueError, match="one per scale"):
            block(y, x, f_y, neighbors=neighbors[:1])

    def test_chunked_equals_unchunked(self, make_data, make_block):
        y, x, f_y = make_data()
        full = make_block(scales=(1.0, 2.0))
        chunked = make_block(scales=(1.0, 2.0), query_chunk_size=7)
        np.testing.assert_allclose(
            full(y, x, f_y), chunked(y, x, f_y), rtol=1e-4, atol=1e-5
        )

    @pytest.mark.parametrize("strategy", ["ratio", "max_neighbors"])
    def test_edge_drop(self, strategy, make_data, make_block):
        y, x, f_y = make_data()
        kwargs = (
            {"edge_drop_ratio": 0.4}
            if strategy == "ratio"
            else {"edge_drop_max_neighbors": 2}
        )
        block = make_block(edge_drop_strategy=strategy, **kwargs)

        plain = block(y, x, f_y, inference=True)
        dropped = block(y, x, f_y, key=jr.key(3))
        assert not np.allclose(plain, dropped)
        # deterministic given the key, different for other keys
        np.testing.assert_array_equal(dropped, block(y, x, f_y, key=jr.key(3)))
        assert not np.allclose(dropped, block(y, x, f_y, key=jr.key(4)))
        # inference disables it, with or without key
        np.testing.assert_array_equal(
            plain, block(y, x, f_y, key=jr.key(3), inference=True)
        )

        with pytest.raises(ValueError, match="PRNG key"):
            block(y, x, f_y)

    def test_edge_drop_ratio_one_is_identity(self, make_data, make_block):
        y, x, f_y = make_data()
        block = make_block(edge_drop_strategy="ratio", edge_drop_ratio=1.0)
        np.testing.assert_allclose(
            block(y, x, f_y, key=jr.key(0)),
            block(y, x, f_y, inference=True),
            atol=1e-6,
        )

    def test_vmap_and_grad(self, make_data, make_block):
        y, x, f_y = make_data()
        block = make_block(edge_drop_strategy="ratio", edge_drop_ratio=0.8)
        f_batch = jr.normal(jr.key(5), (3, C, N))
        keys = jr.split(jr.key(6), 3)
        neighbors = block.build_neighbors(y, x)
        out = jax.vmap(lambda f, k: block(y, x, f, neighbors=neighbors, key=k))(
            f_batch, keys
        )
        assert out.shape == (3, C, M)

        def loss(model):
            return jnp.sum(model(y, x, f_y, key=jr.key(1)) ** 2)

        grads = eqx.filter_grad(loss)(block)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert bool(jnp.all(jnp.isfinite(leaf)))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"channels": 0},
            {"coord_dim": 0},
            {"max_neighbors": 0},
            {"radius": 0.0},
            {"scales": ()},
            {"scales": (1.0, -1.0)},
            {"edge_drop_strategy": "ratio"},
            {"edge_drop_strategy": "ratio", "edge_drop_ratio": 0.0},
            {"edge_drop_strategy": "max_neighbors"},
            {"edge_drop_strategy": "invalid"},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, make_block):
        with pytest.raises((ValueError, TypeCheckError)):
            make_block(**kwargs)
