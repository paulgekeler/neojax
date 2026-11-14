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

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.neighbor_attention import NeighborAttention
from neojax.nn.neighbor_search import Neighbors
from neojax.tests.conftest import (
    assert_filter_jittable,
    assert_reduced_precision_close_to_full,
)

E, C = 3, 5
COUNTS = (6, 4, 1, 0, 3)
NUM_EDGES = sum(COUNTS)
CAPACITY = NUM_EDGES + 2  # two padding edges


@pytest.fixture
def inputs():
    """Edges of 5 queries with 6, 4, 1, 0 and 3 neighbors, plus padding."""
    k1, k2 = jr.split(jr.key(0))
    query_index = np.repeat(np.arange(C), COUNTS)
    query_index = np.concatenate([query_index, [C, C]])
    neighbors = Neighbors(
        query_index=jnp.asarray(query_index, jnp.int32),
        data_index=jnp.zeros(CAPACITY, jnp.int32),
        overflowed=jnp.array(False),
        num_queries=C,
    )
    x_query = jr.normal(k1, (E, C))
    x = x_query[:, np.minimum(query_index, C - 1)]
    y = jr.normal(k2, (E, CAPACITY))
    return x, y, neighbors


def edges_of(neighbors, i):
    return np.nonzero(np.asarray(neighbors.query_index) == i)[0]


@pytest.mark.usefixtures("inputs")
class TestNeighborAttention:
    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_weights_normalized_over_neighbors(self, inputs, attention_type):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(1), E, attention_type, attention_dim=8)
        w = np.asarray(att(x, y, neighbors))
        assert w.shape == (CAPACITY,)
        np.testing.assert_array_equal(w[NUM_EDGES:], 0.0)
        sums = [w[edges_of(neighbors, i)].sum() for i in range(C)]
        # queries with neighbors sum to one, the empty one has no edges
        np.testing.assert_allclose(sums, [1, 1, 1, 0, 1], atol=1e-6)
        assert (w >= 0).all()

    def test_cosine_values(self, inputs):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(1), E, "cosine")
        w = np.asarray(att(x, y, neighbors))
        xn, yn = np.asarray(x), np.asarray(y)
        for i in range(C):
            edges = edges_of(neighbors, i)
            if edges.size == 0:
                continue
            logits = np.array(
                [
                    xn[:, j]
                    @ yn[:, j]
                    / np.linalg.norm(xn[:, j])
                    / np.linalg.norm(yn[:, j])
                    for j in edges
                ]
            )
            expected = np.exp(logits - logits.max())
            np.testing.assert_allclose(w[edges], expected / expected.sum(), atol=1e-5)

    def test_dot_product_values(self, inputs):
        x, y, neighbors = inputs
        dim = 8
        att = NeighborAttention(jr.key(1), E, "dot_product", attention_dim=dim)
        w = np.asarray(att(x, y, neighbors))
        wq, bq = (
            np.asarray(att.query_proj.weights[0]),
            np.asarray(att.query_proj.biases[0]),
        )
        wk, bk = np.asarray(att.key_proj.weights[0]), np.asarray(att.key_proj.biases[0])
        xn, yn = np.asarray(x), np.asarray(y)
        for i in range(C):
            edges = edges_of(neighbors, i)
            if edges.size == 0:
                continue
            logits = np.array(
                [
                    ((wq @ xn[:, j] + bq) @ (wk @ yn[:, j] + bk)) / np.sqrt(dim)
                    for j in edges
                ]
            )
            expected = np.exp(logits - logits.max())
            np.testing.assert_allclose(w[edges], expected / expected.sum(), atol=1e-5)

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_padding_values_are_ignored(self, inputs, attention_type):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(1), E, attention_type)
        w = att(x, y, neighbors)
        # change coordinates of the padding edges
        garbage = jr.normal(jr.key(7), y.shape) * 100.0
        padding = ~neighbors.mask[None]
        w_garbage = att(
            jnp.where(padding, garbage, x), jnp.where(padding, garbage, y), neighbors
        )
        np.testing.assert_allclose(w, w_garbage, atol=1e-6)

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_gradients_are_finite(self, inputs, attention_type):
        x, y, neighbors = inputs
        # points at the origin
        x = x.at[:, :6].set(0.0)
        y = y.at[:, 6].set(0.0)
        att = NeighborAttention(jr.key(1), E, attention_type)

        def loss(att, x, y):
            return jnp.sum(att(x, y, neighbors) * jnp.arange(CAPACITY))

        grads = jax.grad(loss, argnums=(0, 1, 2))(att, x, y)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert np.isfinite(np.asarray(leaf)).all()

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_jittable(self, inputs, attention_type):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(1), E, attention_type)
        assert_filter_jittable(att, x, y, neighbors)

    def test_channel_mismatch_raises(self, inputs):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(1), E + 1)
        with pytest.raises(ValueError, match="channels"):
            att(x, y, neighbors)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"in_channels": 0},
            {"in_channels": 2, "attention_dim": 0},
            {"in_channels": 2, "attention_type": "invalid"},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises((ValueError, TypeCheckError)):
            NeighborAttention(jr.key(0), **kwargs)

    @pytest.mark.parametrize(
        "attention_type,reduced_dtype",
        (
            ("cosine", jnp.float16),
            ("cosine", jnp.bfloat16),
            ("dot_product", jnp.bfloat16),
            ("dot_product", jnp.float16),
        ),
    )
    def test_mixed_precision(self, inputs, attention_type, reduced_dtype):
        x, y, neighbors = inputs
        att = NeighborAttention(jr.key(0), E, attention_type)
        assert_reduced_precision_close_to_full(
            att,
            filter_kwarg_names=["neighbors"],
            reduced_dtype=reduced_dtype,
            x=x,
            y=y,
            neighbors=neighbors,
        )
