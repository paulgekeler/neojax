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

from neojax.nn.edge_drop import edge_drop
from neojax.nn.neighbor_search import Neighbors, radius_neighbor_search
from neojax.tests.conftest import assert_jittable

M, N, CAPACITY = 30, 80, 600


@pytest.fixture
def neighbors():
    k1, k2 = jr.split(jr.key(0))
    data = jr.uniform(k1, (N, 2))
    queries = jr.uniform(k2, (M, 2))
    return radius_neighbor_search(data, queries, 0.25, CAPACITY)


def edges_of(neighbors):
    query, data = np.asarray(neighbors.query_index), np.asarray(neighbors.data_index)
    mask = np.asarray(neighbors.mask)
    return set(zip(query[mask].tolist(), data[mask].tolist(), strict=True))


def assert_well_formed(neighbors):
    """Sorted by query, padding at the end, with the padding convention."""
    query, data = np.asarray(neighbors.query_index), np.asarray(neighbors.data_index)
    mask = np.asarray(neighbors.mask)
    assert np.all(np.diff(query) >= 0)
    assert np.all(query[~mask] == neighbors.num_queries)
    assert np.all(data[~mask] == 0)


class TestEdgeDrop:
    def test_ratio_one_is_identity(self, neighbors):
        out = edge_drop(neighbors, key=jr.key(1), strategy="ratio", ratio=1.0)
        np.testing.assert_array_equal(out.query_index, neighbors.query_index)
        np.testing.assert_array_equal(out.data_index, neighbors.data_index)

    def test_ratio_keeps_subset_with_expected_rate(self, neighbors):
        out = edge_drop(neighbors, key=jr.key(1), strategy="ratio", ratio=0.5)
        # only existing edges can be kept
        assert edges_of(out) <= edges_of(neighbors)
        kept = int(out.num_edges) / int(neighbors.num_edges)
        assert 0.35 < kept < 0.65
        assert_well_formed(out)
        assert out.query_index.shape == neighbors.query_index.shape

    @pytest.mark.parametrize("max_neighbors", [1, 3, 5])
    def test_max_neighbors_caps_per_query(self, neighbors, max_neighbors):
        out = edge_drop(
            neighbors,
            key=jr.key(2),
            strategy="max_neighbors",
            max_neighbors=max_neighbors,
        )
        before = np.asarray(neighbors.neighbor_counts)
        after = np.asarray(out.neighbor_counts)
        np.testing.assert_array_equal(after, np.minimum(before, max_neighbors))
        assert edges_of(out) <= edges_of(neighbors)
        assert_well_formed(out)

    def test_max_neighbors_is_random_and_depends_on_key(self, neighbors):
        a = edge_drop(
            neighbors, key=jr.key(2), strategy="max_neighbors", max_neighbors=2
        )
        b = edge_drop(
            neighbors, key=jr.key(3), strategy="max_neighbors", max_neighbors=2
        )
        assert edges_of(a) != edges_of(b)

    def test_max_neighbors_is_uniform(self, neighbors):
        # each neighbor of a query with 4 neighbors is kept in about half the draws
        counts = np.asarray(neighbors.neighbor_counts)
        query = int(np.nonzero(counts >= 4)[0][0])
        keys = jr.split(jr.key(9), 400)
        out = jax.vmap(
            lambda key: edge_drop(
                neighbors, key=key, strategy="max_neighbors", max_neighbors=2
            )
        )(keys)
        kept = {}
        for i in range(400):
            one = Neighbors(out.query_index[i], out.data_index[i], out.overflowed[i], M)
            for q, d in edges_of(one):
                if q == query:
                    kept[d] = kept.get(d, 0) + 1
        rates = np.array(list(kept.values())) / 400
        np.testing.assert_allclose(rates, 2 / counts[query], atol=0.1)

    def test_max_neighbors_larger_than_any_neighborhood_is_identity(self, neighbors):
        largest = int(neighbors.neighbor_counts.max())
        out = edge_drop(
            neighbors,
            key=jr.key(2),
            strategy="max_neighbors",
            max_neighbors=largest + 5,
        )
        assert edges_of(out) == edges_of(neighbors)

    def test_keeps_overflow_flag_and_method(self, neighbors):
        flagged = Neighbors(
            neighbors.query_index,
            neighbors.data_index,
            jnp.array(True),
            M,
            searchsorted_method="sort",
        )
        out = edge_drop(flagged, key=jr.key(1), strategy="ratio", ratio=0.5)
        assert bool(out.overflowed)
        assert out.searchsorted_method == "sort"

    def test_jittable_and_vmappable(self, neighbors):
        assert_jittable(
            lambda nb, key: edge_drop(nb, key=key, strategy="ratio", ratio=0.7),
            neighbors,
            jr.key(4),
        )
        keys = jr.split(jr.key(5), 3)
        out = jax.vmap(
            lambda key: edge_drop(
                neighbors, key=key, strategy="max_neighbors", max_neighbors=2
            )
        )(keys)
        assert out.query_index.shape == (3, CAPACITY)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"strategy": "ratio"},
            {"strategy": "ratio", "ratio": 0.0},
            {"strategy": "ratio", "ratio": 1.5},
            {"strategy": "max_neighbors"},
            {"strategy": "max_neighbors", "max_neighbors": 0},
        ],
    )
    def test_invalid_arguments_raise(self, neighbors, kwargs):
        with pytest.raises(ValueError):
            edge_drop(neighbors, key=jr.key(0), **kwargs)
