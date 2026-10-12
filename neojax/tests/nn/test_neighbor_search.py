import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.neighbor_search import Neighbors, radius_neighbor_search
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def get_brute_force_neighbors():
    def f(data, queries, radius):
        data, queries = np.asarray(data), np.asarray(queries)
        sq_dist = ((queries[:, None] - data[None]) ** 2).sum(-1)
        return [set(np.nonzero(row <= radius**2)[0]) for row in sq_dist]

    return f


@pytest.fixture
def get_valid_sets():
    def f(neighbors):
        index, mask = np.asarray(neighbors.index), np.asarray(neighbors.mask)
        return [set(index[i][mask[i]]) for i in range(index.shape[0])]

    return f


@pytest.mark.usefixtures("get_valid_sets", "get_brute_force_neighbors")
class TestRadiusNeighborSearch:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_matches_brute_force(self, d, get_valid_sets, get_brute_force_neighbors):
        k1, k2 = jr.split(jr.key(0))
        data = jr.uniform(k1, (60, d))
        queries = jr.uniform(k2, (25, d))
        radius = 0.4

        neighbors = radius_neighbor_search(data, queries, radius, max_neighbors=60)

        assert isinstance(neighbors, Neighbors)
        assert neighbors.index.shape == (25, 60)
        assert neighbors.mask.shape == (25, 60)
        assert neighbors.sq_dist.shape == (25, 60)
        assert get_valid_sets(neighbors) == get_brute_force_neighbors(
            data, queries, radius
        )

    def test_sq_dist_values_and_padding(self):
        k1, k2 = jr.split(jr.key(1))
        data = jr.uniform(k1, (40, 2))
        queries = jr.uniform(k2, (10, 2))
        neighbors = radius_neighbor_search(data, queries, 0.3, max_neighbors=16)

        expected = (
            (
                np.asarray(queries)[:, None]
                - np.asarray(data)[np.asarray(neighbors.index)]
            )
            ** 2
        ).sum(-1)
        mask = np.asarray(neighbors.mask)
        np.testing.assert_allclose(
            np.asarray(neighbors.sq_dist)[mask], expected[mask], rtol=1e-5
        )
        assert np.all(np.asarray(neighbors.sq_dist)[~mask] == 0.0)
        assert np.all(np.asarray(neighbors.sq_dist)[mask] <= 0.3**2)

    def test_coincident_point_and_boundary_are_included(self, get_valid_sets):
        data = jnp.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        queries = jnp.array([[0.0, 0.0]])
        neighbors = radius_neighbor_search(data, queries, 1.0, max_neighbors=3)
        assert get_valid_sets(neighbors) == [{0, 1}]

    def test_truncation_keeps_nearest(self, get_valid_sets):
        data = jnp.arange(10.0)[:, None]
        queries = jnp.array([[0.0]])
        neighbors = radius_neighbor_search(data, queries, 100.0, max_neighbors=3)
        assert get_valid_sets(neighbors) == [{0, 1, 2}]
        assert bool(neighbors.mask.all())

    def test_max_neighbors_clipped_to_num_data(self):
        data = jr.uniform(jr.key(2), (5, 2))
        queries = jr.uniform(jr.key(3), (4, 2))
        neighbors = radius_neighbor_search(data, queries, 0.5, max_neighbors=32)
        assert neighbors.index.shape == (4, 5)

    def test_empty_neighborhood(self):
        data = jnp.array([[0.0, 0.0], [0.1, 0.0]])
        queries = jnp.array([[5.0, 5.0]])
        neighbors = radius_neighbor_search(data, queries, 1.0, max_neighbors=2)
        assert not bool(neighbors.mask.any())

    @pytest.mark.parametrize("chunk_size", [1, 7, 25, 100])
    def test_chunked_equals_unchunked(self, chunk_size):
        k1, k2 = jr.split(jr.key(4))
        data = jr.uniform(k1, (50, 3))
        queries = jr.uniform(k2, (25, 3))
        full = radius_neighbor_search(data, queries, 0.5, max_neighbors=8)
        chunked = radius_neighbor_search(
            data, queries, 0.5, max_neighbors=8, chunk_size=chunk_size
        )
        np.testing.assert_array_equal(full.index, chunked.index)
        np.testing.assert_array_equal(full.mask, chunked.mask)
        np.testing.assert_allclose(full.sq_dist, chunked.sq_dist, rtol=1e-6)

    def test_jittable_and_vmappable(self):
        k1, k2 = jr.split(jr.key(5))
        data = jr.uniform(k1, (30, 2))
        queries = jr.uniform(k2, (12, 2))

        def search(d, q):
            return radius_neighbor_search(d, q, 0.4, max_neighbors=8, chunk_size=5)

        assert_filter_jittable(search, data, queries)

        # per-sample geometries, e.g. one mesh per batch element
        batched = jax.vmap(search)(
            jnp.stack([data, data + 0.1]), jnp.stack([queries] * 2)
        )
        assert batched.index.shape == (2, 12, 8)

    def test_invalid_arguments(self):
        data = jnp.ones((5, 2))
        queries = jnp.ones((3, 2))
        with pytest.raises(ValueError, match="radius"):
            radius_neighbor_search(data, queries, 0.0, max_neighbors=2)
        with pytest.raises(ValueError, match="max_neighbors"):
            radius_neighbor_search(data, queries, 0.1, max_neighbors=0)
        with pytest.raises(ValueError, match="chunk_size"):
            radius_neighbor_search(data, queries, 0.1, max_neighbors=2, chunk_size=0)
