import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.edge_drop import edge_drop
from neojax.nn.neighbor_search import (
    radius_neighbor_search,
)
from neojax.tests.conftest import assert_jittable

M, N, K = 30, 80, 12


@pytest.fixture
def neighbors():
    k1, k2 = jr.split(jr.key(0))
    data = jr.uniform(k1, (N, 2))
    queries = jr.uniform(k2, (M, 2))
    return radius_neighbor_search(data, queries, 0.25, K)


class TestEdgeDrop:
    def test_ratio_one_is_identity(self, neighbors):
        out = edge_drop(neighbors, key=jr.key(1), strategy="ratio", ratio=1.0)
        np.testing.assert_array_equal(out.mask, neighbors.mask)
        np.testing.assert_array_equal(out.sq_dist, neighbors.sq_dist)

    def test_ratio_keeps_subset_with_expected_rate(self, neighbors):
        out = edge_drop(neighbors, key=jr.key(1), strategy="ratio", ratio=0.5)
        # only existing neighbors can be kept
        assert bool(jnp.all(neighbors.mask | ~out.mask))
        kept = float(out.mask.sum() / neighbors.mask.sum())
        assert 0.35 < kept < 0.65
        # dropped entries follow the padding convention
        np.testing.assert_array_equal(
            np.asarray(out.sq_dist)[~np.asarray(out.mask)], 0.0
        )
        np.testing.assert_array_equal(out.index, neighbors.index)

    @pytest.mark.parametrize("max_neighbors", [1, 3, 5])
    def test_max_neighbors_caps_per_query(self, neighbors, max_neighbors):
        out = edge_drop(
            neighbors,
            key=jr.key(2),
            strategy="max_neighbors",
            max_neighbors=max_neighbors,
        )
        before = np.asarray(neighbors.mask.sum(-1))
        after = np.asarray(out.mask.sum(-1))
        np.testing.assert_array_equal(after, np.minimum(before, max_neighbors))
        assert bool(jnp.all(neighbors.mask | ~out.mask))

    def test_max_neighbors_is_random_and_depends_on_key(self, neighbors):
        a = edge_drop(
            neighbors, key=jr.key(2), strategy="max_neighbors", max_neighbors=2
        )
        b = edge_drop(
            neighbors, key=jr.key(3), strategy="max_neighbors", max_neighbors=2
        )
        assert not bool(jnp.all(a.mask == b.mask))

    def test_max_neighbors_larger_than_width_is_identity(self, neighbors):
        out = edge_drop(
            neighbors, key=jr.key(2), strategy="max_neighbors", max_neighbors=K + 5
        )
        np.testing.assert_array_equal(out.mask, neighbors.mask)

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
        assert out.mask.shape == (3, M, K)

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
