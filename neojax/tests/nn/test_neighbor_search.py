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
import numpy as np
import pytest
from jax.experimental.jet import jet

import neojax.utils._jet_rules  # noqa: F401
from neojax.nn.neighbor_search import (
    Neighbors,
    RegularGrid,
    radius_neighbor_search,
    radius_neighbor_search_grid,
)
from neojax.tests.conftest import assert_filter_jittable

SEARCHSORTED_METHODS = ["scan", "scan_unrolled", "sort", "compare_all"]

# compiling the whole search is several times faster than running it op by op
cell_search = eqx.filter_jit(radius_neighbor_search)
grid_search = eqx.filter_jit(radius_neighbor_search_grid)


def make_points(kind, n, d, seed=0):
    """Point sets with very different neighborhood sizes, in the unit cube."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        points = rng.uniform(size=(n, d))
    elif kind == "clusters":
        centers = rng.uniform(0.2, 0.8, size=(5, d))
        points = centers[rng.integers(5, size=n)] + 0.02 * rng.normal(size=(n, d))
    elif kind == "hub":
        points = rng.uniform(size=(n, d))
        points[: n // 2] = 0.5 + 1e-3 * rng.normal(size=(n // 2, d))
    else:
        raise ValueError(kind)
    return jnp.asarray(points, jnp.float32)


def reference_edges(data, queries, radius):
    """All edges, with the float32 arithmetic of the search."""
    data, queries = np.asarray(data), np.asarray(queries)
    radius = np.float32(radius)
    pairs = []
    for i, query in enumerate(queries):
        hits = np.nonzero(((query - data) ** 2).sum(-1) <= radius * radius)[0]
        pairs += [(i, int(j)) for j in hits]
    return set(pairs)


def edges_of(neighbors):
    query, data = np.asarray(neighbors.query_index), np.asarray(neighbors.data_index)
    mask = np.asarray(neighbors.mask)
    return set(zip(query[mask].tolist(), data[mask].tolist(), strict=True))


def unstack(batched, i):
    return Neighbors(
        batched.query_index[i],
        batched.data_index[i],
        batched.overflowed[i],
        batched.num_queries,
    )


def assert_well_formed(neighbors, num_data):
    """Sorted by query with unique edges and padding according to the contract."""
    query, data = np.asarray(neighbors.query_index), np.asarray(neighbors.data_index)
    mask = np.asarray(neighbors.mask)
    assert np.all(np.diff(query) >= 0)
    assert np.all(query <= neighbors.num_queries)
    assert np.all(query[~mask] == neighbors.num_queries)
    assert np.all(data[~mask] == 0)
    assert np.all((data >= 0) & (data < max(num_data, 1)))
    assert len(edges_of(neighbors)) == int(mask.sum())


def assert_taylor_mode_matches_fixed_edges(search, nodes, queries):
    """Second derivative in Taylor mode with the edges held fixed, also under jit."""
    m = queries.shape[0]
    fixed = search(queries)

    def kernel(q, nb):
        diff = q[nb.query_index.clip(0, m - 1)] - nodes[nb.data_index]
        values = jnp.sum(diff**3, axis=-1) * nb.mask
        return jax.ops.segment_sum(values, nb.query_index, m, True)

    def f(q):
        return kernel(q, search(q))

    def along(t):
        return kernel(queries + t * direction, fixed)

    direction = jnp.ones_like(queries)
    series = ((direction, jnp.zeros_like(queries)),)
    t = jnp.zeros(())
    for wrap in (lambda fn: fn, jax.jit):
        primal, (first, second) = wrap(lambda q: jet(f, (q,), series))(queries)
        np.testing.assert_allclose(primal, along(t), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(first, jax.jacfwd(along)(t), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            second, jax.jacfwd(jax.jacfwd(along))(t), rtol=1e-3, atol=1e-3
        )


class TestNeighbors:
    @pytest.fixture
    def neighbors(self):
        return Neighbors(
            query_index=jnp.array([0, 0, 2, 2, 2, 4, 4]),
            data_index=jnp.array([3, 1, 0, 2, 4, 0, 0]),
            overflowed=jnp.array(False),
            num_queries=4,
        )

    def test_properties(self, neighbors):
        np.testing.assert_array_equal(neighbors.mask, [1, 1, 1, 1, 1, 0, 0])
        assert int(neighbors.num_edges) == 5
        np.testing.assert_array_equal(neighbors.row_splits, [0, 2, 2, 5, 5])
        np.testing.assert_array_equal(neighbors.neighbor_counts, [2, 0, 3, 0])
        assert neighbors.searchsorted_method == "scan_unrolled"
        # num_queries and searchsorted_method are static
        assert len(jax.tree_util.tree_leaves(neighbors)) == 3

    def test_searchsorted_method(self, neighbors):
        for method in SEARCHSORTED_METHODS:
            other = Neighbors(
                neighbors.query_index,
                neighbors.data_index,
                neighbors.overflowed,
                neighbors.num_queries,
                searchsorted_method=method,
            )
            assert other.searchsorted_method == method
            np.testing.assert_array_equal(other.row_splits, neighbors.row_splits)


EDGE_CASES = {
    "boundary_is_included": (
        jnp.array([[0.0, 0.0], [0.5, 0.0], [0.75, 0.0], [2.0, 0.0]]),
        jnp.array([[0.0, 0.0]]),
        0.5,
    ),
    "queries_far_outside": (
        make_points("uniform", 200, 2),
        jnp.array([[-1e6, 0.5], [0.5, 1e6], [1.04, 0.5], [-0.04, -0.04]]),
        0.05,
    ),
    "identical_points": (jnp.ones((20, 3)), jnp.ones((2, 3)), 0.1),
    "radius_above_extent": (
        make_points("uniform", 50, 2),
        make_points("uniform", 10, 2, seed=1),
        100.0,
    ),
    "offset_coordinates": (
        make_points("uniform", 300, 2) + 1000.0,
        make_points("uniform", 100, 2, seed=1) + 1000.0,
        0.1,
    ),
}


class TestCellSearch:
    @pytest.mark.parametrize(
        "kind,d,radius",
        [("uniform", 1, 0.01), ("clusters", 2, 0.05), ("hub", 3, 0.15)],
    )
    def test_matches_reference(self, kind, d, radius):
        data = make_points(kind, 600, d)
        queries = make_points("uniform", 200, d, seed=1)
        expected = reference_edges(data, queries, radius)

        # exactly the number of edges is enough
        neighbors = cell_search(
            data, queries, radius, len(expected), max_candidates=200 * 600
        )

        assert edges_of(neighbors) == expected
        assert not neighbors.overflowed
        assert_well_formed(neighbors, 600)

    @pytest.mark.parametrize("case", EDGE_CASES.values(), ids=EDGE_CASES.keys())
    def test_edge_cases(self, case):
        data, queries, radius = case
        expected = reference_edges(data, queries, radius)

        neighbors = cell_search(data, queries, radius, len(expected) + 1)

        assert edges_of(neighbors) == expected
        assert_well_formed(neighbors, data.shape[0])

    def test_empty(self):
        for n, m in [(0, 3), (3, 0)]:
            neighbors = cell_search(
                jnp.zeros((n, 2)), jnp.zeros((m, 2)), 0.5, 4, searchsorted_method="sort"
            )
            assert neighbors.query_index.shape == (4,)
            assert neighbors.num_queries == m
            assert int(neighbors.num_edges) == 0
            assert neighbors.searchsorted_method == "sort"

    def test_edge_overflow(self):
        data = make_points("uniform", 100, 2)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            radius_neighbor_search(data, data, 0.5, 10)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            eqx.filter_jit(lambda x: radius_neighbor_search(x, x, 0.5, 10))(data)

        neighbors = cell_search(data, data, 0.5, 10, raise_on_overflow=False)

        assert neighbors.overflowed
        # what is left is correct and in the right format
        assert int(neighbors.num_edges) == 10
        assert edges_of(neighbors) <= reference_edges(data, data, 0.5)
        assert_well_formed(neighbors, 100)

    def test_candidate_overflow(self):
        data = make_points("uniform", 100, 2)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            cell_search(data, data, 0.5, 10_000, max_candidates=5)

        neighbors = cell_search(
            data, data, 0.5, 10_000, max_candidates=50, raise_on_overflow=False
        )

        assert neighbors.overflowed
        assert edges_of(neighbors) <= reference_edges(data, data, 0.5)
        assert_well_formed(neighbors, 100)

    def test_candidates_beyond_int32_are_flagged(self):
        # a single cell with 50000 candidates for each of 50000 queries, 2.5e9 in all;
        # "scan" since "scan_unrolled" takes seconds to compile for 50000 points
        data = jnp.linspace(0.0, 1.0, 50_000)[:, None]

        neighbors = cell_search(
            data,
            data,
            2.0,
            1000,
            max_candidates=1000,
            raise_on_overflow=False,
            searchsorted_method="scan",
        )

        assert neighbors.overflowed

    def test_invalid_arguments(self):
        points = jnp.zeros((2, 2))
        for kwargs, match in [
            ({"radius": 0.0}, "radius"),
            ({"radius": -1.0}, "radius"),
            ({"max_edges": 0}, "max_edges"),
            ({"max_edges": 2**30}, "max_edges"),
            ({"max_candidates": 0}, "max_candidates"),
            ({"candidate_factors": (0, 4, 9)}, "candidate_factors"),
            ({"candidate_factors": (2, -1, 9)}, "candidate_factors"),
        ]:
            with pytest.raises(ValueError, match=match):
                cell_search(
                    points, points, **({"radius": 0.5, "max_edges": 4} | kwargs)
                )

    def test_jit_with_traced_radius(self):
        data = make_points("uniform", 100, 2)
        queries = make_points("uniform", 30, 2, seed=1)
        assert_filter_jittable(radius_neighbor_search, data, queries, 0.1, 400)

        neighbors = jax.jit(lambda r: cell_search(data, queries, r, 400))(
            jnp.float32(0.1)
        )

        assert edges_of(neighbors) == reference_edges(data, queries, 0.1)

    def test_vmap_over_geometries(self):
        data = jnp.stack(
            [
                make_points("uniform", 100, 2),
                make_points("uniform", 100, 2, seed=1),
                make_points("clusters", 100, 2),
            ]
        )
        queries = make_points("uniform", 30, 2, seed=5)

        batched = jax.vmap(lambda x: cell_search(x, queries, 0.1, 400))(data[:2])

        assert batched.query_index.shape == (2, 400)
        for i in range(2):
            expected = reference_edges(data[i], queries, 0.1)
            assert edges_of(unstack(batched, i)) == expected
        # the clustered geometry overflows, which raises for the whole batch
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            jax.vmap(lambda x: cell_search(x, data[2], 0.5, 400))(data)

    def test_second_derivative_in_taylor_mode(self):
        data = make_points("uniform", 100, 2)
        assert_taylor_mode_matches_fixed_edges(
            lambda q: radius_neighbor_search(data, q, 0.2, 800),
            data,
            make_points("uniform", 30, 2, seed=1),
        )

    def test_gradient_treats_coordinates_as_constants(self):
        data = make_points("uniform", 100, 2)
        queries = make_points("uniform", 30, 2, seed=1)

        def f(q):
            nb = cell_search(data, q, 0.2, 800)
            return jnp.sum(q[nb.query_index.clip(0, 29)] * nb.mask[:, None])

        grad = jax.grad(f)(queries)

        counts = cell_search(data, queries, 0.2, 800).neighbor_counts
        np.testing.assert_allclose(grad, np.repeat(np.asarray(counts)[:, None], 2, 1))

    def test_searchsorted_method(self):
        data = make_points("clusters", 300, 2)
        queries = make_points("clusters", 100, 2, seed=1)
        expected = reference_edges(data, queries, 0.05)
        for method in SEARCHSORTED_METHODS:
            neighbors = cell_search(
                data,
                queries,
                0.05,
                len(expected) + 1,
                max_candidates=100 * 300,
                searchsorted_method=method,
            )
            assert edges_of(neighbors) == expected
            assert neighbors.searchsorted_method == method

    def test_candidate_factors(self):
        data = make_points("uniform", 400, 2)
        expected = reference_edges(data, data, 0.2)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            cell_search(data, data, 0.2, len(expected), candidate_factors=(1, 1, 1))
        neighbors = cell_search(
            data, data, 0.2, len(expected), candidate_factors=(1, 50, 1)
        )
        assert edges_of(neighbors) == expected

        # dimensions beyond the given factors fall back to 3^d
        data = make_points("uniform", 200, 3)
        expected = reference_edges(data, data, 0.2)
        neighbors = cell_search(
            data, data, 0.2, len(expected) + 1, candidate_factors=(2,)
        )
        assert edges_of(neighbors) == expected

    def test_pairs_at_the_radius_with_many_cells(self):
        # float32 with about 16000 cells on an axis, where rounding of the cell
        # index is largest compared to the width of a cell; "scan" since
        # "scan_unrolled" takes seconds to compile for 50000 points
        n = 50_000
        radius = 2.0 ** (-14)
        rng = np.random.default_rng(0)
        for offset in [0.0, 64.0]:
            data = rng.uniform(0, 1, n).astype(np.float32) + np.float32(offset)
            queries = data[:2000] + np.float32(radius)
            data, queries = jnp.asarray(data[:, None]), jnp.asarray(queries[:, None])
            expected = reference_edges(data, queries, radius)

            neighbors = cell_search(
                data,
                queries,
                radius,
                len(expected) + 1,
                max_candidates=2_000_000,
                searchsorted_method="scan",
            )

            assert edges_of(neighbors) == expected

    def test_coarsening_keeps_all_edges(self):
        # a small radius compared to the extent, due to a single outlier
        data = make_points("clusters", 500, 3)
        data = data.at[0].set(1e4)
        queries = make_points("clusters", 100, 3, seed=1)
        expected = reference_edges(data, queries, 1e-2)

        neighbors = cell_search(
            data, queries, 1e-2, len(expected) + 1, max_candidates=100 * 500
        )

        assert edges_of(neighbors) == expected

    def test_default_capacity_covers_typical_meshes(self):
        data = make_points("uniform", 5000, 3)
        queries = make_points("uniform", 2000, 3, seed=1)
        expected = reference_edges(data, queries, 0.1)

        neighbors = cell_search(data, queries, 0.1, len(expected) + 1)

        assert edges_of(neighbors) == expected


def test_low_precision_coordinates_give_the_float32_edges():
    data = make_points("clusters", 400, 2)
    queries = make_points("uniform", 100, 2, seed=1)
    grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (16, 16))
    for dtype in [jnp.float16, jnp.bfloat16]:
        low_data, low_queries = data.astype(dtype), queries.astype(dtype)
        high_data = low_data.astype(jnp.float32)
        high_queries = low_queries.astype(jnp.float32)
        for low, high in [
            (
                cell_search(low_data, low_queries, 0.1, 4000),
                cell_search(high_data, high_queries, 0.1, 4000),
            ),
            (
                grid_search(grid, low_queries, 0.1, 4000),
                grid_search(grid, high_queries, 0.1, 4000),
            ),
        ]:
            np.testing.assert_array_equal(low.query_index, high.query_index)
            np.testing.assert_array_equal(low.data_index, high.data_index)


GRID_CASES = {
    "1d": ((-1.0,), (1.0,), (33,), 0.2),
    "2d": ((-1.0, -1.0), (1.0, 1.0), (32, 32), 0.2),
    "3d": ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (12, 12, 12), 0.35),
    "radius_below_spacing": ((-1.0, -1.0), (1.0, 1.0), (11, 11), 0.05),
    "radius_exactly_three_spacings": ((0.0, 0.0), (1.0, 1.0), (9, 9), 0.375),
    "radius_far_above_spacing": ((-1.0, -1.0), (1.0, 1.0), (40, 40), 0.9),
    "radius_above_domain": ((-1.0, -1.0), (1.0, 1.0), (8, 8), 10.0),
    "anisotropic": ((0.0, -10.0), (1.0, 10.0), (64, 5), 0.9),
    "offset_domain": ((100.0, -50.0), (101.0, -49.0), (20, 20), 0.1),
    "single_node_axis": ((0.0, 0.5), (1.0, 0.5), (11, 1), 0.3),
    "single_node": ((0.25, 0.25), (0.25, 0.25), (1, 1), 0.3),
}


class TestGridSearch:
    @pytest.mark.parametrize("case", GRID_CASES.values(), ids=GRID_CASES.keys())
    def test_matches_reference(self, case):
        lower, upper, resolution, radius = case
        grid = RegularGrid(lower, upper, resolution)
        nodes = np.asarray(grid.points())
        rng = np.random.default_rng(0)
        low, high = np.array(lower), np.array(upper)
        center = (low + high) / 2
        queries = np.concatenate(
            [
                # inside, far beyond and exactly on the nodes
                center + rng.uniform(-0.5, 0.5, (50, len(low))) * (high - low),
                center + rng.uniform(-0.5, 0.5, (50, len(low))) * 3 * (high - low + 1),
                nodes[rng.integers(len(nodes), size=50)],
            ]
        )
        queries = jnp.asarray(queries, jnp.float32)
        expected = reference_edges(nodes, queries, radius)

        neighbors = grid_search(
            grid,
            queries,
            radius,
            len(expected) + 1,
            max_candidates=150 * grid.num_points,
        )

        assert edges_of(neighbors) == expected
        assert not neighbors.overflowed
        assert_well_formed(neighbors, grid.num_points)

    def test_matches_cell_search_with_ascending_neighbors(self):
        grid = RegularGrid((-1.0, -1.0), (1.0, 1.0), (64, 64))
        queries = 2 * make_points("clusters", 1000, 2) - 1

        on_grid = grid_search(grid, queries, 0.1, 40_000)
        cell = cell_search(grid.points(), queries, 0.1, 40_000)

        assert edges_of(on_grid) == edges_of(cell)
        query = np.asarray(on_grid.query_index)
        data = np.asarray(on_grid.data_index)
        edges = query[query < 1000] * grid.num_points + data[query < 1000]
        assert np.all(np.diff(edges) > 0)

    def test_default_capacity_covers_typical_grids(self):
        for grid, radius in (
            (RegularGrid((0.0, 0.0), (1.0, 1.0), (64, 64)), 0.05),
            (RegularGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (24, 24, 24)), 0.1),
        ):
            d = len(grid.resolution)
            queries = make_points("uniform", 2000, d)
            expected = reference_edges(grid.points(), queries, radius)

            neighbors = grid_search(grid, queries, radius, len(expected) + 1)

            assert edges_of(neighbors) == expected

    def test_candidate_factors(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (32, 32))
        queries = make_points("uniform", 200, 2)
        expected = reference_edges(grid.points(), queries, 0.1)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            grid_search(grid, queries, 0.1, len(expected), candidate_factors=(1, 1, 1))

        neighbors = grid_search(
            grid, queries, 0.1, len(expected), candidate_factors=(1, 5, 1)
        )

        assert edges_of(neighbors) == expected

    def test_overflow(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (32, 32))
        queries = make_points("uniform", 100, 2)
        with pytest.raises(RuntimeError, match="ran out of capacity"):
            grid_search(grid, queries, 0.3, 90_000, max_candidates=10)
        candidates = grid_search(
            grid, queries, 0.3, 90_000, max_candidates=10, raise_on_overflow=False
        )
        assert candidates.overflowed

        edges = grid_search(
            grid, queries, 0.3, 50, max_candidates=90_000, raise_on_overflow=False
        )
        assert edges.overflowed
        assert int(edges.num_edges) == 50
        assert_well_formed(edges, grid.num_points)

    def test_candidates_beyond_int32_are_flagged(self):
        grid = RegularGrid((0.0,), (1.0,), (50_000,))
        queries = jnp.full((50_000, 1), 0.5)

        neighbors = grid_search(
            grid, queries, 2.0, 1000, max_candidates=1000, raise_on_overflow=False
        )

        assert neighbors.overflowed

    def test_arguments(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (4, 4))
        empty = grid_search(grid, jnp.zeros((0, 2)), 0.3, 8)
        assert int(empty.num_edges) == 0
        # a concrete array is a static radius
        neighbors = radius_neighbor_search_grid(
            grid, jnp.zeros((3, 2)), jnp.float32(0.5), 30
        )
        assert not neighbors.overflowed

        with pytest.raises(ValueError, match="dimensions"):
            grid_search(grid, jnp.zeros((3, 3)), 0.3, 8)
        with pytest.raises(ValueError, match="candidate_factors"):
            grid_search(grid, jnp.zeros((3, 2)), 0.5, 8, candidate_factors=(2, 0, 3))
        with pytest.raises(ValueError, match="static"):
            jax.jit(lambda r: grid_search(grid, jnp.zeros((3, 2)), r, 8))(
                jnp.float32(0.3)
            )

    def test_jit_with_grid_argument(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (16, 16))
        queries = make_points("uniform", 30, 2)
        assert_filter_jittable(radius_neighbor_search_grid, grid, queries, 0.1, 500)

        neighbors = jax.jit(lambda g, q: grid_search(g, q, 0.1, 500))(grid, queries)

        assert edges_of(neighbors) == reference_edges(grid.points(), queries, 0.1)

    def test_vmap_over_queries(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (16, 16))
        queries = jnp.stack([make_points("uniform", 30, 2, seed=s) for s in range(3)])

        batched = jax.vmap(lambda q: grid_search(grid, q, 0.1, 400))(queries)

        for i in range(3):
            expected = reference_edges(grid.points(), queries[i], 0.1)
            assert edges_of(unstack(batched, i)) == expected

    def test_second_derivative_in_taylor_mode(self):
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (16, 16))
        assert_taylor_mode_matches_fixed_edges(
            lambda q: radius_neighbor_search_grid(grid, q, 0.2, 800),
            grid.points(),
            make_points("uniform", 30, 2),
        )


class TestRegularGrid:
    def test_points(self):
        grid = RegularGrid((-1.0, 0.0), (1.0, 2.0), (4, 3))
        axes = [jnp.linspace(-1.0, 1.0, 4), jnp.linspace(0.0, 2.0, 3)]
        mesh = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        np.testing.assert_array_equal(grid.points(), mesh.reshape(-1, 2))
        np.testing.assert_array_equal(
            grid._points_at(jnp.arange(grid.num_points)), grid.points()
        )
        assert grid.num_points == 12
        assert jax.tree_util.tree_leaves(grid) == []

    def test_invalid_arguments(self):
        for args, match in [
            (((0.0,), (1.0, 1.0), (4,)), "same positive length"),
            (((), (), ()), "same positive length"),
            (((0.0,), (1.0,), (0,)), "positive"),
            (((1.0,), (1.0,), (4,)), "above"),
            (((2.0,), (1.0,), (4,)), "above"),
            (((0.0,), (1.0,), (2**31,)), "too many"),
            (((0.0,) * 3, (1.0,) * 3, (2**11,) * 3), "too many"),
        ]:
            with pytest.raises(ValueError, match=match):
                RegularGrid(*args)

    def test_from_points_recovers_grid(self):
        for grid in [
            RegularGrid((-1.0, -1.0), (1.0, 1.0), (64, 32)),
            RegularGrid((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (5, 6, 7)),
            RegularGrid((0.0,), (1.0,), (50,)),
            RegularGrid((1000.0, -7.0), (1001.0, -6.0), (20, 20)),
            RegularGrid((0.0, 0.5), (1.0, 0.5), (11, 1)),
        ]:
            recovered = RegularGrid.from_points(grid.points())
            assert recovered.resolution == grid.resolution
            np.testing.assert_allclose(recovered.lower, grid.lower, rtol=1e-6)
            np.testing.assert_allclose(recovered.upper, grid.upper, rtol=1e-6)

        # axes first
        points = RegularGrid((0.0, 0.0), (1.0, 1.0), (4, 3)).points()
        assert RegularGrid.from_points(points.reshape(4, 3, 2)).resolution == (4, 3)

    def test_from_points_rejects_other_points(self):
        # same tensor-product structure with the right number of nodes on each axis
        axis = jnp.linspace(0.0, 1.0, 16) ** 2
        non_uniform = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1)
        axes = [jnp.linspace(0.0, 1.0, 4), jnp.linspace(0.0, 1.0, 4)]
        column_major = jnp.stack(jnp.meshgrid(*axes, indexing="xy"), axis=-1)
        grid = RegularGrid((0.0, 0.0), (1.0, 1.0), (8, 8))
        for points in [
            RegularGrid((0.0, 0.0), (1.0, 1.0), (4, 3)).points().reshape(3, 4, 2),
            non_uniform.reshape(-1, 2),
            jax.random.permutation(jax.random.key(0), grid.points()),
            column_major.reshape(-1, 2),
            make_points("uniform", 100, 2),
        ]:
            with pytest.raises(ValueError, match="uniform grid"):
                RegularGrid.from_points(points)

        with pytest.raises(ValueError, match="outside of"):
            jax.jit(RegularGrid.from_points)(grid.points())
