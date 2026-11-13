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

"""Jit-compatible fixed-radius neighbor search that returns sparse edge lists."""

import itertools
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

Radius = float | Float[Array, ""]
SearchsortedMethod = Literal["scan", "scan_unrolled", "sort", "compare_all"]

_MAX_CAPACITY = 2**30


class Neighbors(eqx.Module):
    """Sparse neighbor lists as a padded edge list in COO format.

    Edge `i` connects the query `query_index[i]` with the data point `data_index[i]`.
    The arrays have a fixed capacity for compatibility with jax, which is at
    least the number of edges.

    `query_index` is ascending, so all edges of a query are consecutive.
    Padding is at the end: `query_index` holds the sentinel `num_queries`, which
    `jax.ops.segment_sum` and the functions in `neojax.utils.segment_ops` drop,
    and `data_index` holds zero, a valid but meaningless index. The order of the
    neighbors of a single query depends on the search and is unspecified.

    ??? info "Internal Attributes"
        * **query_index** (`Int[Array, "capacity"]`): Index of the query per edge.
        * **data_index** (`Int[Array, "capacity"]`): Index of the data point per edge.
        * **overflowed** (`Bool[Array, ""]`): `True` if edges were dropped
            because the search ran out of capacity.
        * **num_queries** (`int`): Number of query points, static.
        * **searchsorted_method** (`str`): Method of `jnp.searchsorted` for
            `row_splits`, static. Default is `"scan_unrolled"`, the fastest on GPU.
    """

    query_index: Int[Array, "capacity"]
    data_index: Int[Array, "capacity"]
    overflowed: Bool[Array, ""]
    num_queries: int = eqx.field(static=True)
    searchsorted_method: SearchsortedMethod = eqx.field(
        static=True, default="scan_unrolled"
    )

    @property
    def mask(self) -> Bool[Array, "capacity"]:
        """`True` for actual edges, `False` for padding."""
        return self.query_index < self.num_queries

    @property
    def num_edges(self) -> Int[Array, ""]:
        """Number of edges, excluding padding."""
        return jnp.sum(self.mask)

    @property
    def row_splits(self) -> Int[Array, "splits"]:
        """Offsets of the queries: the edges of query `i` are `row_splits[i:i + 2]`."""
        return jnp.searchsorted(
            self.query_index,
            jnp.arange(self.num_queries + 1),
            side="left",
            method=self.searchsorted_method,
        )

    @property
    def neighbor_counts(self) -> Int[Array, "m"]:
        """Number of neighbors of each query."""
        return jnp.diff(self.row_splits)


class RegularGrid(eqx.Module):
    """A static regular grid, with the points flattened in row-major order.

    The nodes on axis `i` are `jnp.linspace(lower[i], upper[i], resolution[i])`.
    This matches `jnp.meshgrid(*axes, indexing="ij")` flattened with
    `reshape(-1, d)`. An axis with a resolution of one only has the node `lower[i]`.

    The search (in neojax neighbor search) on the grid uses this description
    to find candidates and to compute distances, so it is impossible to pass points
    that disagree with it.

    ??? info "Internal Attributes"
        * **lower** (`tuple[float, ...]`): Coordinate of the first node per axis.
        * **upper** (`tuple[float, ...]`): Coordinate of the last node per axis.
        * **resolution** (`tuple[int, ...]`): Number of nodes per axis.

    Args:
        lower: Coordinate of the first node per axis.
        upper: Coordinate of the last node per axis.
        resolution: Number of nodes per axis.

    Raises:
        ValueError: If the arguments differ in length, a resolution is not
            positive, the number of nodes does not fit into `int32`, or
            `upper` is not above `lower` on an axis with several nodes.
    """

    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    resolution: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        resolution: tuple[int, ...],
    ):
        if not len(lower) == len(upper) == len(resolution) > 0:
            raise ValueError(
                "`lower`, `upper` and `resolution` must have the same positive length."
            )
        if any(res <= 0 for res in resolution):
            raise ValueError(f"`resolution` must be positive, but got {resolution}.")
        if int(np.prod(resolution, dtype=np.int64)) >= _MAX_CAPACITY:
            raise ValueError(f"The grid {resolution} has too many nodes.")
        for low, high, res in zip(lower, upper, resolution, strict=True):
            if res > 1 and not high > low:
                raise ValueError("`upper` must be above `lower` on axes with nodes.")
        self.lower = tuple(float(low) for low in lower)
        self.upper = tuple(float(high) for high in upper)
        self.resolution = tuple(int(res) for res in resolution)

    @classmethod
    def from_points(
        cls, points: Float[Array, "... d"], tolerance: float = 1e-3
    ) -> "RegularGrid":
        """Recovers the grid from its points, and checks that they form one.

        This inspects values, so call it outside of `jit`.

        Args:
            points: The nodes, either flat as (n, d) in row-major order, or
                with the grid axes first as (n_1, ..., n_d, d).
            tolerance: Largest allowed deviation of a point from its node,
                as a fraction of the spacing on that axis.

        Returns:
            The grid. Search with it, as its nodes differ from `points` at most
            by the tolerance.

        Raises:
            ValueError: If `points` are traced, are not a uniform grid or are not
                in row-major order.
        """
        if isinstance(points, jax.core.Tracer):
            raise ValueError(
                "`RegularGrid.from_points` inspects values, call it outside of "
                "`jit` or construct a `RegularGrid` from its static description."
            )
        array = np.asarray(points)
        d = array.shape[-1]
        flat = array.reshape(-1, d)
        if flat.shape[0] == 0:
            raise ValueError("`points` must not be empty.")
        lower, upper = flat.min(0), flat.max(0)
        eps = np.finfo(flat.dtype).eps
        resolution = []
        for axis in range(d):
            # nodes of this axis are separated by more than the rounding of a value
            gap = max(1e-6 * (upper[axis] - lower[axis]), 8 * eps * np.abs(flat).max())
            levels = 1 + np.count_nonzero(np.diff(np.sort(flat[:, axis])) > gap)
            resolution.append(int(levels))
        message = "`points` do not form a uniform grid in row-major order"
        if array.ndim > 2 and array.shape[:-1] != tuple(resolution):
            raise ValueError(f"{message}: {array.shape[:-1]} vs. {resolution}.")
        if np.prod(resolution, dtype=np.int64) != flat.shape[0]:
            raise ValueError(f"{message}: found axes of {resolution} nodes.")
        grid = cls(tuple(lower.tolist()), tuple(upper.tolist()), tuple(resolution))
        spacing = grid._spacing
        nodes = np.asarray(grid.points(), dtype=np.float64)
        error = np.abs(nodes - flat) / spacing
        worst = max(tolerance, 16 * eps * np.abs(flat).max() / spacing.min())
        if error.max() > worst:
            raise ValueError(f"{message}: off by {error.max():.3g} of the spacing.")
        return grid

    @property
    def num_points(self) -> int:
        """Number of nodes of the grid."""
        return int(np.prod(self.resolution))

    @property
    def _spacing(self) -> np.ndarray:
        """Distance of neighboring nodes per axis, one for axes with a single node."""
        low, high = np.array(self.lower), np.array(self.upper)
        res = np.array(self.resolution)
        return np.where(res > 1, (high - low) / np.maximum(res - 1, 1), 1.0)

    @property
    def _strides(self) -> np.ndarray:
        """Row-major strides per axis."""
        res = np.array(self.resolution, np.int64)
        return np.append(np.cumprod(res[::-1])[::-1][1:], 1).astype(np.int32)

    def _axes(self) -> list[Array]:
        return [
            jnp.linspace(low, high, res)
            for low, high, res in zip(
                self.lower, self.upper, self.resolution, strict=True
            )
        ]

    def points(self) -> Float[Array, "n d"]:
        """Coordinates of the nodes, flattened in row-major order."""
        mesh = jnp.meshgrid(*self._axes(), indexing="ij")
        return jnp.stack(mesh, axis=-1).reshape(-1, len(self.resolution))

    def _points_at(self, index: Int[Array, "e"]) -> Float[Array, "e d"]:
        """Coordinates of the nodes with the given flat indices."""
        res = self.resolution
        return jnp.stack(
            [
                axis[(index // stride) % size]
                for axis, stride, size in zip(
                    self._axes(), self._strides.tolist(), res, strict=True
                )
            ],
            axis=-1,
        )


def _validate(radius: Radius, max_edges: int, max_candidates: int | None) -> None:
    """Checks the static arguments that are common to the searches.

    Args:
        radius: The neighbor search radius.
        max_edges: Capacity of the result, at least the number of edges.
        max_candidates: Number of points whose distance can be computed.
    """
    if not isinstance(radius, jax.core.Tracer) and not radius > 0:
        raise ValueError(f"`radius` must be positive, but got {radius}.")
    capacities = {"max_edges": max_edges}
    if max_candidates is not None:
        capacities["max_candidates"] = max_candidates
    for name, value in capacities.items():
        if not isinstance(value, int) or not 0 < value < _MAX_CAPACITY:
            raise ValueError(f"`{name}` must be an int in (0, 2^30), got {value}.")


def _default_candidates(
    candidate_factors: tuple[int, ...],
    fallback_factor: int,
    d: int,
    max_edges: int,
    num_pairs: int,
) -> int:
    """Capacity of the candidates from the expected number of candidates per edge.

    At most `2**30 - 1` capacity, the maximum representable integer in `int32`.

    Args:
        candidate_factors: Candidates per edge for 1-3 dimensions.
        fallback_factor: The upper fallback factor for capacity.
        d: The number of spatial dimensions.
        max_edges: Capacity of the result, at least the number of edges.
        num_pairs: The number of potential pairs. At most `m*n`.

    Returns:
        The capacity of the candidates.
    """
    if any(factor <= 0 for factor in candidate_factors):
        raise ValueError(
            f"`candidate_factors` must be positive, but got {candidate_factors}."
        )
    factor = (
        candidate_factors[d - 1] if d <= len(candidate_factors) else fallback_factor
    )
    return min(factor * max_edges, num_pairs, _MAX_CAPACITY - 1)


def _coordinates(*points: Array) -> tuple[Array, ...]:
    """Coordinates as constants in at least float32.

    The edges then don't depend on the precision of a model, and derivatives,
    also in Taylor mode, don't flow through the search.

    Args:
        points: Arrays of points. Typically data and query points.

    Returns:
        The point arrays with promoted dtypes to at least single precision.
    """
    dtype = jnp.promote_types(jnp.result_type(*points), jnp.float32)
    return tuple(jax.lax.stop_gradient(p).astype(dtype) for p in points)


def _empty(
    num_queries: int,
    max_edges: int,
    searchsorted_method: SearchsortedMethod = "scan_unrolled",
) -> Neighbors:
    """Creates empty Neighbor edge lists.

    Args:
        num_queries: The number of queries.
        max_edges: Capacity of the result, at least the number of edges.
        searchsorted_method: Method of `Neighbors.row_splits` of the result.

    Returns:
        An instance of `Neighbors` with no edges.
    """
    return Neighbors(
        query_index=jnp.full(max_edges, num_queries, jnp.int32),
        data_index=jnp.zeros(max_edges, jnp.int32),
        overflowed=jnp.array(False),
        num_queries=num_queries,
        searchsorted_method=searchsorted_method,
    )


def _finalize(
    query_index: Int[Array, " capacity"],
    data_index: Int[Array, " capacity"],
    overflowed: Bool[Array, ""],
    num_queries: int,
    raise_on_overflow: bool,
    searchsorted_method: SearchsortedMethod = "scan_unrolled",
    overflow_message: str = (
        "The neighbor search ran out of capacity and dropped edges. Increase "
        "`max_edges` or `max_candidates`, or pass `raise_on_overflow=False` to read "
        "`Neighbors.overflowed` instead."
    ),
) -> Neighbors:
    """Constructs neighbor edges from queries and data.

    Raises an overflow error at runtime (even under jit) if `raise_on_overflow`
    is `True` and any neighborhoods overflowed during search.
    """
    if raise_on_overflow:
        # on an integer array, as `error_if` on floats has no jet rule
        query_index = eqx.error_if(query_index, overflowed, overflow_message)
    return Neighbors(
        query_index, data_index, overflowed, num_queries, searchsorted_method
    )


def _checked_total(counts: Int[Array, "r"]) -> tuple[Array, Array, Array]:
    """Cumulative sum of non-negative counts, which detects wrapping around in `int32`.

    Hubs can make the number of candidates exceed `int32`. The first partial sum
    that exceeds it comes out negative, as each count fits into `int32`.

    Args:
        counts: The flattened lengths of the ranges.

    Returns:
        The partial sums, their total and whether they wrapped around.
    """
    partial = jnp.cumsum(counts)
    return partial, partial[-1], jnp.any(partial < 0)


def _expand_and_filter(
    points_at: Callable[[Array], Array],
    num_data: int,
    queries: Float[Array, "m d"],
    radius: Radius,
    start: Int[Array, "m r"],
    count: Int[Array, "m r"],
    max_edges: int,
    max_candidates: int,
    raise_on_overflow: bool,
    order: Int[Array, "n"] | None = None,
    searchsorted_method: SearchsortedMethod = "scan_unrolled",
) -> Neighbors:
    """Expands ranges of candidates, filters them by distance and compacts to COO.

    Each query has `r` ranges of consecutive candidates, ordered such that the
    resulting edges are sorted by query.

    Args:
        points_at: Maps data indices to the coordinates of the points.
        num_data: Number of data points.
        queries: The query points.
        radius: Radius of the balls.
        start: First position of each range.
        count: Length of each range.
        max_edges: Capacity of the output.
        max_candidates: Capacity of the candidates.
        raise_on_overflow: Whether to raise if a capacity is exceeded.
        order: Maps positions in the ranges to data indices, if they differ.
        searchsorted_method: Method of `Neighbors.row_splits` of the result.

    Returns:
        The edges.
    """
    m = queries.shape[0]
    rows_per_query = start.shape[1]
    start, count = start.reshape(-1), count.reshape(-1)
    partial, total, wrapped = _checked_total(count)

    slots = jnp.arange(max_candidates)
    row = jnp.repeat(
        jnp.arange(m * rows_per_query), count, total_repeat_length=max_candidates
    )
    position = start[row] + slots - (partial - count)[row]
    position = jnp.clip(position, 0, num_data - 1)
    query_of_slot = row // rows_per_query
    data_of_slot = position if order is None else order[position]

    sq_dist = jnp.sum((queries[query_of_slot] - points_at(data_of_slot)) ** 2, axis=-1)
    keep = (slots < total) & (sq_dist <= radius * radius)

    picked = jnp.nonzero(keep, size=max_edges, fill_value=max_candidates)[0]
    valid = picked < max_candidates
    picked = jnp.minimum(picked, max_candidates - 1)
    overflowed = wrapped | (total > max_candidates) | (jnp.sum(keep) > max_edges)
    return _finalize(
        jnp.where(valid, query_of_slot[picked], m),
        jnp.where(valid, data_of_slot[picked], 0),
        overflowed,
        m,
        raise_on_overflow,
        searchsorted_method,
    )


def _cell_ranges(
    data: Float[Array, "n d"],
    queries: Float[Array, "m d"],
    radius: Radius,
    searchsorted_method: SearchsortedMethod,
) -> tuple[Int[Array, "n"], Int[Array, "m r"], Int[Array, "m r"]]:
    """Ranges of candidates in the data sorted by cell, without a dense cell buffer.

    Cells that are adjacent along the last axis have consecutive hashes, which
    merges three cells into one range, so there are `3^(d-1)` ranges per query.

    Args:
        data: The `n` data points.
        queries: The `m` query points.
        radius: Radius of the neighborhood. May be traced.
        searchsorted_method: Method used in `jnp.searchsorted`.

    Returns:
        The order that sorts the data by cell, and the ranges into it.
    """
    n, d = data.shape
    lower, upper = jnp.min(data, axis=0), jnp.max(data, axis=0)

    # A pair within the radius must not end up two cells apart due to rounding in
    # the computation of the indices, which is relative to the extent of the data.
    eps = jnp.finfo(data.dtype).eps
    min_width = radius + 4 * eps * (upper - lower + radius)

    per_axis = jnp.maximum(jnp.floor((upper - lower) / min_width), 1.0)
    # coarsen to O(n) cells, which bounds the work and keeps the hashes in int32
    excess = jnp.maximum(jnp.prod(per_axis) / (2 * n), 1.0)
    per_axis = jnp.maximum(jnp.floor(per_axis / excess ** (1.0 / d)), 1.0)
    width = jnp.maximum((upper - lower) / per_axis, min_width)
    num_cells = per_axis.astype(jnp.int32)
    strides = jnp.append(jnp.cumprod(num_cells[::-1])[::-1][1:], jnp.ones(1, jnp.int32))

    last_cell = per_axis - 1
    cell = jnp.clip(jnp.floor((data - lower) / width), 0, last_cell).astype(jnp.int32)
    hashes = cell @ strides
    order = jnp.argsort(hashes)
    sorted_hashes = hashes[order]

    # a query can be outside of the bounding box, -2 and num_cells + 1 are out of reach
    query_cell = jnp.floor((queries - lower) / width)
    query_cell = jnp.clip(query_cell, -2, per_axis + 1).astype(jnp.int32)
    offsets = np.array(
        list(itertools.product((-1, 0, 1), repeat=d - 1)), np.int32
    ).reshape(3 ** (d - 1), d - 1)
    row = query_cell[:, None, : d - 1] + offsets[None]
    base = row @ strides[: d - 1]
    first = jnp.maximum(query_cell[:, -1:] - 1, 0)
    last = jnp.minimum(query_cell[:, -1:] + 1, num_cells[-1] - 1)
    valid = jnp.all((row >= 0) & (row < num_cells[: d - 1]), axis=-1) & (last >= first)
    start = jnp.searchsorted(
        sorted_hashes, base + first, side="left", method=searchsorted_method
    )
    end = jnp.searchsorted(
        sorted_hashes, base + last, side="right", method=searchsorted_method
    )
    return order, start, jnp.where(valid, end - start, 0)


def radius_neighbor_search(
    data: Float[Array, "n d"],
    queries: Float[Array, "m d"],
    radius: Radius,
    max_edges: int,
    max_candidates: int | None = None,
    raise_on_overflow: bool = True,
    candidate_factors: tuple[int, ...] = (2, 4, 9),
    searchsorted_method: SearchsortedMethod = "scan_unrolled",
) -> Neighbors:
    r"""Finds all pairs of queries and data points within a fixed radius.

    Returns an edge for each `x` in `queries` and `y` in `data` with
    $\|x - y\| \leq r$, i.e. points on the boundary of the ball are included and
    a point of `data` that coincides with the query is a neighbor.

    The points are sorted into cells at least as wide as the radius and the search
    only looks at the neighboring cells of a query. Use this for irregular data,
    and `radius_neighbor_search_grid` if the data is a regular grid.

    Coordinates are treated as constants for differentiation, also in Taylor mode,
    and distances are computed in at least float32. The radius may be traced.

    Args:
        data: Search space of possible neighbors, shaped `(n, d)`.
        queries: Points to find the neighbors of, shaped `(m, d)`.
        radius: Radius of the balls around the query points.
        max_edges: Capacity of the result, at least the number of edges.
        max_candidates: Number of pairs whose distance can be computed, at least
            the number of points in the neighboring cells over all queries.
            Default is `None`, which multiplies `max_edges` by `candidate_factors`.
        raise_on_overflow: Whether to raise a runtime error, also under `jit`, if
            a capacity is too small. If `False`, check `Neighbors.overflowed`.
            Edges of the last queries are then missing.
        candidate_factors: Candidates per edge for 1, 2, 3, ... dimensions, used if
            `max_candidates` is `None`. The default covers typical meshes, as a
            query scans about $3^d$ cells. Above its length, the factor is $3^d$.
        searchsorted_method: Method of `jnp.searchsorted` for this search and for
            `Neighbors.row_splits`. `"scan_unrolled"` is the fastest on GPU,
            `"scan"` can be faster on CPU.

    Returns:
        The edges, sorted by query.

    Raises:
        ValueError: If `radius` is not positive, a capacity is not a positive
            integer below $2^{30}$, or a candidate factor is not positive.
        RuntimeError: If `raise_on_overflow` is set and a capacity is too small.

    !!! info "Complexity"
        Sorting takes $\mathcal{O}(n \log n)$ and each query
        $\mathcal{O}(3^{d-1} \log n)$, in addition to one distance per candidate.
        Memory is $\mathcal{O}(n + 3^{d-1} m + c)$ for $c$ candidates.

    !!! info "Choosing the capacities"
        A change of capacity triggers a new compilation, so choose them once
        for a dataset with some slack over the largest sample.
    """
    _validate(radius, max_edges, max_candidates)
    n, d = data.shape
    m = queries.shape[0]
    if max_candidates is None:
        max_candidates = _default_candidates(
            candidate_factors, 3**d, d, max_edges, max(m * n, 1)
        )
    if n == 0 or m == 0:
        return _empty(m, max_edges, searchsorted_method)

    data, queries = _coordinates(data, queries)
    order, start, count = _cell_ranges(data, queries, radius, searchsorted_method)
    return _expand_and_filter(
        lambda index: data[index],
        n,
        queries,
        radius,
        start,
        count,
        max_edges,
        max_candidates,
        raise_on_overflow,
        order,
        searchsorted_method,
    )


def _grid_ranges(
    grid: RegularGrid, queries: Float[Array, "m d"], radius: Radius
) -> tuple[Int[Array, "m r"], Int[Array, "m r"]]:
    """Ranges of candidates on a grid, from arithmetic on the indices only.

    A query covers the nodes in the bounding box of its ball. Along the last axis
    these are consecutive, so a range per node in the other axes covers them.

    Args:
        grid: The grid, whose nodes are the data points.
        queries: Points to find the neighbors of, shaped `(m, d)`.
        radius: Radius of the balls around the query points. It must be static,
            since it determines the number of ranges per query.

    Returns:
        The starts and lengths of each range per query.
    """
    d = len(grid.resolution)
    spacing = grid._spacing
    lower = np.array(grid.lower)
    res = np.array(grid.resolution)
    # wider than the rounding, the distance filter decides exactly
    tolerance = 1e-3 + 16 * float(jnp.finfo(queries.dtype).eps) * res.max()

    low = jnp.ceil((queries - radius - lower) / spacing - tolerance)
    high = jnp.floor((queries + radius - lower) / spacing + tolerance)
    first = jnp.clip(low, 0, res - 1).astype(jnp.int32)
    last = jnp.clip(high, 0, res - 1).astype(jnp.int32)
    # clipping would put a window beyond the grid onto its outermost nodes
    outside = jnp.any((high < 0) | (low > res - 1), axis=-1, keepdims=True)

    # the number of ranges has to be static, so it is the widest possible window
    widths = np.floor(2 * radius / spacing + 3 * tolerance).astype(np.int64) + 1
    widths = np.minimum(widths, res)
    offsets = np.array(
        list(itertools.product(*[range(w) for w in widths[: d - 1]])), np.int32
    ).reshape(int(np.prod(widths[: d - 1])), d - 1)

    strides = grid._strides
    row = first[:, None, : d - 1] + offsets[None]
    valid = jnp.all(row <= last[:, None, : d - 1], axis=-1) & ~outside
    start = row @ strides[: d - 1] + first[:, -1:]
    count = (last[:, -1:] - first[:, -1:] + 1) * valid
    return start, count


def radius_neighbor_search_grid(
    grid: RegularGrid,
    queries: Float[Array, "m d"],
    radius: Radius,
    max_edges: int,
    max_candidates: int | None = None,
    raise_on_overflow: bool = True,
    candidate_factors: tuple[int, ...] = (2, 2, 3),
) -> Neighbors:
    r"""Finds all pairs of queries and nodes of a regular grid within a fixed radius.

    Returns the same edges as `radius_neighbor_search` on `grid.points()`, up to
    the order of the neighbors of a query, which is ascending here, and pairs
    within rounding error of the radius. On a regular grid the candidates of a
    query follow from its coordinates, so this skips the hashing, sorting and
    binary search of the general search.

    Coordinates are treated as constants for differentiation, also in Taylor mode,
    and distances are computed in at least float32.

    Args:
        grid: The grid, whose nodes are the data points.
        queries: Points to find the neighbors of, shaped `(m, d)`.
        radius: Radius of the balls around the query points. It must be static,
            since it determines the number of ranges per query.
        max_edges: Capacity of the result, at least the number of edges.
        max_candidates: Number of pairs whose distance can be computed, at least
            the number of nodes in the bounding boxes of the balls over all queries.
            Default is `None`, which multiplies `max_edges` by `candidate_factors`.
        raise_on_overflow: Whether to raise a runtime error, also under `jit`, if
            a capacity is too small. If `False`, check `Neighbors.overflowed`.
            Edges of the last queries are then missing.
        candidate_factors: Candidates per edge for 1, 2, 3, ... dimensions, used if
            `max_candidates` is `None`. The default covers typical grids, as a query
            scans the bounding box of its ball. Above its length, the factor is $2^d$.

    Returns:
        The edges, sorted by query.

    Raises:
        ValueError: If `radius` is traced or not positive, a capacity is not a
            positive integer below $2^{30}$, a candidate factor is not positive, or
            `queries` do not match the grid.
        RuntimeError: If `raise_on_overflow` is set and a capacity is too small.

    !!! info "Complexity"
        Each query takes one distance per candidate. Memory is
        $\mathcal{O}(w^{d-1} m + c)$ for $c$ candidates, where $w$ is the number of
        nodes across a ball, so this grows with a radius far above the spacing.
    """
    if isinstance(radius, jax.core.Tracer):
        raise ValueError(
            "`radius` must be static for the grid search, as it determines the "
            "number of ranges. Use `radius_neighbor_search` for a traced radius."
        )
    _validate(radius, max_edges, max_candidates)
    radius = float(radius)
    d = len(grid.resolution)
    if queries.shape[-1] != d:
        raise ValueError(f"`queries` must have {d} dimensions like the grid.")
    m = queries.shape[0]
    n = grid.num_points
    if max_candidates is None:
        max_candidates = _default_candidates(
            candidate_factors, 2**d, d, max_edges, max(m * n, 1)
        )
    if m == 0:
        return _empty(m, max_edges)

    (queries,) = _coordinates(queries)
    start, count = _grid_ranges(grid, queries, radius)
    return _expand_and_filter(
        grid._points_at,
        n,
        queries,
        radius,
        start,
        count,
        max_edges,
        max_candidates,
        raise_on_overflow,
    )
