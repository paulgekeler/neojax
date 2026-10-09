"""Implementation of a jit-compatible fixed-radius neighbor search."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Neighbors(eqx.Module):
    """Padded fixed-width neighbor lists.

    Stores for each of the `m` query points the indices of up to `k`
    neighboring data points. All shapes are static, for compatibility
    with jax.

    Entries with `mask == False` are padding and must be ignored.
    Their `index` is a valid (but meaningless) index into the data points
    and their `sq_dist` is zero.

    ??? info "Internal Attributes"
        * **index** (`Int[Array, "m k"]`): Indices of the neighbors in the data points.
        * **mask** (`Bool[Array, "m k"]`): `True` for entries that are
            actual neighbors, `False` for padding.
        * **sq_dist** (`Float[Array, "m k"]`): Squared distances between
            the query points and their neighbors. Zero for padding.
    """

    index: Int[Array, "m k"]
    mask: Bool[Array, "m k"]
    sq_dist: Float[Array, "m k"]


def radius_neighbor_search(
    data: Float[Array, "n d"],
    queries: Float[Array, "m d"],
    radius: float,
    max_neighbors: int,
    chunk_size: int | None = None,
) -> Neighbors:
    r"""Finds the neighbors of each query point within a fixed radius.

    For each point `x` in `queries`, returns the indices of the points `y`
    in `data` with $\|x - y\| \leq r$, i.e. points on the boundary of the ball
    are included. A point of `data` that coincides with the query is a neighbor.

    The result has a fixed width of `max_neighbors`. If more than
    `max_neighbors` points are inside the ball, the `max_neighbors` nearest
    are kept. If less are inside, the remaining entries are masked out.

    Args:
        data: Search space of possible neighbors, shaped (n, d).
        queries: Points to find the neighbors of, shaped (m, d).
        radius: Radius of the balls around the query points.
        max_neighbors: Maximum number of neighbors per query point.
            Clipped to `n` if larger than the number of data points.
        chunk_size: Number of query points to process at once.
            The search computes (chunk_size, n) pairwise distances at a time,
            so a smaller chunk size lowers peak memory at the cost of speed.
            Default is `None`, all query points at once.

    Returns:
        The neighbor lists, with `index`, `mask` and `sq_dist` shaped (m, k)
        where `k = min(max_neighbors, n)`.

    Raises:
        ValueError: If `radius` is not positive.
        ValueError: If `max_neighbors` or `chunk_size` is not a positive integer.

    !!! info "Complexity"
        The search is exact but brute force, $\mathcal{O}(m \cdot n)$ in compute
        and $\mathcal{O}(c \cdot n)$ in memory for chunk size $c$.
        This is typically fine for the problem sizes considered.

    !!! info "Choosing `max_neighbors`"
        Neighbors beyond `max_neighbors` are dropped silently.
        Check `neighbors.mask.all(axis=-1)` for a representative sample:
        if it is `True` for some query, that query's ball may be truncated.
    """
    if not radius > 0:
        raise ValueError(f"`radius` must be positive, but got {radius}.")
    if not isinstance(max_neighbors, int) or max_neighbors <= 0:
        raise ValueError("`max_neighbors` must be a positive integer.")
    if chunk_size is not None and (not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError("`chunk_size` must be a positive integer or None.")

    k = min(max_neighbors, data.shape[0])
    sq_radius = radius**2

    def _search_chunk(
        q: Float[Array, "c d"],
    ) -> tuple[Int[Array, "c k"], Bool[Array, "c k"], Float[Array, "c k"]]:
        sq_dist = jnp.sum((q[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        neg_sq_dist, index = jax.lax.top_k(-sq_dist, k)
        sq_dist_k = -neg_sq_dist
        mask = sq_dist_k <= sq_radius
        return index, mask, jnp.where(mask, sq_dist_k, 0.0)

    m = queries.shape[0]
    if chunk_size is None or chunk_size >= m:
        index, mask, sq_dist = _search_chunk(queries)
    else:
        n_chunks = -(-m // chunk_size)
        padded = jnp.pad(queries, ((0, n_chunks * chunk_size - m), (0, 0)))
        index, mask, sq_dist = jax.lax.map(
            _search_chunk, padded.reshape(n_chunks, chunk_size, -1)
        )
        index = index.reshape(-1, k)[:m]
        mask = mask.reshape(-1, k)[:m]
        sq_dist = sq_dist.reshape(-1, k)[:m]

    return Neighbors(index=index, mask=mask, sq_dist=sq_dist)
