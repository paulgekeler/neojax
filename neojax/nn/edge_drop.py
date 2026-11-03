"""Implementation of the random neighbor sampling ("edge drop") during training."""

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from neojax.nn.neighbor_search import Neighbors


def edge_drop(
    neighbors: Neighbors,
    *,
    key: PRNGKeyArray,
    strategy: Literal["ratio", "max_neighbors"],
    ratio: float | None = None,
    max_neighbors: int | None = None,
) -> Neighbors:
    r"""Randomly drops neighbors, a regularization for graph neural operators.

    Masks out neighbors of padded `Neighbors`. The shape of the lists is unchanged,
    so the result is compatible with `jax.jit` and `jax.vmap`. Dropped
    entries become padding: they are masked out and their squared distance is zero.

    - `"ratio"`: Every neighbor is kept independently with probability `ratio`.
    - `"max_neighbors"`: At most `max_neighbors` neighbors of each query are kept,
        chosen uniformly at random. Queries with fewer neighbors are unchanged.

    Args:
        neighbors: The padded neighbor lists, shaped (m, k).
        key: PRNG key to sample the neighbors with.
        strategy: How to sample the neighbors, `"ratio"` or `"max_neighbors"`.
        ratio: Probability of keeping a neighbor for the `"ratio"` strategy.
            Must be in (0, 1].
        max_neighbors: Maximum number of kept neighbors per query for
            the `"max_neighbors"` strategy. Must be a positive integer.

    Returns:
        The sampled neighbor lists with the same shapes as `neighbors`.

    Raises:
        ValueError: If `strategy` is invalid.
        ValueError: If the `"ratio"` strategy is used and `ratio` is not in (0, 1].
        ValueError: If the `"max_neighbors"` strategy is used and `max_neighbors`
            is not a positive integer.

    !!! info "Naming"
        The `max_neighbors` argument here caps the number of *kept* neighbors at
        random. It is not the width `max_neighbors` of `radius_neighbor_search`,
        which keeps the nearest neighbors.
    """
    if strategy == "ratio":
        if ratio is None or not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"`ratio` must be in (0, 1] for the 'ratio' strategy, but got {ratio}."
            )
        keep = jr.bernoulli(key, ratio, shape=neighbors.mask.shape)
    elif strategy == "max_neighbors":
        if not isinstance(max_neighbors, int) or max_neighbors <= 0:
            raise ValueError(
                "`max_neighbors` must be a positive integer for the "
                f"'max_neighbors' strategy, but got {max_neighbors}."
            )
        k = neighbors.mask.shape[-1]
        if max_neighbors >= k:
            return neighbors
        # random scores, padding last: keep the entries with the largest scores
        scores = jr.uniform(key, neighbors.mask.shape)
        scores = jnp.where(neighbors.mask, scores, -jnp.inf)
        threshold = jax.lax.top_k(scores, max_neighbors)[0][:, -1:]
        keep = scores >= threshold
    else:
        raise ValueError(
            f"strategy must be one of ['ratio', 'max_neighbors'], but got '{strategy}'."
        )

    mask = neighbors.mask & keep
    return Neighbors(
        index=neighbors.index,
        mask=mask,
        sq_dist=jnp.where(mask, neighbors.sq_dist, 0.0),
    )
