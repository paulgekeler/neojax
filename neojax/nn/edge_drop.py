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

"""Implementation of the random neighbor sampling ("edge drop") during training."""

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, PRNGKeyArray

from neojax.nn.neighbor_search import Neighbors


def edge_drop(
    neighbors: Neighbors,
    *,
    key: PRNGKeyArray,
    strategy: Literal["ratio", "max_neighbors"],
    ratio: float | None = None,
    max_neighbors: int | None = None,
) -> Neighbors:
    r"""Randomly drops edges, a regularization for graph neural operators.

    The kept edges stay in their order and the dropped ones become padding at the
    end, so the result is again sorted by query and has the same capacity, which
    keeps it compatible with `jax.jit` and `jax.vmap`.

    - `"ratio"`: Every edge is kept independently with probability `ratio`.
    - `"max_neighbors"`: At most `max_neighbors` edges of each query are kept,
        chosen uniformly at random. Queries with fewer neighbors are unchanged.

    Args:
        neighbors: The edges.
        key: PRNG key to sample the edges with.
        strategy: How to sample the edges, `"ratio"` or `"max_neighbors"`.
        ratio: Probability of keeping an edge for the `"ratio"` strategy.
            Must be in (0, 1].
        max_neighbors: Maximum number of kept edges per query for
            the `"max_neighbors"` strategy. Must be a positive integer.

    Returns:
        The sampled edges with the same capacity as `neighbors`.

    Raises:
        ValueError: If `strategy` is invalid.
        ValueError: If the `"ratio"` strategy is used and `ratio` is not in (0, 1].
        ValueError: If the `"max_neighbors"` strategy is used and `max_neighbors`
            is not a positive integer.
    """
    capacity = neighbors.query_index.shape[0]
    if strategy == "ratio":
        if ratio is None or not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"`ratio` must be in (0, 1] for the 'ratio' strategy, but got {ratio}."
            )
        keep = jr.bernoulli(key, ratio, shape=(capacity,))
    elif strategy == "max_neighbors":
        if not isinstance(max_neighbors, int) or max_neighbors <= 0:
            raise ValueError(
                "`max_neighbors` must be a positive integer for the "
                f"'max_neighbors' strategy, but got {max_neighbors}."
            )
        # rank the edges of each query by a random score, keep the first ones
        scores = jr.uniform(key, (capacity,))
        order = jnp.lexsort((scores, neighbors.query_index))
        sorted_query = neighbors.query_index[order]
        first = jnp.searchsorted(
            sorted_query,
            sorted_query,
            side="left",
            method=neighbors.searchsorted_method,
        )
        rank = jnp.arange(capacity) - first
        keep = jnp.zeros(capacity, bool).at[order].set(rank < max_neighbors)
    else:
        raise ValueError(
            f"strategy must be one of ['ratio', 'max_neighbors'], but got '{strategy}'."
        )
    return _compact(neighbors, keep & neighbors.mask)


def _compact(neighbors: Neighbors, keep: Bool[Array, "capacity"]) -> Neighbors:
    """Moves the kept edges to the front, in order, and pads the rest.

    Args:
        neighbors: Neighbor edge lists.
        keep: The mask of edges to keep.

    Returns:
        The reordered edge lists.
    """
    capacity = keep.shape[0]
    picked = jnp.nonzero(keep, size=capacity, fill_value=capacity)[0]
    valid = picked < capacity
    picked = jnp.minimum(picked, capacity - 1)
    return eqx.tree_at(
        lambda n: (n.query_index, n.data_index),
        neighbors,
        (
            jnp.where(valid, neighbors.query_index[picked], neighbors.num_queries),
            jnp.where(valid, neighbors.data_index[picked], 0),
        ),
    )
