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

"""Reductions over segments of an edge list that tolerate padding.

Every function takes `segment_ids` that may contain the sentinel `num_segments`
for padding, see `Neighbors`. Padded entries do not contribute to any segment.
The first axis of the data indexes the edges, trailing axes are reduced elementwise.

For the sum over segments use `jax.ops.segment_sum`, which drops the sentinel.
Pass `indices_are_sorted=True` for sorted ids, it matters on GPU.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _expand_like(mask: Array, data: Array) -> Array:
    """Appends axes to an edge mask to broadcast it against `data`."""
    return mask.reshape(mask.shape + (1,) * (data.ndim - 1))


def segment_mean(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, "e"],
    num_segments: int,
    indices_are_sorted: bool = False,
) -> Float[Array, "num_segments ..."]:
    """Mean over each segment. Empty segments are zero.

    Args:
        data: Values per edge, shaped (e, ...).
        segment_ids: Segment per edge, `num_segments` for padding.
        num_segments: Number of segments.
        indices_are_sorted: Whether `segment_ids` is ascending.

    Returns:
        The means, shaped (num_segments, ...).
    """
    total = jax.ops.segment_sum(
        data, segment_ids, num_segments, indices_are_sorted=indices_are_sorted
    )
    count = jax.ops.segment_sum(
        jnp.ones_like(segment_ids, dtype=data.dtype),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
    )
    return total / _expand_like(jnp.maximum(count, 1), total)


def segment_max(
    data: Float[Array, "e ..."],
    segment_ids: Int[Array, "e"],
    num_segments: int,
    indices_are_sorted: bool = False,
) -> Float[Array, "num_segments ..."]:
    """Maximum over each segment that supports Taylor mode. Empty segments are zero.

    `jax.ops.segment_max` has no jet rule. Here the maximum only selects which
    entries win and the value is a sum over the winners, so all derivatives flow
    through the sum. Tied entries share the derivative evenly, like `jnp.max`.

    Args:
        data: Values per edge, shaped (e, ...).
        segment_ids: Segment per edge, `num_segments` for padding.
        num_segments: Number of segments.
        indices_are_sorted: Whether `segment_ids` is ascending.

    Returns:
        The maxima, shaped (num_segments, ...).
    """
    frozen = jax.lax.stop_gradient(data)
    top = jax.ops.segment_max(
        frozen, segment_ids, num_segments, indices_are_sorted=indices_are_sorted
    )
    valid = _expand_like(segment_ids < num_segments, data)
    # the id of padding is clamped here, `valid` removes its entries
    winner = valid & (frozen == top[segment_ids])
    total = jax.ops.segment_sum(
        jnp.where(winner, data, 0.0),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
    )
    count = jax.ops.segment_sum(
        winner.astype(data.dtype),
        segment_ids,
        num_segments,
        indices_are_sorted=indices_are_sorted,
    )
    return total / jnp.maximum(count, 1)


def segment_softmax(
    logits: Float[Array, "e ..."],
    segment_ids: Int[Array, "e"],
    num_segments: int,
    indices_are_sorted: bool = False,
) -> Float[Array, "e ..."]:
    """Softmax over each segment. Padded entries get the weight zero.

    Follows `jraph.segment_softmax`, with padding handled in an
    additional segment so that it cannot affect the others.

    Args:
        logits: Logits per edge, shaped (e, ...).
        segment_ids: Segment per edge, `num_segments` for padding.
        num_segments: Number of segments.
        indices_are_sorted: Whether `segment_ids` is ascending.

    Returns:
        The weights per edge, shaped (e, ...). They sum to one over the edges of
        a non-empty segment.
    """
    in_dtype = logits.dtype
    logits = logits.astype(jnp.promote_types(in_dtype, jnp.float32))
    valid = _expand_like(segment_ids < num_segments, logits)
    logits = jnp.where(valid, logits, 0.0)
    top = jax.ops.segment_max(
        jax.lax.stop_gradient(logits),
        segment_ids,
        num_segments + 1,
        indices_are_sorted=indices_are_sorted,
    )
    exp = jnp.exp(logits - top[segment_ids])
    total = jax.ops.segment_sum(
        exp, segment_ids, num_segments + 1, indices_are_sorted=indices_are_sorted
    )
    return jnp.where(valid, exp / total[segment_ids], 0.0).astype(in_dtype)
