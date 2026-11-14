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

"""Implementation of the kernel integral transform of the Graph Neural Operator."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from neojax.nn.neighbor_attention import NeighborAttention
from neojax.nn.neighbor_search import Neighbors
from neojax.nn.pointwise_mlp import PointwiseMLP


class IntegralTransform(eqx.Module):
    r"""Kernel integral transform (Graph Neural Operator layer).

    Approximates the integral of a learned kernel $k$ over the neighborhood
    $A(x)$ of each output point $x$ by a Monte-Carlo sum (or mean)
    over the neighboring input points $y$.
    Depending on `transform_type` it computes one of

    - `"linear_kernelonly"`: $\int_{A(x)} k(x, y) \, dy$,
    - `"linear"`: $\int_{A(x)} k(x, y) \, f(y) \, dy$,
    - `"nonlinear_kernelonly"`: $\int_{A(x)} k(x, y, f(y)) \, dy$,
    - `"nonlinear"`: $\int_{A(x)} k(x, y, f(y)) \, f(y) \, dy$.

    The kernel $k$ is parametrized by a `PointwiseMLP` that acts on the
    concatenated features $(y, x [, f(y)])$ of each pair of points
    and that only mixes channels. The product with $f(y)$ is elementwise.

    The neighborhoods are passed as an edge list, `Neighbors`. The kernel is
    evaluated once per edge and the values are summed per output point. Optional
    per-point quadrature weights $\delta(y)$ turn the sum into a Riemann sum.

    With a `weighting_fn` $w$ the hard ball $A(x) = B_r(x)$ is replaced by a
    smooth weight, i.e. the layer computes $\int w(x, y) \, k(x, y) \, f(y) \, dy$,
    the mollified GNO. This makes the output differentiable in the query
    position $x$.

    With an `attention` the kernel values of the neighbors are reweighed by
    attention weights $\alpha(x, y)$ that are normalized over the neighborhood,
    i.e. the layer computes $\int \alpha(x, y) \, k(x, y) \, f(y) \, dy$,
    the Attentional Graph Neural Operator.

    Args:
        channel_mlp: The `PointwiseMLP` parametrizing the kernel. Its input
            channels must be the sum of the channels of $y$ and $x$ (and of
            $f$ for the nonlinear transforms). For the `"linear"` and
            `"nonlinear"` transforms its output channels must equal the
            channels of $f$.
        transform_type: Integral transform to compute. Default is `"linear"`.
        reduction: How to aggregate over a neighborhood, by summing (`"sum"`)
            or averaging (`"mean"`). Default is `"sum"`. An empty neighborhood
            yields zero for both. Forced to `"sum"` if `weighting_fn`
            or `attention` is given.
        weighting_fn: Optional function mapping the squared distances of the
            edges to weights by which the kernel values are multiplied. This is
            the mollifier of the mollified GNO, see `make_weighting_fn`. The
            squared distances are passed to `__call__`, so that the `GNOBlock`
            can compute them from the coordinates and keep the weights
            differentiable in $x$ even for precomputed neighbors.
            Default is `None`.
        edge_chunk_size: If given, the transform is computed for this many edges
            at a time and rematerialized in the backward pass. The kernel
            `channel_mlp` is evaluated on every edge, which can be prohibitively
            large in memory for wide kernels, so this trades speed for memory.
            Default is `None`.
        attention: Optional `NeighborAttention` computing the attention weights
            $\alpha(x, y)$ of the neighbors from the (embedded) coordinates
            passed to `__call__`. The weights sum to one over the valid neighbors,
            so the mean reduction is replaced by the `"sum"` reduction.
            Default is `None`.

    Raises:
        ValueError: If `transform_type` or `reduction` is invalid.
        ValueError: If `edge_chunk_size` is not a positive integer.

    ??? info "Internal Attributes"
        * **channel_mlp** (`PointwiseMLP`): The kernel network.
        * **transform_type** (`str`): Integral transform type.
        * **reduction** (`str`): Reduction over neighborhoods.
        * **weighting_fn** (`Callable | None`): Weighting function of the squared distances.
        * **edge_chunk_size** (`int | None`): Number of edges per chunk.
        * **attention** (`NeighborAttention | None`): Attention weights of the neighbors.

    !!! warning "Sum or mean with a `weighting_fn`"
        Weights near the boundary of a neighborhood are close to zero, but
        such neighbors still count in a `"mean"`. When the query moves, neighbors
        enter and leave the ball and the mean changes discontinuously. That is why
        a `weighting_fn` forces the `"sum"` reduction.
    """

    channel_mlp: PointwiseMLP
    transform_type: Literal[
        "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
    ] = eqx.field(static=True)
    reduction: Literal["sum", "mean"] = eqx.field(static=True)
    weighting_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]] | None = (
        eqx.field(static=True)
    )
    edge_chunk_size: int | None = eqx.field(static=True)
    attention: NeighborAttention | None

    def __init__(
        self,
        channel_mlp: PointwiseMLP,
        transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        reduction: Literal["sum", "mean"] = "sum",
        weighting_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]]
        | None = None,
        edge_chunk_size: int | None = None,
        attention: NeighborAttention | None = None,
    ) -> None:
        if transform_type not in [
            "linear_kernelonly",
            "linear",
            "nonlinear_kernelonly",
            "nonlinear",
        ]:
            raise ValueError(
                "transform_type must be one of ['linear_kernelonly', 'linear', "
                f"'nonlinear_kernelonly', 'nonlinear'], but got '{transform_type}'."
            )
        if reduction not in ["sum", "mean"]:
            raise ValueError("reduction must be one of ['sum', 'mean'].")
        if edge_chunk_size is not None and (
            not isinstance(edge_chunk_size, int) or edge_chunk_size <= 0
        ):
            raise ValueError("edge_chunk_size must be a positive integer or None.")

        self.channel_mlp = channel_mlp
        self.transform_type = transform_type
        self.reduction = (
            "sum" if weighting_fn is not None or attention is not None else reduction
        )
        self.weighting_fn = weighting_fn
        self.edge_chunk_size = edge_chunk_size
        self.attention = attention

    def __call__(
        self,
        y: Float[Array, "n e_y"],
        x: Float[Array, "m e_x"],
        neighbors: Neighbors,
        f_y: Float[Array, "f_c n"] | None = None,
        quadrature_weights: Float[Array, "n"] | None = None,
        sq_dist: Float[Array, "capacity"] | None = None,
    ) -> Float[Array, "out_c m"]:
        r"""Computes the kernel integral transform.

        Args:
            y: The (embedded) coordinates of the `n` input points
                that are integrated over.
            x: The (embedded) coordinates of the `m` output points.
            neighbors: The edges between the output points and their neighbors
                in `y`. These must be found with the *unembedded* coordinates,
                e.g. by `radius_neighbor_search`.
            f_y: Input function sampled at `y`. Optional for the `"linear"`
                transform, ignored by `"linear_kernelonly"` and required
                by the nonlinear transforms.
            quadrature_weights: Optional quadrature weights $\delta(y)$ of the
                `n` input points, that multiply the kernel values (Riemann sum).
                With a `"sum"` reduction they make the output independent of the
                discretization. Default is `None`, no quadrature weights.
            sq_dist: Squared distances of the edges, required by a `weighting_fn`.

        Returns:
            The output function sampled at `x`.

        Raises:
            ValueError: If `f_y` is required but `None`.
            ValueError: If a `weighting_fn` is set but `sq_dist` is `None`.
            ValueError: If the `"linear"` or `"nonlinear"` transforms
                are used and the kernel output channels do not match the
                channels of `f_y`.
            ValueError: If an `attention` is set and its coordinate
                channels do not match the channels of `x`.
        """
        nonlinear = self.transform_type in ["nonlinear", "nonlinear_kernelonly"]
        if self.transform_type == "linear_kernelonly":
            f_y = None
        if nonlinear and f_y is None:
            raise ValueError(f"f_y is required for transform '{self.transform_type}'.")

        multiply_f = f_y is not None and self.transform_type in [
            "linear",
            "nonlinear",
        ]
        if multiply_f:
            kernel_out_c = self.channel_mlp.weights[-1].shape[0]
            if f_y.shape[0] != kernel_out_c:
                raise ValueError(
                    f"Transform '{self.transform_type}' multiplies the kernel with "
                    f"f_y, so the kernel must output {f_y.shape[0]} channels, "
                    f"but outputs {kernel_out_c}."
                )

        if self.attention is not None and self.attention.in_channels != x.shape[-1]:
            raise ValueError(
                f"The attention takes {self.attention.in_channels} coordinate "
                f"channels, but the coordinates have {x.shape[-1]}."
            )

        if self.weighting_fn is not None and sq_dist is None:
            raise ValueError("sq_dist is required by the weighting_fn.")

        m = x.shape[0]
        query_index = neighbors.query_index
        data_index = neighbors.data_index
        # padding holds the sentinel m, which segment_sum drops
        safe_query = jnp.minimum(query_index, m - 1)

        edge_weights = None
        if self.attention is not None:
            # normalized over all edges of a query before chunking
            edge_weights = self.attention(
                x[safe_query].T, y[data_index].T, neighbors
            )  # (e,)
        if self.weighting_fn is not None:
            mollifier = self.weighting_fn(sq_dist)
            edge_weights = (
                mollifier if edge_weights is None else edge_weights * mollifier
            )

        def transform_edges(
            query: Int[Array, "c"],
            data: Int[Array, "c"],
            weights: Float[Array, "c"] | None,
        ) -> Float[Array, "out_c m"]:
            features = [y[data].T, x[jnp.minimum(query, m - 1)].T]  # (e_y, c), (e_x, c)
            f_nb = None
            if f_y is not None:
                f_nb = f_y[:, data]  # (f_c, c)
                if nonlinear:
                    features.append(f_nb)

            kernel = self.channel_mlp(jnp.concatenate(features, axis=0))
            if multiply_f:
                kernel = kernel * f_nb
            if weights is not None:
                kernel = kernel * weights[None]
            if quadrature_weights is not None:
                kernel = kernel * quadrature_weights[data][None]
            out = jax.ops.segment_sum(kernel.T, query, m, indices_are_sorted=True)
            return out.T

        chunk = self.edge_chunk_size
        capacity = query_index.shape[0]
        if chunk is None or chunk >= capacity:
            out = transform_edges(query_index, data_index, edge_weights)
        else:
            # round up, not down
            n_chunks = -(-capacity // chunk)
            pad = n_chunks * chunk - capacity

            def to_chunks(a: Array, fill: int | float) -> Array:
                return jnp.pad(a, (0, pad), constant_values=fill).reshape(
                    n_chunks, chunk
                )

            chunks = (
                to_chunks(query_index, m),
                to_chunks(data_index, 0),
                None if edge_weights is None else to_chunks(edge_weights, 0.0),
            )
            body = jax.checkpoint(transform_edges)

            def step(total: Array, args: tuple) -> tuple[Array, None]:
                return total + body(*args), None

            first = jax.tree.map(lambda a: a[0], chunks)
            shape = jax.eval_shape(body, *first)
            init = jnp.zeros(shape.shape, shape.dtype)
            out, _ = jax.lax.scan(step, init, chunks)

        if self.reduction == "mean":
            out = out / jnp.maximum(neighbors.neighbor_counts, 1)
        return out
