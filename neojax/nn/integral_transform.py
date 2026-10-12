"""Implementation of the kernel integral transform of the Graph Neural Operator."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Float

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

    The neighborhoods are passed as padded `Neighbors`. Optional per-point
    quadrature weights $\delta(y)$ turn the sum into a Riemann sum.

    With a `weighting_fn` $w$ the hard ball $A(x) = B_r(x)$ is replaced by a
    smooth weight, i.e. the layer computes $\int w(x, y) \, k(x, y) \, f(y) \, dy$,
    the mollified GNO. This makes the output differentiable in the query
    position $x$.

    Args:
        channel_mlp: The `PointwiseMLP` parametrizing the kernel. Its input
            channels must be the sum of the channels of $y$ and $x$ (and of
            $f$ for the nonlinear transforms). For the `"linear"` and
            `"nonlinear"` transforms its output channels must equal the
            channels of $f$.
        transform_type: Integral transform to compute. Default is `"linear"`.
        reduction: How to aggregate over a neighborhood, by summing (`"sum"`)
            or averaging (`"mean"`). Default is `"sum"`. An empty neighborhood
            yields zero for both. Forced to `"sum"` if `weighting_fn` is given.
        weighting_fn: Optional function mapping the squared distances between
            $x$ and its neighbors to weights by which the kernel values are
            multiplied. This is the mollifier of the mollified GNO,
            see `make_weighting_fn`. The squared distances are recomputed by
            the `GNOBlock` from the coordinates, so the weights stay
            differentiable in $x$ even for precomputed neighbors.
            Default is `None`.
        query_chunk_size: If given, the transform is computed for this many
            output points at a time and rematerialized in the backward pass.
            The kernel `channel_mlp` is evaluated on (num pairs) = `m * k` pairs,
            which can be prohibitively large in memory for wide kernels,
            so this trades speed for memory. Default is `None`.

    Raises:
        ValueError: If `transform_type` or `reduction` is invalid.
        ValueError: If `query_chunk_size` is not a positive integer.

    ??? info "Internal Attributes"
        * **channel_mlp** (`PointwiseMLP`): The kernel network.
        * **transform_type** (`str`): Integral transform type.
        * **reduction** (`str`): Reduction over neighborhoods.
        * **weighting_fn** (`Callable | None`): Weighting function of the squared distances.
        * **query_chunk_size** (`int | None`): Number of output points per chunk.

    !!! warning "Sum or mean with a `weighting_fn`"
        Weights near the boundary of a neighborhood are close to zero, but
        such neighbors still count in a `"mean"`. When the query moves, neighbors
        enter and leave the ball and the mean changes discontinuously. That is why
        a `weighting_fn` forces the `"sum"` reduction.

    !!! warning "Neighbor truncation"
        Neighbors beyond the `max_neighbors` nearest are dropped, even if
        they have a nonzero weight. This reintroduces a discontinuity,
        so choose `max_neighbors` large enough that truncation never occurs
        when using a `weighting_fn` for derivatives.
    """

    channel_mlp: PointwiseMLP
    transform_type: Literal[
        "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
    ] = eqx.field(static=True)
    reduction: Literal["sum", "mean"] = eqx.field(static=True)
    weighting_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]] | None = (
        eqx.field(static=True)
    )
    query_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        channel_mlp: PointwiseMLP,
        transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        reduction: Literal["sum", "mean"] = "sum",
        weighting_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]]
        | None = None,
        query_chunk_size: int | None = None,
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
        if query_chunk_size is not None and (
            not isinstance(query_chunk_size, int) or query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be a positive integer or None.")

        self.channel_mlp = channel_mlp
        self.transform_type = transform_type
        self.reduction = "sum" if weighting_fn is not None else reduction
        self.weighting_fn = weighting_fn
        self.query_chunk_size = query_chunk_size

    def __call__(
        self,
        y: Float[Array, "n e_y"],
        x: Float[Array, "m e_x"],
        neighbors: Neighbors,
        f_y: Float[Array, "f_c n"] | None = None,
        quadrature_weights: Float[Array, "n"] | None = None,
    ) -> Float[Array, "out_c m"]:
        r"""Computes the kernel integral transform.

        Args:
            y: The (embedded) coordinates of the `n` input points
                that are integrated over.
            x: The (embedded) coordinates of the `m` output points.
            neighbors: The neighbors in `y` of each output point,
                shaped (m, k). These must be found with the *unembedded*
                coordinates, e.g. by `radius_neighbor_search`.
            f_y: Input function sampled at `y`. Optional for the `"linear"`
                transform, ignored by `"linear_kernelonly"` and required
                by the nonlinear transforms.
            quadrature_weights: Optional quadrature weights $\delta(y)$ of the
                `n` input points, that multiply the kernel values (Riemann sum).
                With a `"sum"` reduction they make the output independent of the
                discretization. Default is `None`, no quadrature weights.

        Returns:
            The output function sampled at `x`.

        Raises:
            ValueError: If `f_y` is required but `None`.
            ValueError: If the `"linear"` or `"nonlinear"` transforms
                are used and the kernel output channels do not match the
                channels of `f_y`.
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

        def transform_chunk(
            x_chunk: Float[Array, "c e_x"], nb: Neighbors
        ) -> Float[Array, "out_c c"]:
            y_nb = jnp.moveaxis(y[nb.index], -1, 0)  # (e_y, c, k)
            x_rep = jnp.broadcast_to(
                x_chunk.T[:, :, None], (x_chunk.shape[1], *nb.index.shape)
            )  # (e_x, c, k)
            features = [y_nb, x_rep]
            f_nb = None
            if f_y is not None:
                f_nb = f_y[:, nb.index]  # (f_c, c, k)
                if nonlinear:
                    features.append(f_nb)

            kernel = self.channel_mlp(jnp.concatenate(features, axis=0))
            if multiply_f:
                kernel = kernel * f_nb
            if quadrature_weights is not None:
                kernel = kernel * quadrature_weights[nb.index][None]
            if self.weighting_fn is not None:
                kernel = kernel * self.weighting_fn(nb.sq_dist)[None]

            kernel = jnp.where(nb.mask[None], kernel, 0.0)
            out = jnp.sum(kernel, axis=-1)
            if self.reduction == "mean":
                out = out / jnp.maximum(jnp.sum(nb.mask, axis=-1), 1)
            return out

        m = x.shape[0]
        chunk = self.query_chunk_size
        if chunk is None or chunk >= m:
            return transform_chunk(x, neighbors)

        n_chunks = -(-m // chunk)
        pad = n_chunks * chunk - m

        def to_chunks(a: Array) -> Array:
            padded = jnp.pad(a, ((0, pad),) + ((0, 0),) * (a.ndim - 1))
            return padded.reshape(n_chunks, chunk, *a.shape[1:])

        out = jax.lax.map(
            jax.checkpoint(lambda args: transform_chunk(*args)),
            (to_chunks(x), jtu.tree_map(to_chunks, neighbors)),
        )  # (n_chunks, out_c, chunk)
        return jnp.moveaxis(out, 0, 1).reshape(out.shape[1], -1)[:, :m]
