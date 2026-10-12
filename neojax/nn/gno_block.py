"""Implementation of the Graph Neural Operator block."""

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.integral_transform import IntegralTransform
from neojax.nn.neighbor_search import Neighbors, radius_neighbor_search
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import SinusoidalEmbedding


class GNOBlock(eqx.Module):
    r"""Graph Neural Operator block.

    A resolution-invariant operator that maps a function sampled on one
    set of points $y$ to a function sampled on another set of points $x$.
    The output at $x$ is a kernel integral over those input points
    that are within a ball of fixed `radius` around $x$,
    see `IntegralTransform` for the available transforms.

    The points can be optionally embedded with a `SinusoidalEmbedding`
    before they enter the kernel network.
    The neighbors are always searched with the raw coordinates.

    With a `weighting_fn` the block is a mollified GNO. The hard ball is
    replaced by a smooth weight, so the output is differentiable in the query
    positions `x`. The weights are computed from the raw coordinates inside
    the block, so the derivatives w.r.t. `x` are also correct for
    precomputed `neighbors`.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of channels of the input function.
            Used by the nonlinear transforms (as input of the kernel)
            and to validate the `"linear"` transforms.
        out_channels: Number of channels of the output function.
        coord_dim: Dimension of the domain on which $x$ and $y$ are defined.
        radius: Radius of the neighborhoods.
        max_neighbors: Maximum number of neighbors per output point. Neighbors
            beyond the nearest `max_neighbors` are dropped. Default is 32.
        transform_type: Integral transform to compute. Default is `"linear"`.
        reduction: How to aggregate over a neighborhood, by summing (`"sum"`)
            or averaging (`"mean"`). Default is `"sum"`.
        weighting_fn: Optional mollifier applied to the squared distances
            between the points and their neighbors, see `make_weighting_fn`.
            Forces the `"sum"` reduction. Default is `None`.
        pos_embedding_type: Type of the `SinusoidalEmbedding` of the points,
            `"transformer"` or `"nerf"`. `None` disables the embedding.
            Default is `"transformer"`.
        pos_embedding_channels: Number of frequencies of the
            embedding per coordinate. Default is 32.
        pos_embedding_max_positions: `max_positions` of the `"transformer"`
            embedding. Default is 10000.
        channel_mlp_hidden_layers: Widths of the hidden layers of the kernel
            network. The input and output widths are inferred. Default is
            `(128, 256, 128)`. Not used if `channel_mlp` is passed.
        activation: Activation function of the kernel network.
            Default is `jax.nn.gelu`.
        channel_mlp: Optional custom kernel network. Must only mix channels
            to keep the integral discretization invariant and have matching
            input and output channels (see the `kernel_in_channels` attribute).
            If passed, `channel_mlp_hidden_layers` and `activation` are ignored.
        query_chunk_size: Number of output points to process at a time
            in the neighbor search and the transform to limit peak memory.
            Default is `None`, all points at once.

    Raises:
        ValueError: If `in_channels`, `out_channels`, `coord_dim`,
            `max_neighbors` or `pos_embedding_channels` are not positive integers.
        ValueError: If `radius` is not positive.
        ValueError: If `transform_type` is `"linear"` or `"nonlinear"`
            and `in_channels` differs from `out_channels`.
        ValueError: If `channel_mlp` has the wrong number of input or output channels.

    ??? info "Internal Attributes"
        * **pos_embedding** (`SinusoidalEmbedding | None`): Embedding of the points.
        * **integral_transform** (`IntegralTransform`): The kernel integral transform.
        * **in_channels** (`int`): Number of channels of the input function.
        * **out_channels** (`int`): Number of channels of the output function.
        * **coord_dim** (`int`): Dimension of the domain.
        * **radius** (`float`): Radius of the neighborhoods.
        * **max_neighbors** (`int`): Maximum number of neighbors per output point.
        * **query_chunk_size** (`int | None`): Number of output points per chunk.

    !!! warning "Neighbor truncation"
        Neighbors beyond the `max_neighbors` nearest are dropped silently.
        With a `weighting_fn` a dropped neighbor can have a nonzero weight, which
        reintroduces a discontinuity in `x`. Choose `max_neighbors` large enough
        that truncation does not occur if you rely on derivatives of the output.
    """

    pos_embedding: SinusoidalEmbedding | None
    integral_transform: IntegralTransform
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    max_neighbors: int = eqx.field(static=True)
    query_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        coord_dim: int,
        radius: float,
        max_neighbors: int = 32,
        transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        reduction: Literal["sum", "mean"] = "sum",
        weighting_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]]
        | None = None,
        pos_embedding_type: Literal["transformer", "nerf"] | None = "transformer",
        pos_embedding_channels: int = 32,
        pos_embedding_max_positions: int = 10000,
        channel_mlp_hidden_layers: Sequence[int] = (128, 256, 128),
        activation: Callable = jax.nn.gelu,
        channel_mlp: PointwiseMLP | None = None,
        query_chunk_size: int | None = None,
    ) -> None:
        for name, value in [
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("coord_dim", coord_dim),
            ("max_neighbors", max_neighbors),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not radius > 0:
            raise ValueError("radius must be positive.")
        if transform_type in ["linear", "nonlinear"] and in_channels != out_channels:
            raise ValueError(
                f"Transform '{transform_type}' multiplies the kernel with the input "
                f"function, so in_channels ({in_channels}) must equal "
                f"out_channels ({out_channels})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim
        self.radius = float(radius)
        self.max_neighbors = max_neighbors
        self.query_chunk_size = query_chunk_size

        if pos_embedding_type is None:
            self.pos_embedding = None
            embedded_dim = coord_dim
        else:
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
                max_positions=pos_embedding_max_positions,
            )
            embedded_dim = self.pos_embedding.out_channels

        kernel_in_channels = 2 * embedded_dim
        if transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            kernel_in_channels += in_channels

        if channel_mlp is None:
            channel_mlp = PointwiseMLP(
                key=key,
                layers=(kernel_in_channels, *channel_mlp_hidden_layers, out_channels),
                activations=activation,
            )
        else:
            if channel_mlp.weights[0].shape[1] != kernel_in_channels:
                raise ValueError(
                    f"channel_mlp must take {kernel_in_channels} input channels "
                    f"(embedded y + embedded x"
                    f"{' + f(y)' if kernel_in_channels > 2 * embedded_dim else ''}), "
                    f"but takes {channel_mlp.weights[0].shape[1]}."
                )
            if channel_mlp.weights[-1].shape[0] != out_channels:
                raise ValueError(
                    f"channel_mlp must output {out_channels} channels, "
                    f"but outputs {channel_mlp.weights[-1].shape[0]}."
                )

        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            transform_type=transform_type,
            reduction=reduction,
            weighting_fn=weighting_fn,
            query_chunk_size=query_chunk_size,
        )

    @property
    def kernel_in_channels(self) -> int:
        """Number of input channels of the kernel network."""
        return int(self.integral_transform.channel_mlp.weights[0].shape[1])

    def __call__(
        self,
        y: Float[Array, "n d"],
        x: Float[Array, "m d"],
        f_y: Float[Array, "in_c n"] | None = None,
        *,
        quadrature_weights: Float[Array, "n"] | None = None,
        neighbors: Neighbors | None = None,
    ) -> Float[Array, "out_c m"]:
        """Computes the neighbor search and kernel integral transform.

        Args:
            y: Coordinates of the `n` input points to integrate over.
            x: Coordinates of the `m` output points. Must share the domain with `y`.
            f_y: Input function sampled at `y`. See `IntegralTransform`
                for when it is required.
            quadrature_weights: Optional quadrature weights of the `n` input
                points, see `IntegralTransform`. Default is `None`.
            neighbors: Optional precomputed neighbors in `y` of each point in `x`,
                e.g. from `radius_neighbor_search`. Skips the neighbor search.
                Useful if the geometry is fixed. Default is `None`.

        Returns:
            The output function sampled at `x`.
        """
        if neighbors is None:
            neighbors = radius_neighbor_search(
                y, x, self.radius, self.max_neighbors, self.query_chunk_size
            )

        if self.integral_transform.weighting_fn is not None:
            # Recompute from the coordinates (instead of using the stored distances)
            # such that the weights are differentiable w.r.t. x, also for
            # precomputed neighbors
            sq_dist = jnp.sum((x[:, None, :] - y[neighbors.index]) ** 2, axis=-1)
            neighbors = eqx.tree_at(
                lambda nb: nb.sq_dist,
                neighbors,
                jnp.where(neighbors.mask, sq_dist, 0.0),
            )

        if self.pos_embedding is not None:
            y = self.pos_embedding(y)
            x = self.pos_embedding(x)

        return self.integral_transform(
            y, x, neighbors, f_y, quadrature_weights=quadrature_weights
        )
