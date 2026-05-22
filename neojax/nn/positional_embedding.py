"""Implementation of general positional embeddings."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class GridEmbeddingNd(eqx.Module):
    """Regular grid embedding for n-dim signals.

    Args:
        in_channels: Number of input channels.
        ndim: Number of spatial dimensions.
        grid_boundaries: Boundaries of regular grid per dimension
            ((low, high), ...) or None.
            Default is None which is a ((0, 1), ...) bounded grid.

    Attributes:
        in_channels: Number of input channels.
        grid_boundaries: Boundaries of regular grid.
    """

    in_channels: int = eqx.field(static=True)
    grid_boundaries: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        ndim: int,
        grid_boundaries: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        self.in_channels = in_channels
        if grid_boundaries is None:
            self.grid_boundaries = tuple([(0, 1)] * ndim)
        else:
            self.grid_boundaries = grid_boundaries

    def _create_grid_nd(self, resolutions: tuple[int, ...]) -> Float[Array, "c ..."]:
        """Creates a n-dim bounded regular grid.

        Args:
            resolutions: Dimensions (d1, ..., dN) without channel axis.
                Resolution is dimension size on a bounded interval.

        Returns:
            N-dim bounded regular grid (c, d1, ..., dN).
        """
        if len(resolutions) != len(self.grid_boundaries):
            raise ValueError(
                f"Mismatch of num resolutions {len(resolutions)} "
                f"and num of grid boundaries {len(self.grid_boundaries)}."
            )
        grid_points_1d = []
        for res, (start, end) in zip(resolutions, self.grid_boundaries, strict=True):
            grid_points_1d.append(jnp.linspace(start, end, res))

        grid = jnp.stack(jnp.meshgrid(*grid_points_1d, indexing="ij"), axis=0)
        return jnp.repeat(grid, self.in_channels, 0)

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Generates n-dim regular grid and appends it to input signal.

        Args:
            x: Input signal, shaped (channels, d1, ..., dN).

        Returns:
            Concatenated signal and grid array (along channel dim).
        """
        grid = self._create_grid_nd(x.shape[1:])
        return jnp.concat([x, grid], axis=0)
