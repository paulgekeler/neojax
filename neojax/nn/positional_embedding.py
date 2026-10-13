"""Implementation of general positional embeddings."""

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact


class GridEmbeddingNd(eqx.Module):
    """Regular grid embedding for n-dim signals.

    Args:
        in_channels: Number of channels of the input signal.
            The embedding appends one coordinate channel per spatial dimension.
        ndim: Number of spatial dimensions.
        grid_boundaries: Boundaries of regular grid per dimension
            ((low, high), ...) or None.
            Default is None which is a ((0, 1), ...) bounded grid.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **in_channels** (`int`): Number of channels of the input signal.
        * **grid_boundaries** (`tuple[tuple[int, int], ...]`): Boundaries of regular grid.
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

    @property
    def n_extra_channels(self) -> int:
        """Number of channels appended to the input signal."""
        return len(self.grid_boundaries)

    def _create_grid_nd(self, resolutions: tuple[int, ...]) -> Float[Array, "c ..."]:
        """Creates a n-dim bounded regular grid.

        Args:
            resolutions: Dimensions (d1, ..., dN) without channel axis.
                Resolution is dimension size on a bounded interval.

        Returns:
            N-dim bounded regular grid (c, d1, ..., dN).

        Raises:
            ValueError: If the number of resolutions
                and grid boundaries don't match.
        """
        if len(resolutions) != len(self.grid_boundaries):
            raise ValueError(
                f"Mismatch of num resolutions {len(resolutions)} "
                f"and num of grid boundaries {len(self.grid_boundaries)}."
            )
        grid_points_1d = []
        for res, (start, end) in zip(resolutions, self.grid_boundaries, strict=True):
            grid_points_1d.append(jnp.linspace(start, end, res))

        return jnp.stack(jnp.meshgrid(*grid_points_1d, indexing="ij"), axis=0)

    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Generates n-dim regular grid and appends it to input signal.

        Args:
            x: Input signal, shaped (channels, d1, ..., dN).

        Returns:
            Concatenated signal and grid array (along channel dim),
            shaped (in_channels + ndim, d1, ..., dN).

        Raises:
            ValueError: If the number of input channels does not match `in_channels`.
        """
        if x.shape[0] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels along axis 0, "
                f"but got shape {x.shape}."
            )
        grid = self._create_grid_nd(x.shape[1:])
        return jnp.concat([x, grid], axis=0)


class SinusoidalEmbedding(eqx.Module):
    r"""Sinusoidal embedding of point coordinates.

    Lifts low-dimensional coordinates to a richer spectral representation
    by encoding each coordinate as sines and cosines at multiple frequencies.
    This helps coordinate-based networks, like the kernels of `GNOBlock`,
    to resolve fine-scale variations.

    For coordinate $p$ and frequency level $k \in \{0, \dots, L-1\}$ the
    embedding contains the pair $(\sin(\omega_k p), \cos(\omega_k p))$ with

    - `"transformer"`: $\omega_k = M^{-2k/L}$ with $M$ = `max_positions`,
    - `"nerf"`: $\omega_k = 2^k \pi$.

    The pairs are laid out per coordinate and then per frequency,
    so the embedding of `d` coordinates has `2 * L * d` channels.

    Args:
        in_channels: Number of coordinates per point, i.e. the spatial dimension.
        num_frequencies: Number of frequency levels $L$.
        embedding_type: Type of embedding, `"transformer"` or `"nerf"`.
            Default is `"transformer"`.
        max_positions: Maximum number of positions $M$ of the `"transformer"`
            embedding. Not used for `"nerf"`. Default is 10000.

    Raises:
        ValueError: If `in_channels` or `num_frequencies` are not positive integers.
        ValueError: If `embedding_type` is invalid.
        ValueError: If `max_positions` is not a positive number.

    ??? info "Internal Attributes"
        * **in_channels** (`int`): Number of coordinates per point.
        * **num_frequencies** (`int`): Number of frequency levels.
        * **embedding_type** (`Literal["transformer", "nerf"]`): Type of embedding.
        * **max_positions** (`int`): Maximum number of positions
            of the `"transformer"` embedding.

    !!! info "Aliasing"
        Make sure the highest frequency resolves the discretization.
        E.g. for `"nerf"`, $2^{L-1} < N/2$ for $N$ sample points.

    ??? cite

        [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

        ```bibtex
        @inproceedings{vaswani2017attention,
            title={Attention is all you need},
            author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
                Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
                Kaiser, {\L}ukasz and Polosukhin, Illia},
            booktitle={Advances in Neural Information Processing Systems},
            year={2017}
        }
        ```

        [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](
        https://arxiv.org/abs/2003.08934)

        ```bibtex
        @inproceedings{mildenhall2020nerf,
            title={NeRF: Representing Scenes as Neural Radiance Fields
                for View Synthesis},
            author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew
                and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
            booktitle={European Conference on Computer Vision},
            year={2020}
        }
        ```
    """

    in_channels: int = eqx.field(static=True)
    num_frequencies: int = eqx.field(static=True)
    embedding_type: Literal["transformer", "nerf"] = eqx.field(static=True)
    max_positions: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        num_frequencies: int,
        embedding_type: Literal["transformer", "nerf"] = "transformer",
        max_positions: int = 10000,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(num_frequencies, int) or num_frequencies <= 0:
            raise ValueError("num_frequencies must be a positive integer.")
        if embedding_type not in ["transformer", "nerf"]:
            raise ValueError("embedding_type must be one of ['transformer', 'nerf'].")
        if not max_positions > 0:
            raise ValueError("max_positions must be positive.")
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.embedding_type = embedding_type
        self.max_positions = max_positions

    @property
    def out_channels(self) -> int:
        """Number of channels of the embedding."""
        return 2 * self.num_frequencies * self.in_channels

    def __call__(self, x: Float[Array, "... in_c"]) -> Float[Array, "... out_c"]:
        """Embeds the coordinates.

        Note that the coordinates are along the *last* axis,
        which is the convention for point coordinates (as opposed to
        channel-first signals).

        Args:
            x: Point coordinates shaped (..., in_channels),
                e.g. (n_points, in_channels).

        Returns:
            The embedded coordinates shaped (..., 2 * num_frequencies * in_channels).

        Raises:
            ValueError: If the size of the last axis
                does not match `in_channels`.
        """
        if x.ndim < 1 or x.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected coordinates with {self.in_channels} channels along "
                f"the last axis, but got shape {x.shape}."
            )

        levels = jnp.arange(self.num_frequencies)
        if self.embedding_type == "nerf":
            freqs = 2.0**levels * jnp.pi
        else:
            freqs = (1.0 / self.max_positions) ** (2.0 * levels / self.num_frequencies)

        angles = x[..., None] * freqs  # (..., in_c, L)
        emb = jnp.stack(
            [jnp.sin(angles), jnp.cos(angles)], axis=-1
        )  # (..., in_c, L, 2)
        return emb.reshape(*x.shape[:-1], self.out_channels)


class ReferenceGridEmbedding(eqx.Module):
    r"""Reference-grid distance embedding for n-dim coordinates.

    Embeds every point $x \in \mathbb{R}^d$ by its Euclidean distances to the
    nodes $\{r_j\}_{j=1}^{\text{ref}^d}$ of a small, fixed, regular reference grid
    (a.k.a. "unified position" embedding in Transolver), i.e.
    $$
    e_j(x) = \lVert x - r_j \rVert_2, \quad j = 1, \dots, \text{ref}^d.
    $$
    The embedding is well-defined for structured grids and unstructured meshes alike.

    Args:
        ndim: Number of spatial dimensions (coordinate dimension).
        ref: Number of reference grid nodes per dimension.
            The embedding has `ref ** ndim` features. Default is 8.
        grid_boundaries: Boundaries of the reference grid per dimension
            ((low, high), ...) or None.
            Default is None which is a ((0, 1), ...) bounded grid.

    Raises:
        ValueError: If `ndim` or `ref` is not positive.
        ValueError: If the number of grid boundaries doesn't match `ndim`.

    ??? info "Internal Attributes"
        * **ndim** (`int`): Number of spatial dimensions.
        * **ref** (`int`): Number of reference grid nodes per dimension.
        * **grid_boundaries** (`tuple[tuple[int | float, int | float], ...]`): Boundaries of the reference grid.
    """

    ndim: int = eqx.field(static=True)
    ref: int = eqx.field(static=True)
    grid_boundaries: tuple[tuple[int | float, int | float], ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        ndim: int,
        ref: int = 8,
        grid_boundaries: tuple[tuple[int | float, int | float], ...] | None = None,
    ) -> None:
        if ndim <= 0:
            raise ValueError("`ndim` must be a positive integer.")
        if ref <= 0:
            raise ValueError("`ref` must be a positive integer.")
        if grid_boundaries is None:
            grid_boundaries = tuple([(0.0, 1.0)] * ndim)
        elif len(grid_boundaries) != ndim:
            raise ValueError(
                f"Mismatch of num grid boundaries {len(grid_boundaries)} "
                f"and ndim {ndim}."
            )
        self.ndim = ndim
        self.ref = ref
        self.grid_boundaries = tuple(grid_boundaries)

    @property
    def n_features(self) -> int:
        """Number of embedding features per point (`ref ** ndim`)."""
        return self.ref**self.ndim

    def regular_grid(self, resolutions: tuple[int, ...]) -> Float[Array, "d ..."]:
        """Creates a n-dim regular grid of coordinates on the grid boundaries.

        Args:
            resolutions: Dimensions (d1, ..., dN) without coordinate axis.

        Returns:
            Regular grid of coordinates (d, d1, ..., dN).

        Raises:
            ValueError: If the number of resolutions doesn't match `ndim`.
        """
        if len(resolutions) != self.ndim:
            raise ValueError(
                f"Mismatch of num resolutions {len(resolutions)} and ndim {self.ndim}."
            )
        axes = [
            jnp.linspace(start, end, res)
            for res, (start, end) in zip(resolutions, self.grid_boundaries, strict=True)
        ]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=0)

    def __call__(self, coords: Float[Array, "d ..."]) -> Float[Array, "g ..."]:
        """Embeds coordinates by their distances to the reference grid nodes.

        Args:
            coords: Coordinates shaped (d, d1, ..., dN) or (d, N).

        Returns:
            Distance features shaped (ref ** d, d1, ..., dN) or (ref ** d, N).

        Raises:
            ValueError: If the size of the leading axis doesn't match `ndim`.
        """
        if coords.shape[0] != self.ndim:
            raise ValueError(
                f"Expected coordinates with {self.ndim} components along axis 0, "
                f"but got shape {coords.shape}."
            )
        axes = [
            jnp.linspace(start, end, self.ref) for start, end in self.grid_boundaries
        ]
        nodes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, self.ndim
        )  # (g, d)
        points = coords.reshape(self.ndim, -1).T  # (n, d)
        diff = points[None, :, :] - nodes[:, None, :]  # (g, n, d)
        dist = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))  # (g, n)
        return dist.reshape(nodes.shape[0], *coords.shape[1:])
