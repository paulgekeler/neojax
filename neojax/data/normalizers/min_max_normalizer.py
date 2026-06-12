"""Implementation of a min/max normalizer."""

import warnings
from typing import Literal, final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.data.normalizers.base_normalizer import BaseNormalizer


@final
class MinMaxNormalizer(BaseNormalizer):
    """Normalizes data to be within a given min and max.

    Minima and maxima can be passed the following ways:

    1. During initialization if known or explicitly required
        (e.g. known physical bounds).
    2. Learned from data by passing them to `compute_stats`

    Data can be clipped or scaled to [min, max] range.

    Args:
        minima: Optional min for dynamical scaling.
                Default is None, i.e., use learned min
                from `self.stats`.
                If Array, should be broadcast-compatible
                with transformation data and `maxima`.
        maxima: Optional max for dynamical scaling.
                Default is None, i.e., use learned max
                from `self.stats`.
                If Array, should be broadcast-compatible
                with transformation data and `minima`.

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **stats** (`dict[str, Float[Array, ...] | None]`): Dict of min and max if passed at initialization, else None.

    !!! warning
        Inverse transformation can only be applied for scaled transformations.
    """

    def __init__(
        self,
        minima: float | int | Float[Array, "..."] | None = None,
        maxima: float | int | Float[Array, "..."] | None = None,
    ) -> None:
        self.stats = {
            "min": jnp.asarray(minima) if minima is not None else None,
            "max": jnp.asarray(maxima) if maxima is not None else None,
        }

    @property
    def minima(self) -> Float[Array, "..."] | None:
        """Accessor for the min stored in stats."""
        return self.stats["min"]

    @property
    def maxima(self) -> Float[Array, "..."] | None:
        """Accessor for the max stored in stats."""
        return self.stats["max"]

    def compute_stats(
        self,
        data: Float[Array, "c ..."],
        axis: int | tuple[int, ...] | None = None,
    ) -> "MinMaxNormalizer":
        """Computes min and max from given data.

        To normalize per-channel for input (c, d1, ..., dN),
        pass axis=tuple(range(1, data.ndim)). To normalize
        across all axes, pass axis=None.

        Args:
            data: The input array.
            axis: Axis or axes along which the statistics are computed.
                The default (None) computes the global min/max.

        Returns:
            A new MinMaxNormalizer instance
            with updated min and max.
        """
        if self.stats["min"] is not None or self.stats["max"] is not None:
            warnings.warn(
                "Overwriting existing min/max from stats.",
                type=UserWarning,
                stacklevel=1,
            )
        # We keep dims to ensure broadcasting works
        minima = jnp.min(data, axis=axis, keepdims=True)
        maxima = jnp.max(data, axis=axis, keepdims=True)
        new_stats = {"min": minima, "max": maxima}

        return eqx.tree_at(lambda n: n.stats, self, new_stats)

    def transform(
        self,
        x: Float[Array, "..."],
        mode: Literal["clip", "scale"] = "scale",
    ) -> Float[Array, "..."]:
        """Normalizes input by clipping or rescaling it.

        Uses learned minima or maxima from `self.stats`.

        Args:
            x: The input array to transform.
            mode: Whether to `"clip"` values to be in
                [`minima`, `maxima`] (inclusive)
                or `"scale"` values inside range with standard
                min/max scaling (x - min) / (max - min + eps).
                Default is `"scale"`.

        Note:
            `"clip"` is a destructive operation. Clipped data cannot be
            perfectly reconstructed:
            ```python
                inverse_transform(transform(x, mode="clip", ...), ...) != x
            ```

        Returns:
            The min/max normalized array.

        Raises:
            ValueError: If the transformation mode is unsupported.
        """
        if mode == "scale":
            return (x - self.stats["min"]) / (
                self.stats["max"] - self.stats["min"] + 1e-7
            )
        elif mode == "clip":
            warnings.warn(
                "'clip' is a destructive operation."
                " "
                "The inverse transform does not exist.",
                stacklevel=1,
            )
            return jnp.clip(x, self.stats["min"], self.stats["max"])
        raise ValueError(f"Unknown mode {mode}. Use either 'clip' or 'scale'.")

    def inverse_transform(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Reverts scaled min/max normalization.

        Can only reverse scaled normalization.
        Clipped normalizations are irreversible.

        Args:
            x: The standardized array to revert.

        Returns:
            The de-normalized array.
        """
        return (self.stats["max"] - self.stats["min"] + 1e-7) * x + self.stats["min"]

    def __call__(
        self,
        x: Float[Array, "..."],
        mode: Literal["clip", "scale"] = "scale",
    ) -> Float[Array, "..."]:
        """Standardizes input using learned min and max.

        See `transform` for details.

        Args:
            x: Input array.
            mode: Whether to `"clip"` values to be in
                [`minima`, `maxima`] (inclusive)
                or `"scale"` values inside range with standard
                min/max scaling (x - min) / (max - min + eps).
                Default `"scale"`.

        Returns:
            Standardized input array.
        """
        return self.transform(x, mode)
