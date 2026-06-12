"""Implementation of a robust normalizer."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.data.normalizers.base_normalizer import BaseNormalizer


@final
class RobustNormalizer(BaseNormalizer):
    """Normalizes data using median and interquartile range.

    This normalizer is robust to outliers as it uses the median
    and the interquartile range (IQR).

    Args:
        median: Optional median for dynamical scaling.
            Default is None, i.e., use learned median from `self.stats`.
        scale: Optional interquartile range for dynamical scaling.
            Default is None, i.e., use learned iqr from `self.stats`.
        quantile_range:

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **quantile_range** (`tuple[float, float]`): The lower and upper quantile bounds.
        * **stats** (`dict[str, Float[Array, ...] | None]`): Dict containing 'median' and 'scale' (iqr).
    """

    quantile_range: tuple[float, float] = eqx.field(static=True)

    def __init__(
        self,
        median: float | Float[Array, "..."] | None = None,
        scale: float | Float[Array, "..."] | None = None,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ) -> None:
        self.quantile_range = quantile_range
        self.stats = {
            "median": jnp.asarray(median) if median is not None else None,
            "scale": jnp.asarray(scale) if scale is not None else None,
        }

    @property
    def median(self) -> Float[Array, "..."] | None:
        """Accessor for the median stored in stats."""
        return self.stats["median"]

    @property
    def scale(self) -> Float[Array, "..."] | None:
        """Accessor for the scale stored in stats."""
        return self.stats["scale"]

    def compute_stats(
        self,
        data: Float[Array, "c ..."],
        axis: int | tuple[int, ...] | None = None,
    ) -> "RobustNormalizer":
        """Computes median and quantile range from given data.

        Args:
            data: The input array to compute statistics from.
            axis: Axis or axes along which the statistics are computed.
                The default (None) computes the global statistics.

        Returns:
            A new RobustNormalizer instance with updated statistics.
        """
        low, high = self.quantile_range
        q = jnp.array([low, 50.0, high]) / 100.0

        quant = jnp.quantile(data, q, axis=axis, keepdims=True)
        new_stats = {
            "median": quant[1],
            "scale": quant[2] - quant[0],
        }

        return eqx.tree_at(lambda n: n.stats, self, new_stats)

    def transform(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Normalizes input using median and quantile range.

        Args:
            x: The input array to transform.

        Returns:
            The normalized array.
        """
        return (x - self.median) / (self.scale + 1e-7)

    def inverse_transform(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Reverts robust normalization.

        Args:
            x: The transformed array to revert.

        Returns:
            The de-normalized array.
        """
        return (self.scale + 1e-7) * x + self.median

    def __call__(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Standardizes input using median and quantile range.

        Args:
            x: Input array.

        Returns:
            Normalied input array.
        """
        return self.transform(x)
