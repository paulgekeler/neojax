"""Implementation of a unit normalizer."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.data.normalizers.base_normalizer import BaseNormalizer


@final
class UnitGaussianNormalizer(BaseNormalizer):
    """Normalizes data to be mean zero and unit std.

    Args:
        mean: Optional initial mean if known. Default is 0.0.
        std: Optional initial standard deviation if known.
            Default is unit 1.0.

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **stats** (`dict[str, Float[Array, ...] | None]`): Dict of mean and std.
    """

    def __init__(
        self,
        mean: float | Float[Array, "..."] = 0.0,
        std: float | Float[Array, "..."] = 1.0,
    ) -> None:
        self.stats = {"mean": jnp.asarray(mean), "std": jnp.asarray(std)}

    @property
    def mean(self) -> Float[Array, "..."]:
        """Accessor for the mean stored in stats."""
        return self.stats["mean"]

    @property
    def std(self) -> Float[Array, "..."]:
        """Accessor for the std stored in stats."""
        return self.stats["std"]

    def compute_stats(
        self,
        data: Float[Array, "c ..."],
        axis: int | tuple[int, ...] | None = None,
    ) -> "UnitGaussianNormalizer":
        """Computes mean and std from diven data.

        To normalize per-channel for input (c, d1, ..., dN),
        pass axis=tuple(range(1, data.ndim)). To normalize
        across all axes, pass axis=None.

        Args:
            data: The input array.
            axis: Axis or axes along which the statistics are computed.
                The default (None) computes the global mean/std.

        Returns:
            A new UnitGaussianNormalizer instance
            with updated mean and std.
        """
        # We keep dims to ensure broadcasting works
        mean = jnp.mean(data, axis=axis, keepdims=True)
        std = jnp.std(data, axis=axis, keepdims=True)
        new_stats = {"mean": mean, "std": std}

        return eqx.tree_at(lambda n: n.stats, self, new_stats)

    def transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Standardizes input using stored mean and std.

        Args:
            x: The input array to transform.

        Returns:
            The standardized array.
        """
        return (x - self.mean) / (self.std + 1e-7)

    def inverse_transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Reverts standardization.

        Args:
            x: The standardized array to revert.

        Returns:
            The de-standardized array.
        """
        return self.std * x + self.mean

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Standardizes input using stored mean and std.

        Args:
            x: Input array.

        Returns:
            Standardized input array.
        """
        return self.transform(x)
