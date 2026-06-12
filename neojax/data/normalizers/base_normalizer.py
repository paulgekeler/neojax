"""Implementation of the abstract normalizer class."""

import abc

import equinox as eqx
from jaxtyping import Array, Float, PyTree


class BaseNormalizer(eqx.Module):
    """Base class for all normalizers.

    All normalizers should inherit from BaseNormalizer.
    Following the "Abstract or Final" pattern, this class contains no
    logic and only defines the interface. Subclasses should be marked
    as final.

    Normalizer methods follow the naming conventions of major
    libraries, where `transform` is the forward operation
    and `inverse_transform` the inverse.

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **stats** (`PyTree`): A PyTree containing the normalization statistics (e.g. mean, std).
    """

    stats: PyTree

    @abc.abstractmethod
    def compute_stats(self, data: Float[Array, "..."]) -> "BaseNormalizer":
        """Computes statistics from the provided data.

        This function should set the stats attribute.

        Args:
            data: The data to compute statistics from.

        Returns:
            A new BaseNormalizer instance with updated stats.
        """
        ...

    @abc.abstractmethod
    def transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Applies the forward normalization transform.

        Args:
            x: The input array to transform.

        Returns:
            The transformed array.
        """
        ...

    @abc.abstractmethod
    def inverse_transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Applies the inverse normalization transform.

        Args:
            x: The transformed array to revert.

        Returns:
            The array in its original scale
        """
        ...
