"""Implementation of a physics normalizer."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.data.normalizers.base_normalizer import BaseNormalizer
from neojax.data.scales import PhysicalScale


@final
class PhysicsNormalizer(BaseNormalizer):
    """Normalizes data based on spatial dimensions/physical units.

    This normalizer applies a product of multiple `PhysicalScale` factors
    to non-dimensionalize the input data.

    Args:
        *scales: Variable number of `PhysicalScale` providers to be
            applied to the data.

    !!! info "Internal Attributes"
        These fields store the internal state of the normalizer.

        * **scales** (`tuple[PhysicalScale, ...]`): Tuple of physical scale providers.
        * **stats** (`dict[str, Float[Array, ...] | None]`): Dict containing 'scale_product', which is the resulting element-wise product of all computed scales.

    Examples:
        ```python
        from neojax.data.scales import CharacteristicLengthScale, ReynoldsScale

        # Compose multiple physical scales
        norm = PhysicsNormalizer(
            CharacteristicLengthScale(L_ref=1.0),
            ReynoldsScale(U=10.0, L=1.0, nu=1e-3)
        )

        # Compute and apply
        norm = norm.compute_stats(data)
        non_dimensional = norm(data)
        ```
    """

    scales: tuple[PhysicalScale, ...]

    def __init__(self, *scales: PhysicalScale) -> None:
        self.scales = scales
        self.stats = {"scale_product": None}

    def compute_stats(
        self,
        data: Float[Array, "c ..."],
        axis: int | tuple[int, ...] | None = None,
    ) -> "PhysicsNormalizer":
        """Computes the product of all scales for the given data.

        Args:
            data: The input array to compute statistics from.
            axis: Unused for physics-based scaling but kept for
                API compatibility.

        Returns:
            A new PhysicsNormalizer instance with the final computed
            scale product in `stats`.
        """
        # Calculate product of all scales
        scale_product = jnp.ones_like(data)
        for scale_provider in self.scales:
            scale_product = scale_product * scale_provider.get_scale(data)

        new_stats = {"scale_product": scale_product}
        return eqx.tree_at(lambda n: n.stats, self, new_stats)

    def transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Non-dimensionalizes input by dividing by the scale product.

        Args:
            x: The input array to transform.

        Returns:
            The non-dimensionalized array.
        """
        return x / (self.stats["scale_product"] + 1e-7)

    def inverse_transform(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Reverts non-dimensionalization by multiplying by the scale product.

        Args:
            x: The transformed array to revert.

        Returns:
            The array in physical units.
        """
        return x * self.stats["scale_product"]

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Standardizes input using the computed physical scale product.

        Args:
            x: Input array.

        Returns:
            Normalized input array.
        """
        return self.transform(x)
