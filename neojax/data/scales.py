"""Implementation of physical scales."""

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


@runtime_checkable
class PhysicalScale(Protocol):
    """Protocol for physical scaling logic.

    Physical scales are used to non-dimensionalize inputs or outputs
    based on domain-specific physics
    (e.g., fluid dynamics, solid mechanics).

    !!! info
        To implement a custom scale, create a class
        (an `equinox.Module`) that implements the `get_scale` method.
        This method should return an array
        compatible with the input shape `x`. The resulting scale object
        can then be passed to a `PhysicsNormalizer`.
    """

    def get_scale(self, x: Float[Array, "..."], **kwargs: Any) -> Array:
        """Returns the scaling factor(s) for the given input.

        Args:
            x: The input array to be scaled.
            **kwargs: Additional parameters for the scaling logic.

        Returns:
            The scaling array, which should be broadcast-compatible with `x`.
        """
        ...


class CharacteristicLengthScale(eqx.Module):
    """Scales data using a constant reference length.

    Commonly used for simple non-dimensionalization
    where a single spatial constant
    (like the chord of an airfoil or the diameter of a pipe)
    governs the system.

    Args:
        L_ref: The constant reference length scale.

    !!! info "Internal Attributes"
        These fields store the internal state of the scale.

        * **L_ref** (`float | Float[Array, "..."]`): The stored reference length.

    Examples:
        ```python
        scale = CharacteristicLengthScale(L_ref=1.0)
        factors = scale.get_scale(x)
        ```
    """

    L_ref: float | Float[Array, "..."]

    def get_scale(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Returns the reference length as a JAX array.

        Args:
            x: Input array (unused here but required by protocol).

        Returns:
            The reference length scale.
        """
        return jnp.asarray(self.L_ref)


class ReynoldsScale(eqx.Module):
    r"""Scales data based on the Reynolds number $Re = UL / \\nu$.

    A fundamental scaling in fluid mechanics used to non-dimensionalize
    velocity and pressure fields.

    Args:
        U: Reference velocity.
        L: Reference length.
        nu: Kinematic viscosity.

    !!! info "Internal Attributes"
        These fields store the internal state of the scale.

        * **U** (`float | Float[Array, "..."]`): Reference velocity.
        * **L** (`float | Float[Array, "..."]`): Reference length.
        * **nu** (`float | Float[Array, "..."]`): Kinematic viscosity.

    Examples:
        ```python
        # Define physics for a flow problem
        scale = ReynoldsScale(U=10.0, L=1.0, nu=1e-3)
        re_number = scale.get_scale(x)
        ```
    """

    U: float | Float[Array, "..."]
    L: float | Float[Array, "..."]
    nu: float | Float[Array, "..."]

    def get_scale(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Computes the Reynolds number.

        Args:
            x: Input array (unused here but required by protocol).

        Returns:
            The computed Reynolds number.
        """
        return jnp.asarray((self.U * self.L) / self.nu)


class GridBasedScale(eqx.Module):
    """Applies spatially-varying scaling factors from a grid.

    Used when the scaling depends on the location in the domain, such as
    varying mesh density or coordinate-dependent physics.

    Args:
        scale_field: An array of scaling factors
            defined on the domain grid.

    !!! info "Internal Attributes"
        These fields store the internal state of the scale.

        * **scale_field** (`Float[Array, "..."]`): The spatially-varying scaling factors.

    Examples:
        ```python
        grid_factors = jnp.linspace(1.0, 2.0, 32)
        scale = GridBasedScale(scale_field=grid_factors)
        factors = scale.get_scale(x_batch)
        ```
    """

    scale_field: Float[Array, "..."]

    def get_scale(
        self, x: Float[Array, "..."], eps: float = 1e-7
    ) -> Float[Array, "..."]:
        """Returns the spatially-varying scale field, broadcast to `x`.

        Args:
            x: The input array (e.g., a batch of velocity fields).
            eps: Small value to prevent division by zero.
                Defaults to 1e-7.

        Returns:
            The scaling field broadcast to the shape of `x`.
        """
        s_field = jnp.where(self.scale_field > eps, self.scale_field, eps)

        if x.ndim > s_field.ndim:
            return jnp.broadcast_to(s_field, x.shape)
        return s_field


__all__ = [
    "PhysicalScale",
    "CharacteristicLengthScale",
    "ReynoldsScale",
    "GridBasedScale",
]
