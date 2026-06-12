"""Implementation of an abstract base loss class."""

import abc

import equinox as eqx
from jaxtyping import Array, Float


class BaseLoss(eqx.Module):
    """Abstract base loss.

    New losses should inherit from BaseLoss.
    This class cannot be instantiated,
    its child class must be final,
    following the "Abstract or Final" pattern.

    !!! info "Internal Attributes"
        These fields store the internal state of the loss.

        * **weight** (`Float[Array, ""]`): Weight for weighted losses e.g. in compositions. Optionally learnable if specified in child classes.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
    """

    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Computes loss.

        Args:
            model: The model being trained. Default `None`.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
                Required for operator losses (e.g. Sobolev).
            pred: Optional pre-computed model prediction array.
            **kwargs: Additional arguments for specific losses.

        Returns:
            Scalar loss.
        """
        ...
