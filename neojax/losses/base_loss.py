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

    Attributes:
        weight: Weight for weighted losses
            e.g. in compositions.
            Optionally learnable if specified
            in child classes.
    """

    weight: Float[Array, "1"]

    @abc.abstractmethod
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        pred: Float[Array, "c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, "1"]: ...
