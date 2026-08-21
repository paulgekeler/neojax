"""Implementation of an abstract base metric class."""

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, PyTree


class BaseMetric(eqx.Module):
    """Abstract base metric.

    New metrics and loss functions should inherit from BaseMetric.
    This class cannot be instantiated directly. Subclasses must be final
    following the "Abstract or Final" class design pattern.

    Every metric evaluates predictions against a ground-truth target. It can also
    act as a differentiable loss function during training. To support learnable
    weighting (e.g., in multi-task loss compositions), each metric carries a weight
    field that can optionally be flagged as learnable.

    ??? info "Attributes"
        * **weight** (`Float[Array, ""]`): Scalar weight coefficient for the metric.
            Used to balance multiple metrics/losses in compositions.
        * **learnable_weight** (`bool`): Static flag indicating whether the weight
            is learnable via gradient descent.
    """

    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)

    @abstractmethod
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: PyTree[Float[Array, "b ..."]] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes metric.

        Args:
            model: The neural operator model being evaluated. Default is None.
                Required for operator-dependent metrics (e.g., Sobolev metrics).
            target: Ground truth array of shape (batch, channels, *spatial).
            x: Input array to the model of shape (batch, in_channels, *spatial).
                Required for operator-dependent metrics.
            pred: Optional pre-computed model prediction array of shape
                (batch, channels, *spatial). If not provided, the metric will
                attempt to compute it by evaluating the model on `x`.
            **kwargs: Additional parameters for custom metrics.

        Returns:
            Scalar metric value.
        """
        ...
