"""Implementation of loss composition class."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.losses.base_loss import BaseLoss


@final
class ComposedLoss(BaseLoss):
    """Wraps loss compositions.

    Composes different losses by passing them to the
    classes init method.

    Args:
        losses: Arbitrary number of loss instances to compose.
        weight: Optional weighting for ComposedLoss.
            Default is 1.0, i.e., no weighting.

    !!! info "Internal Attributes"
        These fields store the internal state of the loss.

        * **weight** (`Float[Array, ""]`): Learnable loss weight. Filter during training to prevent updates.
        * **losses** (`tuple[BaseLoss, ...]`): Composed loss functions. The composition is the sum.

    !!! info
        This loss may be called without a `model` by using keywords:
        `loss_fn(pred=y_hat, target=y)`. The `model` parameter is kept
        for clean compatibility with JAX transformations and
        if composing model-dependent losses (e.g. Sobolev loss).

    Examples:
        ```python
        import jax.numpy as jnp
        from neojax.losses import ComposedLoss, LpLoss, RelativeLpLoss

        loss_1 = LpLoss(p=2)
        loss_2 = RelativeLpLoss(p=2)

        composed_loss = ComposedLoss(loss_1, loss_2)

        x = jnp.ones((1, 10))
        y = jnp.zeros((1, 10))

        loss_val = composed_loss(pred=x, target=y)
        ```
    """

    losses: tuple[BaseLoss, ...]

    def __init__(
        self,
        *losses: BaseLoss,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.losses = tuple(losses)
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Computes the composed loss.

        Args:
            model: The model being trained.
            target: Ground truth array.
            x: Model input array.
            pred: Model prediction array.

        Returns:
            Scalar loss.
        """
        return self.weight * jnp.sum(
            jnp.array(
                [
                    loss(model=model, target=target, x=x, pred=pred, **kwargs)
                    for loss in self.losses
                ]
            )
        )
