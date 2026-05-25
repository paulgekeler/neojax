"""Implementation of loss composition class."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.losses.base_loss import BaseLoss


class ComposedLoss(BaseLoss):
    """Wraps loss compositions.

    Composes different losses by passing them to the
    classes init method.

    Args:
        losses: Arbitrary number of loss instances to compose.
        weight: Optional weighting for ComposedLoss.
            Default is 1.0, i.e., no weighting.

    Attributes:
        weight: Learnable loss weight. Filter during training
            to prevent updates.
        losses: Composed loss functions. The composition is the sum.

    Note:
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

    def __init__(self, *losses: Sequence[BaseLoss], weight: float = 1.0) -> None:
        self.losses = tuple(losses)
        self.weight = jnp.array(weight)

    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        pred: Float[Array, "c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, "1"]:
        """Computes the composed loss.

        Args:
            model: The model being trained. Default `None` in
                data-fitting tasks where only `pred` and
                `target` are required. Kept for compatibility with
                Jax and Equinox.
            pred: Model prediction array shaped (c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).

        Returns:
            Scalar loss.
        """
        return self.weight * jnp.sum(
            jnp.array(
                [loss(model=model, pred=pred, target=target) for loss in self.losses]
            )
        )
