"""Implementations of various $L^{p}$-losses."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.losses.base_loss import BaseLoss


class LpLoss(BaseLoss):
    r"""General $L^{p}$-loss.

    Computes
    $$
    \Vert y - \hat{y}\Vert_{L^p} = \Bigg(\int_{\Omega} \Big| y - \hat{y} \Big|^p dx \Bigg)^{\frac{1}{p}}
    $$
    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    To compute batch losses, use `jax.vmap` and e.g. `jnp.mean`
    ```python
    vmapped_loss = jax.vmap(lambda p, t: loss(pred=p, target=t), in_axes=(0, 0))
    batch_loss = vmapped_loss(preds, targets).mean()
    ```

    Args:
        p: Power of the norm.
        weight: (Learnable) weight. Loss is computed as `weight` * `loss`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_loss_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.

    Attributes:
        weight: Learnable loss weight. Filter during training
            to prevent updates.
        p: Power of the norm.
        learnable_weight: Flag indicating whether `weight` is learnable.

    Note:
        This loss should be called without a `model` by using keywords:
        `loss_fn(pred=y_hat, target=y)`. The `model` parameter is kept
        for clean compatibility with JAX transformations.
    """

    p: float | int = eqx.field(static=True)
    learnable_weight: bool = eqx.field(static=True)

    def __init__(
        self, p: float | int, weight: float = 1.0, learnable_weight: bool = False
    ) -> None:
        self.p = p
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        pred: Float[Array, "c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, "1"]:
        """Computes the $L^{p}$-loss.

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
        diff_pow = jnp.pow(jnp.abs(target - pred), self.p)
        mean = jnp.mean(diff_pow)
        return self.weight * jnp.pow(mean, 1 / self.p)


class RelativeLpLoss(BaseLoss):
    r"""General relative $L^{p}$-loss.

    Computes
    $$
    \frac{\Vert y - \hat{y}\Vert_{L^p}}{\Vert y \Vert_{L^p}}
    $$
    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    To compute batch losses, use `jax.vmap` and e.g. `jnp.mean`
    ```python
    vmapped_loss = jax.vmap(lambda p, t: loss(pred=p, target=t), in_axes=(0, 0))
    batch_loss = vmapped_loss(preds, targets).mean()
    ```

    Args:
        p: Power of the norm.
        weight: (Learnable) weight. Loss is computed as `weight` * `loss`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_loss_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.

    Attributes:
        weight: Learnable loss weight. Filter during training
            to prevent updates.
        p: Power of the norm.
        learnable_weight: Flag indicating whether `weight` is learnable.

    Note:
        This loss should be called without a `model` by using keywords:
        `loss_fn(pred=y_hat, target=y)`. The `model` parameter is kept
        for clean compatibility with JAX transformations.
    """

    p: float | int = eqx.field(static=True)
    learnable_weight: bool = eqx.field(static=True)

    def __init__(
        self, p: float | int, weight: float = 1.0, learnable_weight: bool = False
    ) -> None:
        self.p = p
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        pred: Float[Array, "c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, "1"]:
        """Computes relative $L^{p}$-loss.

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
        diff_pow_error = jnp.pow(jnp.abs(target - pred), self.p)
        mean_error = jnp.mean(diff_pow_error)
        lp_norm = jnp.pow(mean_error, 1 / self.p)

        target_diff_pow = jnp.pow(jnp.abs(target), self.p)
        target_mean = jnp.mean(target_diff_pow)
        target_norm = jnp.pow(target_mean, 1 / self.p)
        return self.weight * (lp_norm / target_norm)
