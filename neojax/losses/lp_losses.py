"""Implementations of various $L^{p}$-losses."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from neojax.losses.base_loss import BaseLoss


@final
class LpLoss(BaseLoss):
    r"""General $L^{p}$-loss.

    Computes
    $$
    \Vert y - \hat{y}\Vert_{L^p} = \Bigg(\int_{\Omega} \Big| y - \hat{y} \Big|^p dx \Bigg)^{\frac{1}{p}}
    $$
    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    Args:
        p: Power of the norm.
        weight: (Learnable) weight. Loss is computed as `weight` * `loss`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_loss_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.

    !!! info "Internal Attributes"
        These fields store the internal state of the loss.

        * **weight** (`Float[Array, ""]`): Learnable loss weight. Filter during training to prevent updates.
        * **p** (`float | int`): Power of the norm.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
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
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Computes the $L^{p}$-loss.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the loss computes `vmap(model)(x)`.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If `pred` is `None` or `model` and `x` are None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "LpLoss requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_loss(p_i, t_i):
            diff_pow = jnp.pow(jnp.abs(t_i - p_i), self.p)
            return jnp.pow(jnp.mean(diff_pow), 1 / self.p)

        batch_loss = jax.vmap(single_loss)(pred, target)
        return self.weight * jnp.mean(batch_loss)


@final
class RelativeLpLoss(BaseLoss):
    r"""General relative $L^{p}$-loss.

    Computes
    $$
    \frac{\Vert y - \hat{y}\Vert_{L^p}}{\Vert y \Vert_{L^p}}
    $$
    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    Args:
        p: Power of the norm.
        weight: (Learnable) weight. Loss is computed as `weight` * `loss`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_loss_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.

    !!! info "Internal Attributes"
        These fields store the internal state of the loss.

        * **weight** (`Float[Array, ""]`): Learnable loss weight. Filter during training to prevent updates.
        * **p** (`float | int`): Power of the norm.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
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
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Computes relative $L^{p}$-loss.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the loss computes `vmap(model)(x)`.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If `pred` is `None` or `model` and `x` are None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "RelativeLpLoss requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_loss(p_i, t_i):
            diff_pow_error = jnp.pow(jnp.abs(t_i - p_i), self.p)
            lp_norm = jnp.pow(jnp.mean(diff_pow_error), 1 / self.p)
            target_norm = jnp.pow(jnp.mean(jnp.pow(jnp.abs(t_i), self.p)), 1 / self.p)
            return lp_norm / (target_norm + 1e-7)

        batch_loss = jax.vmap(single_loss)(pred, target)
        return self.weight * jnp.mean(batch_loss)
