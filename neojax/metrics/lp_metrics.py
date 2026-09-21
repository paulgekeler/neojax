"""Implementations of various $L^{p}$-metrics."""

import math
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from typing_extensions import override

from neojax.metrics.base_metric import BaseMetric


@final
class LpMetric(BaseMetric):
    r"""General $L^{p}$-metric.

    Computes

    $$
    \Vert y - \hat{y}\Vert_{L^p} = \Bigg(\int_{\Omega} \Big| y - \hat{y} \Big|^p dx \Bigg)^{\frac{1}{p}}
    $$

    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    Args:
        p: Power of the norm. Can be a number or "inf" (representing L-infinity norm).
        weight: (Learnable) scalar weight coefficient of the metric.
            The metric is computed as 'metric * weight'. Default is 1.0.
            If `learnable_weight` is True, weight has to be strictly positive and
            will be optimized via gradient descent.
        learnable_weight: Optional flag indicating whether `weight` is learnable.
            If `True`, this flag is used in the `is_learnable_metric_weight` filter function
            to indicate to `eqx.filter_...` or `eqx.partition` that `weight` should be
            adapted by the optimizer. Default is `False`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **p** (`float | int | Literal["inf"]`): Power of the norm.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **is_p_inf** (`bool`): Flag indicating if `p` is infinity.
    """

    p: float | int | Literal["inf"] = eqx.field(static=True)
    is_p_inf: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float | int | Literal["inf"],
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.p = p
        super().__init__(weight=weight, learnable_weight=learnable_weight)
        self.is_p_inf = (p == "inf") or (isinstance(p, int | float) and math.isinf(p))

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: PyTree[Float[Array, "b ..."]] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the $L^{p}$-metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the metric computes `vmap(model)(x)`.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `pred` is `None` or `model` and `x` are None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "LpMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        if self.is_p_inf:

            def single_metric(p_i, t_i):
                return jnp.max(jnp.abs(t_i - p_i))
        else:

            def single_metric(p_i, t_i):
                diff_pow = jnp.pow(jnp.abs(t_i - p_i), self.p)
                return jnp.pow(jnp.mean(diff_pow), 1 / self.p)

        batch_metric = jax.vmap(single_metric)(pred, target)
        return self.weight * jnp.mean(batch_metric)


@final
class RelativeLpMetric(BaseMetric):
    r"""General relative $L^{p}$-metric.

    Computes

    $$
    \frac{\Vert y - \hat{y}\Vert_{L^p}}{\Vert y \Vert_{L^p}}
    $$

    where $y$ is the ground truth and $\hat{y}$ is the model prediction.

    Args:
        p: Power of the norm. Can be a number or "inf" (representing L-infinity norm).
        weight: (Learnable) scalar weight coefficient of the metric.
            The metric is computed as 'metric * weight'. Default is 1.0.
            If `learnable_weight` is True, weight has to be strictly positive and
            will be optimized via gradient descent.
        learnable_weight: Optional flag indicating whether `weight` is learnable.
            If `True`, this flag is used in the `is_learnable_metric_weight` filter function
            to indicate to `eqx.filter_...` or `eqx.partition` that `weight` should be
            adapted by the optimizer. Default is `False`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **p** (`float | int | Literal["inf"]`): Power of the norm.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **is_p_inf** (`bool`): Flag indicating if `p` is infinity.
    """

    p: float | int | Literal["inf"] = eqx.field(static=True)
    is_p_inf: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float | int | Literal["inf"],
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.p = p
        super().__init__(weight=weight, learnable_weight=learnable_weight)
        self.is_p_inf = (p == "inf") or (isinstance(p, int | float) and math.isinf(p))

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: PyTree[Float[Array, "b ..."]] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes relative $L^{p}$-metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the metric computes `vmap(model)(x)`.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `pred` is `None` or `model` and `x` are None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "RelativeLpMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        if self.is_p_inf:

            def single_metric(p_i, t_i):
                lp_norm = jnp.max(jnp.abs(t_i - p_i))
                target_norm = jnp.max(jnp.abs(t_i))
                return lp_norm / (target_norm + 1e-7)
        else:

            def single_metric(p_i, t_i):
                diff_pow_error = jnp.pow(jnp.abs(t_i - p_i), self.p)
                lp_norm = jnp.pow(jnp.mean(diff_pow_error), 1 / self.p)
                target_norm = jnp.pow(
                    jnp.mean(jnp.pow(jnp.abs(t_i), self.p)), 1 / self.p
                )
                return lp_norm / (target_norm + 1e-7)

        batch_metric = jax.vmap(single_metric)(pred, target)
        return self.weight * jnp.mean(batch_metric)
