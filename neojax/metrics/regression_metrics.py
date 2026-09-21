"""Implementations of standard regression metrics."""

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from neojax.metrics.base_metric import BaseMetric


@final
class MSEMetric(BaseMetric):
    r"""Mean Squared Error (MSE) metric.

    Computes the mean squared difference between predictions and targets:

    $$
    \text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    $$

    where $y$ is the ground truth target and $\hat{y}$ is the model prediction.

    Args:
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
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
    """

    def __init__(
        self,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        super().__init__(weight=weight, learnable_weight=learnable_weight)

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the Mean Squared Error (MSE) metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the metric computes `vmap(model)(x)`.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `pred` is `None` and either `model` or `x` is None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "MSEMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_metric(p_i, t_i):
            return jnp.mean(jnp.square(t_i - p_i))

        batch_metric = jax.vmap(single_metric)(pred, target)
        return self.weight * jnp.mean(batch_metric)


@final
class RMSEMetric(BaseMetric):
    r"""Root Mean Squared Error (RMSE) metric.

    Computes the square root of the mean squared difference between predictions and targets:

    $$
    \text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \epsilon}
    $$

    where $y$ is the ground truth target, $\hat{y}$ is the model prediction,
    and $\epsilon = 10^{-7}$ is a small regularization constant to prevent NaN gradients at zero.

    Args:
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
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
    """

    def __init__(
        self,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        super().__init__(weight=weight, learnable_weight=learnable_weight)

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the Root Mean Squared Error (RMSE) metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the metric computes `vmap(model)(x)`.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `pred` is `None` and either `model` or `x` is None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "RMSEMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_metric(p_i, t_i):
            return jnp.sqrt(jnp.mean(jnp.square(t_i - p_i)) + 1e-7)

        batch_metric = jax.vmap(single_metric)(pred, target)
        return self.weight * jnp.mean(batch_metric)


@final
class R2Metric(BaseMetric):
    r"""Coefficient of Determination ($R^2$ score) metric.

    Computes the proportion of the variance in the target that is predictable from the model prediction:

    $$
    R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2 + \epsilon}
    $$

    where $y$ is the ground truth target, $\hat{y}$ is the model prediction,
    $\bar{y}$ is the mean of the ground truth target, and $\epsilon = 10^{-7}$
    is a small regularization constant to prevent division by zero.

    Args:
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
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
    """

    def __init__(
        self,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        super().__init__(weight=weight, learnable_weight=learnable_weight)

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the Coefficient of Determination ($R^2$ score) metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Ground truth array shaped (batch, c, d1, ..., dN).
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array.
                If None, the metric computes `vmap(model)(x)`.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `pred` is `None` and either `model` or `x` is None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "R2Metric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_metric(p_i, t_i):
            mean_t = jnp.mean(t_i)
            ss_res = jnp.sum(jnp.square(t_i - p_i))
            ss_tot = jnp.sum(jnp.square(t_i - mean_t))
            return 1.0 - ss_res / (ss_tot + 1e-7)

        batch_metric = jax.vmap(single_metric)(pred, target)
        return self.weight * jnp.mean(batch_metric)
