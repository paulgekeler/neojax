"""Implementation of metric composition class."""

import warnings
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
from jaxtyping import Array, Float
from typing_extensions import override

from neojax.metrics.base_metric import BaseMetric


@final
class ComposedMetric(BaseMetric):
    """Wraps metric compositions.

    Composes different metrics by passing them to the class init method and
    evaluating them as a weighted sum.

    Args:
        metrics: Arbitrary number of metric instances to compose.
        composition_fn: Callable combining the given (unweighted) metrics
            into a scalar metric. Called as `composition_fn(*metrics)`.
            May be a plain function, or a stateful `eqx.Module` with its own
            learnable parameters (see the note below). It is required.
        weight: (Learnable) scalar weight coefficient of the metric.
            The metric is computed as 'metric * weight'. Default is 1.0.
            If `learnable_weight` is True, weight has to be strictly positive and
            will be optimized via gradient descent.
        learnable_weight: Optional flag indicating whether `weight` is learnable.
            If `True`, this flag is used in the `is_learnable_metric_weight` filter function
            to indicate to `eqx.filter_...` or `eqx.partition` that `weight` should be
            adapted by the optimizer. Default is `False`.

    Warns:
        UserWarning: If any of the composed metrics' `learnable_weight` is `True`.
            Individual learnable weights should be controlled with the given
            `composition_fn`.

    Examples:
        ```python
        import jax.numpy as jnp
        from neojax.metrics import ComposedMetric, LpMetric, RelativeLpMetric

        metric_1 = LpMetric(p=2)
        metric_2 = RelativeLpMetric(p=2)

        composed_metric = ComposedMetric(metric_1, metric_2, composition_fn=lambda ms: jnp.sum(*ms))

        x = jnp.ones((1, 10))
        y = jnp.zeros((1, 10))

        metric_val = composed_metric(pred=x, target=y)
        ```

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable scalar metric weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **metrics** (`tuple[BaseMetric, ...]`): Composed metric functions. The composition is the sum.
        * **composition_fn** (`Callable[[BaseMetric, ...], Float[Array, ""]]`): Callable combining the given metrics
            into a scalar metric.

    !!! info
        This metric may be called without a `model` by using keywords:
        `metric_fn(pred=y_hat, target=y)`. The `model` parameter is kept
        for clean compatibility with JAX transformations and
        if composing model-dependent metrics (e.g. Sobolev metric).

    !!! info "A stateful, learnable `composition_fn`"
        `composition_fn` doesn't have to be a plain function, it can be an
        `eqx.Module` with its own learnable parameters, e.g. Kendall & Gal
        uncertainty weighting's per-term log-variances:
        ```python
        class UncertaintyWeighting(eqx.Module):
            log_vars: Float[Array, " n"]

            def __call__(self, *raw_vals):
                return sum(
                    jnp.exp(-s) * v + s for v, s in zip(raw_vals, self.log_vars)
                )

        composed = ComposedMetric(
            metric_1, metric_2, composition_fn=UncertaintyWeighting(jnp.zeros(2))
        )
        ```
        `is_learnable_metric_weight` (see its own docs) marks any inexact
        array found inside `composition_fn` as trainable unconditionally, so
        `log_vars` above is included in the trainable partition regardless of
        `ComposedMetric.learnable_weight`.

    !!! warning "Why a `composition_fn` must be passed"
        A naive composition function such as `sum(*metrics)` gives the optimizer an
        easy way to cancel its components: drive the weights of its metrics to zero.
        To prevent such a 'free lunch', make sure to pass a `composition_fn`
        that has no such flaw, such as Kendall & Gal's uncertainty weighting (see above).

    !!! warning "Using metric weighting with `ComposedMetric`"
        Any weight, whether learnable or fixed of the composed metrics is ignored
        inside `ComposedMetric`. A `UserWarning` is raised if any of the weights
        is not 1.0, indicating it was set by the user. To weight individual metrics,
        use the `composition_fn` instead.
    """

    metrics: tuple[BaseMetric, ...]
    composition_fn: Callable[[BaseMetric, ...], Float[Array, ""]]

    def __init__(
        self,
        *metrics: BaseMetric,
        composition_fn: Callable[[BaseMetric, ...], Float[Array, ""]],
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        for metric in metrics:
            if metric.weight != 1.0:
                warnings.warn(
                    f"{type(metric).__name__}'s own 'learnable_weight' has no "
                    "effect once composed: ComposedMetric passes 'composition_fn' "
                    "each metric's raw (unweighted) value, so this weight can "
                    "never receive a gradient. This also applies to fixed weights. "
                    "Control weighting through 'composition_fn' (see documentation).",
                    UserWarning,
                    stacklevel=2,
                )
        self.metrics = tuple(metrics)
        self.composition_fn = composition_fn
        super().__init__(weight=weight, learnable_weight=learnable_weight)

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        return_components: bool = False,
        **kwargs: Any,
    ) -> Float[Array, ""] | tuple[Float[Array, ""], tuple[Float[Array, ""], ...]]:
        """Computes the composed metric.

        Args:
            model: The model being trained.
            target: Ground truth array.
            x: Model input array.
            pred: Model prediction array.
            return_components: If `True`, also returns each metric's
                own raw (unweighted) value alongside the composed
                value.

        Returns:
            Scalar metric value.
        """
        raw_vals = tuple(
            metric(model=model, target=target, x=x, pred=pred, **kwargs) / metric.weight
            for metric in self.metrics
        )
        res = self.weight * self.composition_fn(*raw_vals)
        if return_components:
            return res, raw_vals
        return res
