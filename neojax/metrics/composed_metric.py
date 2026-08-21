"""Implementation of metric composition class."""

from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
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
        weight: Optional weighting for ComposedMetric.
            Default is 1.0, i.e., no weighting.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable scalar metric weight. Filter during training to prevent updates.
        * **metrics** (`tuple[BaseMetric, ...]`): Composed metric functions. The composition is the sum.

    !!! info
        This metric may be called without a `model` by using keywords:
        `metric_fn(pred=y_hat, target=y)`. The `model` parameter is kept
        for clean compatibility with JAX transformations and
        if composing model-dependent metrics (e.g. Sobolev metric).

    Examples:
        ```python
        import jax.numpy as jnp
        from neojax.metrics import ComposedMetric, LpMetric, RelativeLpMetric

        metric_1 = LpMetric(p=2)
        metric_2 = RelativeLpMetric(p=2)

        composed_metric = ComposedMetric(metric_1, metric_2)

        x = jnp.ones((1, 10))
        y = jnp.zeros((1, 10))

        metric_val = composed_metric(pred=x, target=y)
        ```
    """

    metrics: tuple[BaseMetric, ...]
    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)

    def __init__(
        self,
        *metrics: BaseMetric,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.metrics = tuple(metrics)
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

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
        """Computes the composed metric.

        Args:
            model: The model being trained.
            target: Ground truth array.
            x: Model input array.
            pred: Model prediction array.

        Returns:
            Scalar metric value.
        """
        vals = jnp.array(
            [
                metric(model=model, target=target, x=x, pred=pred, **kwargs)
                for metric in self.metrics
            ]
        )
        return self.weight * jnp.sum(vals)
