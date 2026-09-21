"""Metric utility functions."""

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree

from neojax.metrics.base_metric import BaseMetric
from neojax.metrics.composed_metric import ComposedMetric


def is_learnable_metric_weight(metric: PyTree) -> PyTree:
    """Creates a filter spec to correctly handle learnable weights of metrics.

    Produces a boolean PyTree that can be used with `eqx.partition` to separate
    learnable metric weights from non-learnable weights.

    It returns `False` for most fields, but `True` for:

    - Metric `weight` attrs only if their `learnable_weight` is `True`.

    Args:
        metric: The metric function or PyTree of metric to generate a filter for.

    Returns:
        A boolean PyTree of the same structure as `metric`.

    Examples:
        ```python
        import equinox as eqx
        import jax
        import jax.tree_util as jtu
        from neojax.metrics.utils import is_learnable_metric_weight

        # Define a combined model and metric wrapper class
        class ModelWithMetric(eqx.Module):
            model: eqx.Module
            metric_fn: eqx.Module

        model_with_metric = ModelWithMetric(model, metric_fn)

        # Combine model parameters and learnable metric weights filter specs
        filter_spec = ModelWithMetric(
            model=jtu.tree_map(eqx.is_inexact_array, model),
            metric_fn=is_learnable_metric_weight(metric_fn)
        )
        # Note that the same may be achieved using eqx.filter_...
        # Partition into differentiable and static parts
        diff, static = eqx.partition(model_with_metric, filter_spec)

        # Compute metric and gradients using jax.value_and_grad
        @jax.value_and_grad
        def grad_metric_fn(diff_args, static_args, x, y):
            combined = eqx.combine(diff_args, static_args)
            pred = jax.vmap(combined.model)(x)
            return combined.metric_fn(pred=pred, target=y)

        metric_val, grads = grad_metric_fn(diff, static, x, y)

        # Perform gradient update step
        updates = jtu.tree_map(lambda g: -g if g is not None else None, grads)
        new_diff = eqx.apply_updates(diff, updates)
        new_model_with_metric = eqx.combine(new_diff, static)
        ```

    !!! info "Usage with a stateful `composition_fn`"
        A `ComposedMetric.composition_fn` may itself be a stateful
        `eqx.Module` with its own learnable parameters. Since that
        learnability is a property of `composition_fn` itself, not of the
        enclosing `ComposedMetric.weight`, any inexact array found inside
        `composition_fn` is always marked trainable here, regardless of
        `ComposedMetric.learnable_weight`.
    """

    def _get_mask(node):
        if isinstance(node, BaseMetric):
            # Only allow the weight to be 'True' if metric.learnable_weight.
            # Nested BaseMetric children of ComposedMetric are recursed into.
            def _leaf_mask(path, leaf):
                if isinstance(leaf, BaseMetric):
                    return _get_mask(leaf)
                if (
                    isinstance(node, ComposedMetric)
                    and path
                    and getattr(path[0], "name", None) == "composition_fn"
                ):
                    # composition_fn's own learnable state (if any) is
                    # governed by whatever constructed it, not by the
                    # enclosing ComposedMetric's own (unrelated) weight flag.
                    return eqx.is_inexact_array(leaf)
                return eqx.is_inexact_array(leaf) and node.learnable_weight

            return jtu.tree_map_with_path(
                _leaf_mask,
                node,
                is_leaf=lambda x: (
                    (x is not node and isinstance(x, BaseMetric))
                    or eqx.is_inexact_array(x)
                ),
            )

        return False

    return jtu.tree_map(
        _get_mask,
        metric,
        is_leaf=lambda x: isinstance(x, BaseMetric) or eqx.is_inexact_array(x),
    )


__all__ = ["is_learnable_metric_weight"]
