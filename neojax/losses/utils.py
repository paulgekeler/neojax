"""Loss utility functions."""

import equinox as eqx
import jax
from jaxtyping import PyTree

from neojax.losses.base_loss import BaseLoss


def is_learnable_loss_weight(loss: PyTree) -> PyTree:
    """Creates a filter spec to correctly handle learnable weights of losses.

    Produces a boolean PyTree that can be used with `eqx.partition` to separate
    learnable loss weights from non-learnable weights.

    It returns `False` for most fields, but `True` for:

    - Loss `weight` attrs only if their `learnable_weight` is `True`.

    Args:
        loss: The loss function or PyTree of loss to generate a filter for.

    Returns:
        A boolean PyTree of the same structure as `loss`.

    Example:
        ```python
        import equinox as eqx
        import jax
        from neojax.losses.utils import is_learnable_loss_weight

        # Define a combined model and loss wrapper class
        class ModelWithLoss(eqx.Module):
            model: eqx.Module
            loss_fn: eqx.Module

        model_with_loss = ModelWithLoss(model, loss_fn)

        # Combine model parameters and learnable loss weights filter specs
        filter_spec = ModelWithLoss(
            model=jax.tree_util.tree_map(eqx.is_inexact_array, model),
            loss_fn=is_learnable_loss_weight(loss_fn)
        )

        # Partition into differentiable and static parts
        diff, static = eqx.partition(model_with_loss, filter_spec)

        # Compute loss and gradients using jax.value_and_grad
        @jax.value_and_grad
        def grad_loss_fn(diff_args, static_args, x, y):
            combined = eqx.combine(diff_args, static_args)
            pred = jax.vmap(combined.model)(x)
            return combined.loss_fn(pred=pred, target=y)

        loss_val, grads = grad_loss_fn(diff, static, x, y)

        # Perform gradient update step
        updates = jax.tree_util.tree_map(lambda g: -g if g is not None else None, grads)
        new_diff = eqx.apply_updates(diff, updates)
        new_model_with_loss = eqx.combine(new_diff, static)
        ```
    """

    def _get_mask(node):
        if isinstance(node, BaseLoss):
            # Only allow the weight to be 'True'
            # if the user enabled loss.learnable_weight
            return jax.tree_util.tree_map(
                lambda leaf: eqx.is_inexact_array(leaf) and node.learnable_weight, node
            )
        # For everything else, return False
        return False

    return jax.tree_util.tree_map(
        _get_mask,
        loss,
        is_leaf=lambda x: isinstance(x, BaseLoss) or eqx.is_inexact_array(x),
    )


__all__ = ["is_learnable_loss_weight"]
