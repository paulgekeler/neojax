"""Loss utility functions."""

import equinox as eqx
import jax
from jaxtyping import PyTree

from neojax.losses.base_loss import BaseLoss


def is_learnable_loss_weight(model: PyTree) -> PyTree:
    """Creates a filter spec to correctly handle learnable weights.

    Produces a boolean PyTree that can be used directly with
    `eqx.partition`, `eqx.filter_grad`, or `eqx.filter_jit`.
    It returns `False` for most fields, but `True` for:
    1. All inexact arrays (trainable parameters) in the model.
    2. Loss `weight` attrs only if their `learnable_weight` is `True`.

    Args:
        model: The model or list of losses to generate a filter for.

    Returns:
        A boolean PyTree of the same structure as `model`.

    Example:
        ```python
        import equinox as eqx
        from neojax.losses.utils import is_learnable_loss_weight

        # Directly use as the filter for grad
        filter_spec = is_learnable_loss_weight(model)
        grads = eqx.filter_grad(loss_fn, filter=filter_spec)(model, ...)
        ```
    """

    def _get_mask(node):
        if isinstance(node, BaseLoss):
            # Only allow the weight to be 'True'
            # if the user enabled loss.learnable_weight
            return jax.tree_util.tree_map(
                lambda leaf: eqx.is_inexact_array(leaf) and node.learnable_weight, node
            )
        # For everything else (like FNO parameters), use standard array check
        return eqx.is_inexact_array(node)

    return jax.tree_util.tree_map(
        _get_mask,
        model,
        is_leaf=lambda x: isinstance(x, BaseLoss) or eqx.is_inexact_array(x),
    )


__all__ = ["is_learnable_loss_weight"]
