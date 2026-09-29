"""This submodule contains additional tree utilities."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PyTree


def tree_add_broadcast_to(
    tree: PyTree,
    *,
    axis: int,
    size: int,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Broadcasts the given axis of each leaf of the given tree to given size.

    How this function works depends on if the given axis has the given
    size or not:

    - If the given axis already has the given size, the leaf is
    returned unchanged.
    - If the given axis has size 1, this axis is broadcasted to size.
    - If the given axis is not broadcastable to size, a new axis is inserted
    and broadcasted.

    Args:
        tree: Pytree of arrays to broadcast.
        axis: Array axis to add.
        size: Size of given axis.
        is_leaf: Optional callable specifying whether to traverse the current
            object at each flattening step. Default is `None`, that is only
            arrays are treated as leaves.

    Returns:
        Pytree of arrays with additional axis of `size` length.

    Examples:
        >>> import jax.numpy as jnp
        >>> pytree = {"one": jnp.ones((4, 2, 1, 3)), "two": jnp.ones((4, 2, 3)) * 2.0}
        >>> axis = 2
        >>> size = 3
        >>> bc_tree = tree_add_broadcast_to(pytree, axis=axis, size=size)
        >>> bc_tree["one"].shape
        (4, 2, 3, 3)
        >>> bc_tree["two"].shape
        (4, 2, 3)

    !!! warning "Out-of-bounds axis"
        This function silently wraps an out-of-bounds axis
        via modulo.
    """

    def leaf_add_broadcast_to(leaf: Array, axis: int, size: int) -> Array:
        if -leaf.ndim <= axis < leaf.ndim and leaf.shape[axis] == size:  # noqa: F823
            return leaf
        elif -leaf.ndim <= axis < leaf.ndim and leaf.shape[axis] == 1:
            if axis == -1:
                # for axis=-1, shape[0:] appends the entire shape
                return jnp.broadcast_to(leaf, leaf.shape[:axis] + (size,))
            return jnp.broadcast_to(
                leaf, leaf.shape[:axis] + (size,) + leaf.shape[axis + 1 :]
            )

        else:
            # normalize negative axes
            axis = axis % (leaf.ndim + 1)
            return jnp.broadcast_to(
                jnp.expand_dims(leaf, axis=axis),
                leaf.shape[:axis] + (size,) + leaf.shape[axis:],
            )

    return jt.map(
        lambda x: leaf_add_broadcast_to(x, axis, size),
        tree,
        is_leaf=is_leaf,
    )
