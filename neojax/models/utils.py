"""Utilities shared by the models."""

from typing import Any

import equinox as eqx
from jaxtyping import PyTree


def apply_layer(
    layer: eqx.Module,
    x: PyTree,
    state: eqx.nn.State | None = None,
    **kwargs: Any,
) -> tuple[PyTree, eqx.nn.State | None]:
    """Applies a layer that may or may not be stateful.

    Follows the convention of `eqx.nn.Sequential`: A layer is called as
    `out, state = layer(x, state)` if it is an `eqx.nn.StatefulLayer` with
    `layer.is_stateful()`, otherwise as `out = layer(x)`. This allows to write
    models in terms of a threaded `state` that does not care whether
    individual layers (or the whole model, depending on its configuration)
    have state, see
    [Equinox docs](https://docs.kidger.site/equinox/api/nn/stateful/).

    Args:
        layer: The layer to call.
        x: Input of the layer.
        state: The current state or `None` if there is none.
        kwargs: Keyword arguments (e.g. `key` or `inference`) passed on to the
            layer in both cases.

    Returns:
        The output of the layer and the (updated) state. A stateless layer
        returns the state unchanged.

    Examples:
        ```python
        import equinox as eqx
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.models.utils import apply_layer

        key = jr.key(0)
        x = jnp.ones((3, 8))

        # Stateless layer: state (here `None`) is passed through
        linear = eqx.nn.Linear(8, 4, key=key)
        out, state = apply_layer(linear, x, None)
        ```
    """
    if isinstance(layer, eqx.nn.StatefulLayer) and layer.is_stateful():
        return layer(x, state, **kwargs)
    return layer(x, **kwargs), state
