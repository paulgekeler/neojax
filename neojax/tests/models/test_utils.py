import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.models.utils import apply_layer
from neojax.nn.cno_blocks import CNOConvBlock


class TestApplyLayer:
    def test_stateless_layer(self):
        linear = eqx.nn.Linear(4, 3, key=jr.key(0))
        x = jr.normal(jr.key(1), (4,))
        out, state = apply_layer(linear, x, None)
        np.testing.assert_array_equal(out, linear(x))
        assert state is None

    def test_stateless_layer_passes_state_through(self):
        _, state = eqx.nn.make_with_state(
            lambda: eqx.nn.BatchNorm(2, "b", mode="batch")
        )()
        linear = eqx.nn.Linear(4, 3, key=jr.key(0))
        _, out_state = apply_layer(linear, jnp.ones(4), state)
        assert out_state is state

    def test_stateful_layer_updates_state(self):
        norm, state = eqx.nn.make_with_state(
            lambda: eqx.nn.BatchNorm(2, "batch", mode="batch")
        )()
        x = jr.normal(jr.key(1), (4, 2, 8))
        out, new_state = jax.vmap(
            lambda xi, s: apply_layer(norm, xi, s),
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )(x, state)
        assert out.shape == x.shape
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree_util.tree_leaves(state),
                jax.tree_util.tree_leaves(new_state),
                strict=True,
            )
        )

    def test_kwargs_are_forwarded(self):
        norm, state = eqx.nn.make_with_state(
            lambda: eqx.nn.BatchNorm(2, "batch", mode="batch")
        )()
        x = jr.normal(jr.key(1), (4, 2, 8))
        _, inf_state = jax.vmap(
            lambda xi, s: apply_layer(norm, xi, s, inference=True),
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )(x, state)
        for a, b in zip(
            jax.tree_util.tree_leaves(state),
            jax.tree_util.tree_leaves(inf_state),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        # also for stateless layers
        linear = eqx.nn.Linear(4, 3, key=jr.key(0))
        out, _ = apply_layer(linear, jnp.ones(4), None, key=jr.key(1))
        assert out.shape == (3,)

    def test_maybe_stateful_layer_follows_is_stateful(self):
        # a `StatefulLayer` that is not stateful is called without state
        kwargs = {"in_channels": 2, "out_channels": 4, "ndim": 2}
        stateless = CNOConvBlock(key=jr.key(0), batch_norm=False, **kwargs)
        x = jr.normal(jr.key(1), (2, 8, 8))
        out, state = apply_layer(stateless, x, None)
        np.testing.assert_array_equal(out, stateless(x))
        assert state is None

        stateful, init_state = eqx.nn.make_with_state(CNOConvBlock)(
            key=jr.key(0), batch_norm=True, **kwargs
        )
        xs = jr.normal(jr.key(1), (4, 2, 8, 8))
        outs, new_state = jax.vmap(
            lambda xi, s: apply_layer(stateful, xi, s),
            axis_name="batch",
            in_axes=(0, None),
            out_axes=(0, None),
        )(xs, init_state)
        assert outs.shape == (4, 4, 8, 8)
        assert isinstance(new_state, eqx.nn.State)

    def test_stateful_layer_without_state_raises(self):
        stateful, _ = eqx.nn.make_with_state(CNOConvBlock)(
            key=jr.key(0),
            in_channels=2,
            out_channels=4,
            ndim=2,
            batch_norm=True,
        )
        with pytest.raises(ValueError):
            apply_layer(stateful, jnp.ones((2, 8, 8)), None)
