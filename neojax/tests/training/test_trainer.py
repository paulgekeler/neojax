import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from neojax.metrics import RelativeLpMetric
from neojax.metrics.utils import is_learnable_metric_weight
from neojax.tests.conftest import assert_filter_jittable
from neojax.training.trainer import Trainer, TrainState


def test_trainer_and_train_state():
    key = jr.key(0)

    # Initialize a simple MLP model skeleton and optimizer
    model = eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=1, key=key)
    optimizer = optax.adam(1e-2)

    # Define a simple MSE loss function
    def loss_fn(model, batch):
        x, y = batch
        preds = jax.vmap(model)(x)
        return jnp.mean((preds - y) ** 2)

    trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn)

    # Verify state creation
    state = trainer.create_train_state(model, metadata={"epoch": 0, "batch_idx": 0})
    assert state.step == 0
    assert state.metadata == {"epoch": 0, "batch_idx": 0}

    # Verify single JIT training step
    x = jr.normal(key, (10, 2))
    y = jr.normal(key, (10, 1))
    batch = (x, y)

    state_next, loss_val = trainer.train_step(state, batch)
    assert state_next.step == 1
    assert state_next.metadata == {"epoch": 0, "batch_idx": 0}
    assert loss_val > 0

    assert_filter_jittable(lambda s: trainer.train_step(s, batch)[1], state)

    # Verify save and load checkpoint roundtrip
    model_hyperparams = {"in_size": 2, "out_size": 1, "width_size": 4, "depth": 1}

    def make_fn(in_size, out_size, width_size, depth, key=None):
        if key is None:
            key = jr.key(0)
        return eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.bin"

        state_next.save(checkpoint_path, model_hyperparams)

        loaded_state = TrainState.load(checkpoint_path, make_fn, optimizer)

        assert loaded_state.step == 1
        assert loaded_state.metadata["step"] == 1
        assert loaded_state.metadata["epoch"] == 0
        assert loaded_state.metadata["batch_idx"] == 0

        original_leaves = jax.tree_util.tree_leaves(
            eqx.filter(state_next.model, eqx.is_array)
        )
        loaded_leaves = jax.tree_util.tree_leaves(
            eqx.filter(loaded_state.model, eqx.is_array)
        )
        for o, l in zip(original_leaves, loaded_leaves, strict=True):
            assert jnp.allclose(o, l)

        # Verify loaded optimizer states match
        original_opt_leaves = jax.tree_util.tree_leaves(
            eqx.filter(state_next.opt_state, eqx.is_array)
        )
        loaded_opt_leaves = jax.tree_util.tree_leaves(
            eqx.filter(loaded_state.opt_state, eqx.is_array)
        )
        for o, l in zip(original_opt_leaves, loaded_opt_leaves, strict=True):
            assert jnp.allclose(o, l)


def test_trainer_with_custom_filter_spec():
    key = jr.key(0)
    model = eqx.nn.MLP(in_size=2, out_size=1, width_size=4, depth=1, key=key)
    metric = RelativeLpMetric(p=2.0, weight=0.5, learnable_weight=True)

    class Combined(eqx.Module):
        model: eqx.nn.MLP
        loss_metric: RelativeLpMetric

    combined = Combined(model, metric)

    # Custom filter spec: model parameters + learnable metric weights
    filter_spec = Combined(
        model=jax.tree_util.tree_map(eqx.is_inexact_array, model),
        loss_metric=is_learnable_metric_weight(metric),
    )

    optimizer = optax.adam(1e-2)

    def loss_fn(combined_mod, batch):
        x, y = batch
        preds = jax.vmap(combined_mod.model)(x)
        return combined_mod.loss_metric(pred=preds, target=y)

    trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn, filter_spec=filter_spec)
    state = trainer.create_train_state(combined)

    x = jr.normal(key, (10, 2))
    y = jr.normal(key, (10, 1))
    batch = (x, y)

    # Perform a step and assert that the metric weight was updated
    initial_weight = state.model.loss_metric.weight
    state_next, loss_val = trainer.train_step(state, batch)
    updated_weight = state_next.model.loss_metric.weight

    assert not jnp.allclose(initial_weight, updated_weight)
