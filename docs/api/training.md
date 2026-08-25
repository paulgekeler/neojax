# Training API Reference

This page contains the API reference for the training utilities in `neojax` designed to orchestrate functional optimization loops under the JAX/Equinox paradigm.

## Functional Training Orchestration

In `neojax`, the `Trainer` class encapsulates optimization steps while keeping the state explicit via `TrainState`. Because JAX expects pure functions, `TrainState` is an immutable PyTree that holds the model, the optimizer state, and static metadata (such as epoch or batch logs).

### Example: Custom Optimization Loop

Below is a complete example showing how to initialize `Trainer`, set up the training state, and perform JIT-compiled optimization steps:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from neojax.models import FNO
from neojax.metrics import RelativeLpMetric
from neojax.training import Trainer, TrainState

# Instantiate model, optimizer, and loss metric
key = jr.key(0)
model = FNO(
    key=key,
    in_channels=2,
    out_channels=1,
    hidden_channels=32,
    n_layers=2,
    modes=(16, 16),
)
optimizer = optax.adam(learning_rate=1e-3)
loss_metric = RelativeLpMetric(p=2.0)

# Define functional loss taking (model, batch_data)
def loss_fn(model, batch):
    x, y = batch  # x shape: (batch, channels, *spatial), y shape: (batch, channels, *spatial)
    predictions = jax.vmap(model)(x)
    return loss_metric(pred=predictions, target=y)

# Create trainer and training state
trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn)
state = trainer.create_train_state(model, metadata={"epoch": 0})

# Perform functional updates
dummy_batch = (jnp.ones((10, 2, 64, 64)), jnp.ones((10, 1, 64, 64)))
new_state, loss_val = trainer.train_step(state, dummy_batch)

print(f"Step: {new_state.step}, Loss: {loss_val}")
```

### Example: Resuming Training Checkpoints

`TrainState` provides a simple way to save and restore all training variables (including model weights, optimizer history, and step metadata) to/from disk.

```python
# Save current state (includes model parameters, adam states, step, and custom metadata)
model_hyperparams = {
    "in_channels": 2,
    "out_channels": 1,
    "hidden_channels": 32,
    "n_layers": 2,
    "modes": (16, 16),
}
state.save("checkpoint.bin", model_hyperparams)

# Load state and continue training using a constructor make_fn
def make_fn(in_channels, out_channels, hidden_channels, n_layers, modes, key=None):
    if key is None:
        key = jr.key(0)
    return FNO(
        key=key,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        modes=modes,
    )

loaded_state = TrainState.load("checkpoint.bin", make_fn, optimizer)
print(f"Resumed at step: {loaded_state.step}")
```

---

## Trainer Class

::: neojax.training.Trainer

---

## TrainState PyTree

::: neojax.training.TrainState
