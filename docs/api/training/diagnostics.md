# Diagnostics Reference

`neojax.training.diagnostics` provides plain functions over a gradient PyTree (as produced by `eqx.filter_grad` or `Trainer.train_step`) for catching common training pathologies:

- A parameter receiving no gradient (e.g. from a stray `static=True` field, or an over-broad `filter_spec`).
- Or an exploding gradient.

There's no callback/hook system here by design: JAX transformations require pure functions, so side effects (printing, logging, raising) can't be cleanly embedded inside a jitted training step without `jax.debug.callback` host-transfer overhead.
Instead, these functions expose the data as plain PyTrees/dicts and leave what to do with it to your own (non-jitted) training loop. Any custom check is just your own function over the same data, no plugin API required.

## Usage

```python
from neojax.training import Trainer, compute_grad_report, select_by_path

trainer = Trainer(optimizer=optimizer, loss_fn=loss_fn)
state = trainer.create_train_state(model)

# Ad hoc, against a live or checkpointed TrainState
# computes gradients for one batch and flags anomalous norms.
report = compute_grad_report(trainer, state, batch, min_norm=1e-8, max_norm=1e4)
if report:
    print("Anomalous gradients:", report)

# Narrow down to one part of a large model.
select_by_path(report, "*fno_blocks*")
```

`compute_grad_report` is meant for ad hoc checks: run it every N steps, or point it at a checkpointed `TrainState`, rather than on every step. `Trainer.train_step` itself doesn't expose gradients, by design, so there's no way to fold this into the hot training loop without computing gradients a second time.
`grad_norms` is pure and jit/vmap-transparent, unlike `flag_anomalous_norms`, which filters by the gradients' actual values and so can only run outside `jit`.
So if you do want it on every step, call it directly on the result of your own `eqx.filter_value_and_grad`, before handing the gradients to the optimizer.

---

::: neojax.training.diagnostics.grad_norms

::: neojax.training.diagnostics.flag_anomalous_norms

::: neojax.training.diagnostics.compute_grad_report

::: neojax.training.diagnostics.select_by_path
