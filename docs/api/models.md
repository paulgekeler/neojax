# Models Reference

This page contains the API reference for all pre-built models in `neojax`. Currently, the library supports the Fourier Neural Operator (FNO), Tucker-factorized FNO (TFNO), and Deep Operator Networks (DeepONet).

## Improved Parameter Naming (FNO & TFNO)

If you are migrating from the original PyTorch `neuraloperator` library, you will notice that `neojax` introduces different parameter naming convention for residual connections and skip connections. The original library relies on ambiguous parameter names (like `fno_skip`). 

To provide a clean, modern API, `neojax` enforces the following standard across all FNO and TFNO architectures:

- Any parameters containing the word `...local_operator...` refer to the local operator.
- Any parameters containing the word `...residual...` refer to Resnet-style residual connections around components.

## Fourier Neural Operator (FNO)

::: neojax.models.fno.FNO

## Tucker-factorized FNO (TFNO)

::: neojax.models.tfno.TFNO

## Deep Operator Networks (DeepONet)

**neojax** implements a general modular `DeepONet` (arbitrary branch and trunk networks) and a preconfigured `MLPDeepONet` (MLP branch and MLP trunk network).

The DeepONets are implemented to evaluate the underlying operator with input function $u \in \mathbb{R}^m$ at the output $y \in \mathbb{R}^d$. In case of higher dimensional input spaces, e.g. $\mathbb{R}^2, \mathbb{R}^3, \dots$ the sensor locations `m` may not be 1D arrays, as long as the branch network handles this correctly.

In practice, it is more useful to evaluate the output across a grid of points `y`. See below for details.

**How to evaluate the input function across a grid of points**: To evaluate a single input
function $u$ across a grid (or batch) of points $y$, use `jax.vmap`. Here is how you would
evaluate on a $256 \times 256$ grid:

```python
import jax
import jax.numpy as jnp
from neojax.models import DeepONet

# Initialize model
model = DeepONet(...)

# Define inputs
u = jnp.ones((100,))  # One function input
y_grid = jnp.meshgrid(jnp.linspace(0, 1, 256), jnp.linspace(0, 1, 256))
y_points = jnp.stack(y_grid, axis=-1)  # Shape: (256, 256, 2)

# Vectorize the model over the coordinate axes
# in_axes: (None, 0) means 'u' is fixed, 'y' is mapped over its 0-th dimension
vmapped_inner = jax.vmap(model, in_axes=(None, 0))          # Maps (256, 2) -> (256, 1)
vmapped_outer = jax.vmap(vmapped_inner, in_axes=(None, 0))  # Maps (256, 256, 2) -> (256, 256, 1)

# Generate predictions
predictions = vmapped_outer(u, y_points)  # Shape: (256, 256, 1)
```

**Pro Tip**: If you want to evaluate a batch of functions across a batch of points,
just add a third `jax.vmap` call.

```python
# Batch of 32 functions, each evaluated at 256 points
u_batch = jnp.ones((32, 100))
y_batch = jnp.zeros((32, 256, 2))

# vmap over functions (axis 0) and points (axis 0)
batch_model = jax.vmap(vmapped_inner, in_axes=(0, 0))
batch_preds = batch_model(u_batch, y_batch) # Shape: (32, 256, 1)
```

### General DeepONet

::: neojax.models.deeponet.DeepONet

### MLPDeepONet

::: neojax.models.deeponet.MLPDeepONet