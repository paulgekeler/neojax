# Models Reference

This page contains the API reference for all pre-built models in `neojax`. Currently, the library supports:

- Fourier Neural Operator (FNO)
- Tucker-factorized FNO (TFNO)
- Deep Operator Networks (DeepONet)
- U-shaped Neural Operator (UNO)
- Geometry-aware FNO (GeoFNO)

## Shared Model Functionality

All models inherit useful general functionality from `BaseNO` such as:

- `size()`: Get the model size in MB.
- `astype()`: Cast model weights to a different datatype.
- `profile_compile()`: Try to jit-compile the model and get a compilation summary.
- `load()`: Load model and hyperparameters.
- `save()`: Save model and hyperparameters.
- `load_weights()`: Load only the model weights.
- `save_weights()`: Save only the model weights.

## Improved Parameter Naming (FNO & TFNO)

If you are migrating from the original PyTorch `neuraloperator` library, you will notice that `neojax` introduces different parameter naming convention for residual connections and skip connections. The original library relies on ambiguous parameter names (like `fno_skip`). 

To provide a clean, modern API, `neojax` enforces the following standard across all FNO and TFNO architectures:

- Any parameters containing the word `...local_operator...` refer to the local operator.
- Any parameters containing the word `...residual...` refer to Resnet-style residual connections around components.

## Stochastic Layers & PRNG Key Management

In `neojax`, models that use stochastic layers (such as dropout) require a JAX PRNG key to generate random masks during training. Because JAX is purely functional, key propagation must be handled explicitly:

* **Training Mode**: To apply stochastic behavior (e.g., dropout), pass a `jax.random.PRNGKey` to the model call via the `key` argument, and set `inference=False`.
* **Inference Mode**: To compute the deterministic forward pass, pass `inference=True` (or omit the `key` argument). When `inference=True`, the model operates in evaluation mode and disables all dropout masking.

### Example: Running FNO with Dropout

```python
import jax.numpy as jnp
import jax.random as jr
from neojax.models import FNO

key = jr.key(0)
model_key, train_key = jr.split(key)

# Initialize FNO with channel MLP dropout
model = FNO(
    key=model_key,
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
    modes=(16,),
    channel_mlp_dropout=0.1,
)

x = jnp.ones((1, 64))

# Training forward pass (stochastic, requires key)
out_train = model(x, key=train_key, inference=False)

# Inference forward pass (deterministic, no key needed)
out_eval = model(x, inference=True)
```

---

## Fourier Neural Operator (FNO)

::: neojax.models.fno.FNO

---

## Tucker-factorized FNO (TFNO)

::: neojax.models.tfno.TFNO

---

## Geometry-aware FNO (Geo-FNO)

The Geometry-aware Fourier Neural Operator (Geo-FNO) learns a coordinate deformation (diffeomorphism map) to map general/unstructured physical domains into a regular latent grid space.

::: neojax.models.geo_fno.GeoFNO

---

## U-shaped FNO (UNO)

::: neojax.models.uno.UNO

---

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

---

### General DeepONet

::: neojax.models.deeponet.DeepONet

---

### MLPDeepONet

::: neojax.models.deeponet.MLPDeepONet

---

## Custom Models

When implementing custom models, inherit from `BaseNO` to inherit useful functionality such as convenient saving and loading of models, etc.

::: neojax.models.baseno.BaseNO