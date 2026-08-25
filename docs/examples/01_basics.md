### Example 1: Getting familiar with **neojax**

[:material-download: Download Notebook](01_basics.ipynb){ .md-button }

In this notebook we introduce the basic concepts of working with models in **neojax**.

We also provide some examples how training in **Equinox**, and therefore **neojax** differs from **PyTorch**. For a more in depth introduction to **Equinox**, please refer to [its documentation](https://docs.kidger.site/equinox/). All **Equinox** and **JAX** concepts apply equally to **neojax**.

!!! info
    This tutorial doesn't introduce advanced concepts such as `Schemas`, `DataBundles` or the `BundleProcessor`. Have a look [here](03_data_pipeline.md) for an example on how to use them.

First we install and import the needed packages.


```python
!pip3 install equinox neojax-operators
```


```python
import jax
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp

from neojax.models import FNO
```


#### Simple 2D Forward Pass
We first demonstrate a basic 2D forward pass with a Fourier Neural Operator (FNO). The logic should be familiar coming from other libraries and frameworks.

Input arrays are passed **channel-first** to the models.


```python
# jax randomness works differently than Pytorch or Numpy
# See here: https://docs.jax.dev/en/latest/jax.random.html
# We need to set a deterministic seed and pass it to models for random initialization of weights.
key = jr.key(seed=0)

# Initialize a 2D FNO
model = FNO(
    key=key,
    in_channels=3,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
    modes=(8, 8)
)

# Create 2D dummy input (channels, height, width)
x = jnp.ones((3, 64, 64))

# Forward pass
out = model(x)
print(f"Output shape: {out.shape}") # (1, 64, 64)
```

**Output:**
```text
Output shape: (1, 64, 64)
```


#### Handling Batches
After seeing the example above, you might be wondering why we didn't add a leading batch dimension to the input array.  
Thanks to `jax.vmap` vectorization, all models and functions can be written for single samples.  
Wrapping a model in a `jax.vmap` call then vectorizes the model forward pass over a batch.  
This is exactly how **Equinox** works as well.

Here is a forward pass with a batch of inputs:


```python
# Batch of 4 samples
batch_x = jnp.ones((4, 3, 64, 64))

# Apply model to entire batch
batch_out = jax.vmap(model)(batch_x)
print(f"Batch output: {batch_out.shape}") # (4, 1, 64, 64)
```

**Output:**
```text
Batch output: (4, 1, 64, 64)
```

#### Computing Gradients
Computing gradients with respect to model weights is also the same as in **Equinox**.


```python
def loss_fn(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y)**2)

# Filter ensures we only take the gradients wrt the traceable arrays
loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, out)

# pred is equal to out so the loss will be zero
print(f"Loss: {loss}")
```

**Output:**
```text
Loss: 0.0
```

!!! info
    If the `equinox.filter_...` syntax is new to you, we suggest looking into the **Equinox** [documentation](https://docs.kidger.site/equinox/all-of-equinox/).

#### Saving and Loading

Each **neojax** model has four convenience methods for serialisation, wrapping `equinox.{tree_deserialise_leaves,tree_serialise_leaves}` under the hood.

```python
from functools import partial

# Save only model weights
weights_save_pth = "./weights"
model.save_weights(weights_save_pth)

# Load only model weights
new_model = model.load_weights(weights_save_pth)

# Save model and hyperparameters to disk
hyperparams = dict(
    in_channels=3,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
    modes=(8, 8))

model_save_pth = "./model.eqx"
model.save(file=model_save_pth, hyperparams=hyperparams)

# The model creation function may define arbitrary logic
creation_fn = lambda key, **kwargs: FNO(key=key, **kwargs)

# We pass the random key via 'ft.partial' here as an example
key = jr.key(0)
partial_cfn = partial(creation_fn, key=key)

# Load model into an existing model structure
new_model = FNO.load(file=model_save_pth, make_fn=partial_cfn)
# Alternatively:
# new_model = FNO.load(file=model_save_pth, make_fn=creation_fn, key=key)

# Make sure we didn't mess up
assert new_model == model
```

You may have noticed that `load` is a class method. This is a common pattern in **neojax**,
whenever attributes need to be mutated.

Because **Equinox** models and therefore also **neojax** models (and most of its components) are immutable, all attributes are immutable after an object has been instantiated. In order to update
any attributes, methods that perform out-of-place mutation, return new instances.

#### Model Inspection

Each **neojax** model provides a few handy utilities for inspection:

1. Get the size of a model
    ```python
    size_in_mb, n_params = model.size(return_n_params=True)
    print(f"Model size: {size_in_mb} MB; Num params: {n_params}")
    ```
    **Output:**
    ```bash
    Model size: 4.247428 MB; Num params: 537569
    ```
2. Cast a model to a different datatype. May be useful to speedup training and decrease memory footprint
    ```python
    bfloat_model = model.astype(jnp.bfloat16)
    ```

3. Compile the XLA-graph of the model ahead of time to assert the model compiles and returns a summary including the approximate FLOPS of the compiled model and formatted exceptions if compilation failed.
    ```python
    import pprint

    dummy_input = jnp.ones((3, 64, 64))
    model_compiled, compilation_summary = model.profile_compile(dummy_input)
    print("Model compiled: ", model_compiled)
    print("Compilation summary: ")
    pprint.pprint(compilation_summary, sort_dicts=False)
    ```
    **Output:**
    ```bash
    Model compiled:  True
    Compilation summary: 
    {'lowering_cost_analysis': {'flops': 357511168.0,
                                'transcendentals': 1970180.0,
                                'bytes accessed': 382391616.0,
                                'utilization0{}': 545.48486328125,
                                'utilization1{}': 285.0,
                                'bytes accessed0{}': 124006928.0,
                                'bytes accessed1{}': 104087888.0,
                                'bytes accessedout{}': 154034736.0,
                                'utilization2{}': 20.0,
                                'bytes accessed2{}': 262160.0},
    'compiled_cost_analysis': {'utilization0{}': 131.0,
                                'utilization6{}': 8.0,
                                'bytes accessed8{}': 2097152.0,
                                'bytes accessed3{}': 4194368.0,
                                'utilization10{}': 4.0,
                                'bytes accessed10{}': 512.0,
                                'bytes accessed9{}': 2097152.0,
                                'bytes accessed2{}': 4195136.0,
                                'bytes accessed5{}': 2098176.0,
                                'utilization2{}': 29.0,
                                'transcendentals': 1970180.0,
                                'flops': 367742976.0,
                                'utilization8{}': 4.0,
                                'utilization9{}': 4.0,
                                'utilization4{}': 24.0,
                                'utilization7{}': 8.0,
                                'bytes accessed4{}': 4194368.0,
                                'bytes accessed1{}': 23593636.0,
                                'bytes accessedout{}': 34209972.0,
                                'utilization5{}': 12.0,
                                'utilization1{}': 112.0,
                                'bytes accessed6{}': 2097168.0,
                                'bytes accessed7{}': 528.0,
                                'bytes accessed0{}': 25774868.0,
                                'bytes accessed': 99965280.0,
                                'utilization3{}': 24.0},
    'exception': None,
    'lowering': ''}
    ```


Because **neojax** components and models are **Equinox** modules, they also work with **Equinox**'s other utilities, such as pretty-printing trees with `equinox.{tree_pprint,tree_pformat}(model)` and [more](https://docs.kidger.site/equinox/api/manipulation/).

For an introduction to model training, refer to the second [example](02_fno_burgers.md).