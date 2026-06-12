### Example 1: Getting familiar with **neojax**

[:material-download: Download Notebook](01_basics.ipynb){ .md-button }

In this notebook we introduce the basic concepts of working with models in **neojax**.

We also provide some examples how training in **equinox**, and therefore **neojax** differs from **PyTorch**. For a more in depth introduction to **equinox**, please refer to [its documentation](https://docs.kidger.site/equinox/). All **equinox** and **Jax** concepts apply equally to **neojax**.

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
Wrapping a model in `jax.vmap` then vectorizes the model call over a batch.  
This is exactly how **equinox** works as well.

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
Computing gradients with respect to model weights is also the same as in **equinox**.


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


#### Saving and Loading
Serialising and deserialising weights from disk is easy with **equinox**.


```python
# Save to disk
eqx.tree_serialise_leaves("fno.eqx", model)

# Load into an existing model structure
loaded_model = eqx.tree_deserialise_leaves("fno.eqx", model)
```


#### Model Inspection
Neural Operators can get quite large, so here is a convenient **equinox** way to count the paramters of a model.


```python
params = eqx.filter(model, eqx.is_array)
num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total parameters: {num_params:,}")
```

**Output:**
```text
Total parameters: 537,313
```

