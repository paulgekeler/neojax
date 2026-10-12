## General

### Shared Model Functionality

All models inherit useful general functionality from `BaseNO` such as:

- `size()`: Get the model size in MB.
- `astype()`: Cast model weights to a different datatype.
- `profile_compile()`: Try to jit-compile the model and get a compilation summary.
- `load()`: Load model and hyperparameters.
- `save()`: Save model and hyperparameters.
- `load_weights()`: Load only the model weights.
- `save_weights()`: Save only the model weights.

### Improved Parameter Naming (FNO & TFNO)

If you are migrating from the original PyTorch `neuraloperator` library, you will notice that `neojax` introduces different parameter naming convention for residual connections and skip connections. The original library relies on ambiguous parameter names (like `fno_skip`). 

To provide a clean, modern API, `neojax` enforces the following standard across all FNO and TFNO architectures:

- Any parameters containing the word `...local_operator...` refer to the local operator.
- Any parameters containing the word `...residual...` refer to Resnet-style residual connections around components.

### Stochastic Layers & PRNG Key Management

In `neojax`, models that use stochastic layers (such as dropout) require a JAX PRNG key to generate random masks during training. Because JAX is purely functional, key propagation must be handled explicitly:

* **Training Mode**: To apply stochastic behavior (e.g., dropout), pass a `jax.random.PRNGKey` to the model call via the `key` argument, and set `inference=False`.
* **Inference Mode**: To compute the deterministic forward pass, pass `inference=True` (or omit the `key` argument). When `inference=True`, the model operates in evaluation mode and disables all dropout masking.

#### Example: Running FNO with Dropout

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