## General

Physical data often spans widely different scales across fields, which affects training stability. `neojax` provides several normalizers to rescale data before it reaches the model.

All normalizers share a common API, inherit from `BaseNormalizer`, and can be composed to apply several transformations in sequence.

### The Normalizer API

We adopt the common naming conventions of scientific computing libraries. Every normalizer implements:

- **`compute_stats(data, axis=...)`**: 'Learns' the normalization statistics (e.g., min/max, mean/std) from the given data. Because `neojax` uses `equinox` and is strictly functional, this returns a **new** normalizer instance with the updated statistics.
- **`transform(x)`** / **`__call__(x)`**: Applies the normalization to the input.
- **`inverse_transform(x)`**: Reverses the normalization, projecting predictions back into the original physical space.

### Example Usage

```python
import jax.numpy as jnp
from neojax.data.normalizers import MinMaxNormalizer

# Initialize the normalizer
normalizer = MinMaxNormalizer()

# Learn statistics from training data
train_data = jnp.array([[-10.0, 0.0, 10.0]])
normalizer = normalizer.compute_stats(train_data)

# Transform data before passing to the model
normalized_data = normalizer(train_data) 
# normalized_data is now scaled to [0, 1]

# Revert model predictions back to the original scale
predictions = model(normalized_data)
physical_predictions = normalizer.inverse_transform(predictions)
```
