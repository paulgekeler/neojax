# Normalizer Reference

Data normalization is essential for the stable and efficient training of Neural Operators. Skewed training data distributions heavily influence model performance and training, especially for physical data which may range across vastly different scales. **neojax** provides multiple normalizers for reliable training. 

All normalizers share a common API, inherit from `BaseNormalizer`, and can be arbitrarily composed to build complex normalization pipelines.

## The Normalizer API

We adopt the common naming conventions of scientific computing libraries. Every normalizer implements:

- **`compute_stats(data, axis=...)`**: 'Learns' the normalization statistics (e.g., min/max, mean/std) from the given data. Because `neojax` uses `equinox` and is strictly functional, this returns a **new** normalizer instance with the updated statistics.
- **`transform(x)`** / **`__call__(x)`**: Applies the normalization to the input.
- **`inverse_transform(x)`**: Reverses the normalization, projecting predictions back into the original physical space.

### Example Usage

```python
import jax.numpy as jnp
from neojax.data.normalizers import MinMaxNormalizer

# 1. Initialize the normalizer
normalizer = MinMaxNormalizer()

# 2. Learn statistics from training data
train_data = jnp.array([[-10.0, 0.0, 10.0]])
normalizer = normalizer.compute_stats(train_data)

# 3. Transform data before passing to the model
normalized_data = normalizer(train_data) 
# normalized_data is now scaled to [0, 1]

# 4. Revert model predictions back to the original scale
predictions = model(normalized_data)
physical_predictions = normalizer.inverse_transform(predictions)
```

## Composing Normalizers

Data pipelines often require multiple transformation steps. The `ComposedNormalizer` allows you to pipe normalizers together. 

By default, `compute_stats` works **sequentially** through the pipeline: the second normalizer computes its statistics on the data *after* it has been transformed by the first normalizer. You can disable this by passing `sequential=False`.

```python
from neojax.data.normalizers import ComposedNormalizer, RobustNormalizer, MinMaxNormalizer

# Build a pipeline
pipeline = ComposedNormalizer(
    RobustNormalizer(),
    MinMaxNormalizer()
)

# Computes robust stats, transforms data, then computes min/max stats on the output
pipeline = pipeline.compute_stats(data)
```

## Base Normalizer

:::neojax.data.normalizers.base_normalizer.BaseNormalizer

## Unit Gaussian Normalizer

:::neojax.data.normalizers.unit_gaussian_normalizer.UnitGaussianNormalizer

## Robust Normalizer

::: neojax.data.normalizers.robust_normalizer.RobustNormalizer

## Min/Max Normalizer

*(Supports both `"scale"` and destructive `"clip"` modes)*
::: neojax.data.normalizers.min_max_normalizer.MinMaxNormalizer

## Physics Normalizer

*(Non-dimensionalizes data using physical scales. See Scales Reference.)*
::: neojax.data.normalizers.physics_normalizer.PhysicsNormalizer

## Composed Normalizer

::: neojax.data.normalizers.composed_normalizer.ComposedNormalizer