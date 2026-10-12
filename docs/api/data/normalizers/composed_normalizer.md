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

## Composed Normalizer

::: neojax.data.normalizers.composed_normalizer.ComposedNormalizer