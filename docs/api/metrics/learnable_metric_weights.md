## Learnable Metric Weights

When combining multiple metrics, balancing their static weights can be notoriously difficult. `neojax` solves this by supporting dynamically learnable metric weights. 

If you enable learnable weights in a `ComposedMetric`, custom `BaseMetric`, or any other metric instance, you must ensure JAX differentiates with respect to them. `neojax` provides the `is_learnable_metric_weight` utility to create the correct filter specification for the metric functions.

::: neojax.metrics.utils.is_learnable_metric_weight