## Composing Metrics

Real-world physics problems often require minimizing multiple objectives simultaneously (e.g., combining a data metric with a physics-informed regularization term). `ComposedMetric` allows you to pipe multiple metric functions together, computing the weighted sum of its constituent metrics.

::: neojax.metrics.composed_metric.ComposedMetric