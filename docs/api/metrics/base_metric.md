## Base Metric

::: neojax.metrics.base_metric.BaseMetric

---

## Custom Metrics

When implementing custom metrics, inherit from `BaseMetric`.

Each custom metric should implement the `__call__` method, taking the predicted array and the target array as inputs and returning a scalar metric.
