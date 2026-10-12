## Lp-Metrics

The standard approach for training Neural Operators is using continuous $L^p$ norms. 

- **`LpMetric`**: Computes the standard $L^p$ norm of the error over the spatial domain.
- **`RelativeLpMetric`**: Computes the $L^p$ error normalized by the $L^p$ norm of the true target function. This is standard practice in operator learning because physical data often ranges across different magnitudes, and a relative metric ensures stable gradients across samples.

::: neojax.metrics.lp_metrics.LpMetric

---

::: neojax.metrics.lp_metrics.RelativeLpMetric