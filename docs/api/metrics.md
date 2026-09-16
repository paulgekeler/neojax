# Metric Reference

Training Neural Operators effectively requires evaluating the error in function space rather than just on discrete grid points. Standard ML metrics (like basic MSE) are discretization-dependent. To ensure our models are resolution-invariant and respect the underlying continuous functions, `neojax` provides discretization-invariant, norm-based metrics.

## Lp-Metrics

The standard approach for training Neural Operators is using continuous $L^p$ norms. 

- **`LpMetric`**: Computes the standard $L^p$ norm of the error over the spatial domain.
- **`RelativeLpMetric`**: Computes the $L^p$ error normalized by the $L^p$ norm of the true target function. This is standard practice in operator learning because physical data often ranges across different magnitudes, and a relative metric ensures stable gradients across samples.

::: neojax.metrics.lp_metrics.LpMetric

::: neojax.metrics.lp_metrics.RelativeLpMetric

## Sobolev Metrics

While $L^p$ metrics only measure the error in the predicted function values, **Sobolev metrics** ($H^1$, $W^{k,p}$) also penalize errors in the derivatives of the functions. By aligning the gradients of the prediction with the true gradients, the network is forced to learn the underlying physical differential operators more accurately.

### Implementation Details and Efficient Usage

Computing (higher order) derivatives in AD requires consideration of input and output dimensions. Choosing the wrong method or differentiation mode can determine whether training is efficient or fails due to Out-of-Memory (OOM) issues or excessive runtimes.

Throughout this section, we assume a model mapping inputs to outputs:
$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m
$$
where $n$ is the input dimension (e.g., number of input channels times spatial grid size) and $m$ is the output dimension.

`neojax` supports three primary methods for computing the derivatives, chosen via the `method` argument:

#### 1. Exact Mode (`method="exact"`, `k==1`)
This method computes the exact Jacobian of the model $f(x)$ with respect to $x$ using standard first-order AD.

- **Inner AD Mode (`diff_mode`)**:
    * **Forward-mode (`fwd`)**: Computes Jacobian column-by-column, scaling with the input dimension $n$.
    * **Reverse-mode (`bwd`)**: Computes Jacobian row-by-row, scaling with the output dimension $m$.
    * **Auto (`auto`)**: `neojax` uses forward-mode AD if $n \le m$, and backward-mode AD if $n > m$.

*Even when $n \approx m$, `diff_mode="fwd"` is typically superior due to its memory efficiency.*

#### 2. Interpolated Mode (`method="interpolate"`, `k > 1`)
For higher-order derivatives ($k \ge 2$), `neojax` uses JAX's Taylor-mode AD (`jax.experimental.jet`), which propagates truncated Taylor polynomials.

- The interpolated method evaluates the Taylor expansion of $f(x)$ in all $n$ canonical basis directions.
- This computes the exact higher-order derivatives, but requires a `jax.vmap` loop over $n$ directions.
- It is efficient for moderate input dimensions ($n$), but becomes expensive as the spatial resolution or input channel count increases.

#### 3. Stochastic Mode (`method="stochastic"`)
When the input dimension $n$ or output dimension $m$ (in case of `diff_mode="bwd"`) is very large (e.g., high-resolution grids),
computing the exact or interpolated derivatives is intractable. 

- The stochastic method uses an unbiased estimator of the norm instead of the exact norm.
- Instead of computing derivatives along every basis direction, it samples a small number of random basis vectors/tensors and evaluates the derivatives only in those random directions.

#### Auto Routing (`method="auto"`)
If you set `method="auto"`, `neojax` automatically routes to the most suitable strategy:

- **$k = 1$**: Uses `"exact"` if the input or output dimensions are small (configurable) or `"stochastic"` for large grids (random subsampling of basis vectors).
- **$k \ge 2$**: Uses `"interpolate"` if the input dimensions are small (configurable), otherwise `"stochastic"` (using STDE; random subsampling of basis tensors).

---

::: neojax.metrics.sobolev_metrics.SobolevMetric

---

## Physical Metrics

::: neojax.metrics.physical_metrics.BoundaryConsistencyMetric

---

::: neojax.metrics.physical_metrics.ConservationMetric

---

::: neojax.metrics.physical_metrics.ResidualMetric

## Regression Metrics

::: neojax.metrics.regression_metrics.MSEMetric

---

::: neojax.metrics.regression_metrics.RMSEMetric

---

::: neojax.metrics.regression_metrics.R2Metric

## Composing Metrics

Real-world physics problems often require minimizing multiple objectives simultaneously (e.g., combining a data metric with a physics-informed regularization term). `ComposedMetric` allows you to pipe multiple metric functions together, computing the weighted sum of its constituent metrics.

::: neojax.metrics.composed_metric.ComposedMetric

### Learnable Metric Weights

When combining multiple metrics, balancing their static weights can be notoriously difficult. `neojax` solves this by supporting dynamically learnable metric weights. 

If you enable learnable weights in a `ComposedMetric`, custom `BaseMetric`, or any other metric instance, you must ensure JAX differentiates with respect to them. `neojax` provides the `is_learnable_metric_weight` utility to create the correct filter specification for the metric functions.

::: neojax.metrics.utils.is_learnable_metric_weight

## Custom Metrics

When implementing custom metrics, inherit from `BaseMetric`.

Each custom metric should implement the `__call__` method, taking the predicted array and the target array as inputs and returning a scalar metric.

::: neojax.metrics.base_metric.BaseMetric