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

Computing derivatives in Automatic Differentiation (AD) for Neural Operator training requires careful consideration of input and output dimensions. Choosing the wrong method or differentiation mode can determine whether training is highly efficient or fails due to Out-of-Memory (OOM) issues or excessive runtimes.

Throughout this section, we assume a model mapping inputs to outputs:
$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m
$$
where $n$ is the input dimension (e.g., number of input channels times spatial grid size) and $m$ is the output dimension.

`neojax` supports three primary methods for computing the derivatives, chosen via the `method` argument:

#### 1. Exact Mode (`method="exact"`)
This method computes the exact Jacobian of the model $f(x)$ with respect to $x$ using standard first-order AD.

- **Inner AD Mode (`diff_mode`)**:
    * **Forward-mode (`fwd`)**: Computes derivatives column-by-column, scaling with the input dimension $n$.
    * **Reverse-mode (`bwd`)**: Computes derivatives row-by-row, scaling with the output dimension $m$.
    * **Auto (`auto`)**: `neojax` automatically uses forward-mode AD if $n \le m$, and backward-mode AD if $n > m$.
- **Compounding Outer-Pass Effect**:
  During model training, the outer optimizer loop takes the gradient of the scalar metric with respect to the model parameters $\theta$ (using a backward pass, e.g. `jax.grad`).
    * If the inner mode is **Forward** (`fwd`), JAX performs **Reverse-over-Forward** differentiation, which is highly memory-efficient because it does not store a nested backward tape.
    * If the inner mode is **Reverse** (`bwd`), JAX performs **Reverse-over-Reverse** differentiation, which requires storing nested Wengert tapes for both passes, drastically increasing the memory footprint.
  
*Tip: Even when $n \approx m$, `diff_mode="fwd"` is therefore often superior due to its memory efficiency.*

#### 2. Interpolated Mode (`method="interpolate"`)
For higher-order derivatives ($k \ge 2$), computing the exact Jacobian recursively scales exponentially. Instead, `neojax` uses JAX's Taylor-mode AD (`jax.experimental.jet`), which propagates truncated Taylor polynomials.

- The interpolated method evaluates the Taylor expansion of $f(x)$ in all $n$ canonical basis directions.
- This computes the exact higher-order derivatives, but requires a `jax.vmap` loop over $n$ directions.
- It is highly efficient for moderate input dimensions ($n$), but becomes expensive as the spatial resolution or input channel count increases.

#### 3. Stochastic Mode (`method="stochastic"`)
When the input dimension $n$ is very large (e.g., high-resolution grids), computing the Taylor polynomial in all $n$ directions is intractable. 

- The stochastic method uses the **Stochastic Taylor Derivative Estimator (STDE)**.
- Instead of computing derivatives along every canonical basis, it samples a small number of random direction vectors (from a Rademacher or Normal distribution) and evaluates the Taylor expansion only in those random directions.
- The computational cost is independent of the input dimension $n$ and scales only with the number of random samples (`n_random_samples`), which defaults to 10.

#### Auto Routing (`method="auto"`)
If you set `method="auto"`, `neojax` automatically routes to the most suitable strategy:

- **$k = 1$**: Uses `"exact"`.
- **$k \ge 2$**: Uses `"interpolate"` if all spatial grid dimensions of the input $x$ are $\le 10$, otherwise defaults to `"stochastic"` to prevent OOM and runtime issues.

::: neojax.metrics.sobolev_metrics.SobolevMetric

## Physical Metrics

::: neojax.metrics.physical_metrics.BoundaryConsistencyMetric

::: neojax.metrics.physical_metrics.ConservationMetric

::: neojax.metrics.physical_metrics.ResidualMetric

## Regression Metrics

::: neojax.metrics.regression_metrics.MSEMetric

::: neojax.metrics.regression_metrics.RMSEMetric

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