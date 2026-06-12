# Loss Reference

Training Neural Operators effectively requires evaluating the error in function space rather than just on discrete grid points. Standard ML losses (like basic MSE) are discretization-dependent. To ensure our models are resolution-invariant and respect the underlying continuous functions, `neojax` provides discretization-invariant, norm-based losses.

## Lp-Losses

The standard approach for training Neural Operators is using continuous $L^p$ norms. 

- **`LpLoss`**: Computes the standard $L^p$ norm of the error over the spatial domain.
- **`RelativeLpLoss`**: Computes the $L^p$ error normalized by the $L^p$ norm of the true target function. This is standard practice in operator learning because physical data often ranges across different magnitudes, and a relative loss ensures stable gradients across samples.

::: neojax.losses.lp_losses.LpLoss

::: neojax.losses.lp_losses.RelativeLpLoss

## Sobolev Losses

While $L^p$ losses only measure the error in the predicted function values, **Sobolev losses** ($H^1$, $W^{k,p}$) also penalize errors in the derivatives of the functions. By aligning the gradients of the prediction with the true gradients, the network is forced to learn the underlying physical differential operators more accurately.

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
  - **Forward-mode (`fwd`)**: Computes derivatives column-by-column, scaling with the input dimension $n$.
  - **Reverse-mode (`bwd`)**: Computes derivatives row-by-row, scaling with the output dimension $m$.
  - **Auto (`auto`)**: `neojax` automatically uses forward-mode AD if $n \le m$, and backward-mode AD if $n > m$.
- **Compounding Outer-Pass Effect**:
  During model training, the outer optimizer loop takes the gradient of the scalar loss with respect to the model parameters $\theta$ (using a backward pass, e.g. `jax.grad`).
  - If the inner mode is **Forward** (`fwd`), JAX performs **Reverse-over-Forward** differentiation, which is highly memory-efficient because it does not store a nested backward tape.
  - If the inner mode is **Reverse** (`bwd`), JAX performs **Reverse-over-Reverse** differentiation, which requires storing nested Wengert tapes for both passes, drastically increasing the memory footprint.
  *Tip: Even when $n \approx m$, `diff_mode="fwd"` is often superior due to its memory efficiency.*

#### 2. Interpolated Mode (`method="interpolate"`)
For higher-order derivatives ($k \ge 2$), computing the exact Jacobian recursively scales exponentially. Instead, `neojax` uses JAX's Taylor-mode AD (`jax.experimental.jet`), which propagates truncated Taylor polynomials.
- The interpolated method evaluates the Taylor expansion of $f(x)$ in all $n$ canonical basis directions.
- This computes the exact higher-order derivatives, but requires a `jax.vmap` loop over $n$ directions.
- It is highly efficient for moderate input dimensions ($n$), but becomes expensive as the spatial resolution or input channel count increases.

#### 3. Stochastic Mode (`method="stochastic"`)
When the input dimension $n$ is very large (e.g., high-resolution grids), computing the Taylor polynomial in all $n$ directions is intractable. 
- The stochastic method uses the **Stochastic Taylor Derivative Estimator (STDE)**.
- Instead of computing derivatives along every canonical basis, it samples a small number of random direction vectors (Rademacher or Normal distribution) and evaluates the Taylor expansion only in those random directions.
- The computational cost is independent of the input dimension $n$ and scales only with the number of random samples (`n_random_samples`), which defaults to 10.

#### Auto Routing (`method="auto"`)
If you set `method="auto"`, `neojax` automatically routes to the most suitable strategy:
- **$k = 1$**: Uses `"exact"`.
- **$k \ge 2$**: Uses `"interpolate"` if all spatial grid dimensions of the input $x$ are $\le 10$, otherwise defaults to `"stochastic"` to prevent OOM and runtime issues.

::: neojax.losses.sobolev_losses.SobolevLoss

## Composing Losses

Real-world physics problems often require minimizing multiple objectives simultaneously (e.g., combining a data loss with a physics-informed regularization term). `ComposedLoss` allows you to pipe multiple loss functions together, computing the weighted sum of its constituent losses.

::: neojax.losses.composed_loss.ComposedLoss

### Learnable Loss Weights

When combining multiple losses, balancing their static weights can be notoriously difficult. `neojax` solves this by supporting dynamically learnable loss weights. 

If you enable learnable weights in a `ComposedLoss`, custom `BaseLoss`, or any other loss instance, you must ensure JAX differentiates with respect to them. `neojax` provides the `is_learnable_loss_weight` utility to create the correct filter specification for `equinox`.

```python
import equinox as eqx
from neojax.losses.utils import is_learnable_loss_weight

# 1. Create a filter spec that isolates the network parameters AND learnable loss weights
filter_spec = is_learnable_loss_weight(model_and_losses)

# 2. Pass the spec to filter_grad
grads = eqx.filter_grad(loss_fn, filter=filter_spec)(model_and_losses, data, targets)
```

::: neojax.losses.utils.is_learnable_loss_weight

## Custom Losses

When implementing custom losses, inherit from `BaseLoss`.

Each custom loss should implement the `__call__` method, taking the predicted array and the target array as inputs and returning a scalar loss.
