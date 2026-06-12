"""Implementation of normalization layers."""

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class InstanceNorm(eqx.Module):
    r"""Computes per instance normalization.

    Computes
    $$
    \hat{x_{b,c,\mathbf{d}}} = \frac{x_{b,c,\mathbf{d}} - \mu_{b,c}}{\sqrt{\sigma_{b,c}^2 + \eps}}
    $$
    where the mean and variance are over the spatial dimensions only.
    Optionally applies the transform $\gamma_c \hat{x}_{b,c,\mathbf{d}} + \beta_c$
    with learnable parameters $\gamma_c$ and $\beta_c$.

    Args:
        shape: Overall shape of the input signal, i.e., `(channels, d1, ..., dN)`.
        eps: A small value added to the variance for numerical stability.
            Defaults to `1e-7`.
        use_weight: Whether to include a learnable affine weight.
            Defaults to `True`.
        use_bias: Whether to include a learnable affine bias.
            Defaults to `True`.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **shape** (`tuple[int, ...]`): The input shape.
        * **eps** (`float`): The numerical stability value.
        * **weight** (`Float[Array, ...] | None`): The learnable weights or `None`.
        * **bias** (`Float[Array, ...] | None`): The learnable bias or `None`.
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_weight: bool = eqx.field(default=True, static=True)
    use_bias: bool = eqx.field(default=True, static=True)
    weight: Float[Array, "c ..."] | None
    bias: Float[Array, "c ..."] | None

    def __init__(
        self,
        shape: Sequence[int],
        eps: float = 1e-7,
        use_weight: bool = True,
        use_bias: bool = True,
    ) -> None:
        self.shape = tuple(shape)
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias

        c = self.shape[0]
        spatial_dims = len(self.shape[1:])
        self.weight = jnp.ones((c, *([1] * spatial_dims))) if use_weight else None
        self.bias = jnp.zeros((c, *([1] * spatial_dims))) if use_bias else None

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Compute Instance Norm.

        Args:
            x: Input array with the same shape as `shape` in `__init__`
                (channel, d1, ..., dN).

        Returns:
            Normalized array.
        """
        spatial_axes = tuple(range(1, x.ndim))
        mean = jnp.mean(x, axis=spatial_axes, keepdims=True)
        var = jnp.var(x, axis=spatial_axes, keepdims=True)
        rec = jax.lax.rsqrt(var + self.eps)
        out = (x - mean) * rec

        if self.use_weight:
            out = self.weight * out
        if self.use_bias:
            out = out + self.bias
        return out
