"""Implementation of the standard dense spectral tensor."""

from collections.abc import Sequence
from typing import Literal, final

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, PRNGKeyArray

from neojax.tensor.base_tensor import BaseTensor


@final
class DenseTensor(BaseTensor):
    """Standard dense spectral tensor.

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels + out_channels)) ** 0.5.
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        * **weights** (`tuple[Complex[Array, ...], ...]`): Learnable complex weights for the retained Fourier modes.
        * **separable** (bool): Whether to use separable implementation of contraction.
    """

    weights: tuple[Complex[Array, "..."], ...]
    separable: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        init_std: float | Literal["auto"] = "auto",
        separable: bool = False,
    ) -> None:
        self.separable = separable
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        num_corners = 2 ** (len(modes) - 1)
        if isinstance(init_std, str) and init_std == "auto":
            if separable:
                init_std = (2.0 / in_channels) ** 0.5
            else:
                init_std = (2.0 / (in_channels * out_channels)) ** 0.5
        if not isinstance(init_std, float):
            raise ValueError("'init_std' must be float or 'auto'.")
        keys = jr.split(key, num_corners)

        weights_list = []
        if separable:
            weight_shape = (in_channels,) + tuple(modes)
        else:
            weight_shape = (out_channels, in_channels) + tuple(modes)

        for k in keys:
            w_real = jr.normal(k, weight_shape)
            w_imag = jr.normal(jr.split(k)[0], weight_shape)
            weights_list.append(init_std * (w_real + 1j * w_imag))

        self.weights = tuple(weights_list)

    def to_dense(self) -> Complex[Array, "n_corners ..."]:
        """Reconstruct dense tensors into single dense tensor.

        Returns:
            Dense tensors stacked along first axis.
        """
        return jnp.stack(self.weights, axis=0)

    @classmethod
    def from_weights(
        cls, weights: Complex[Array, "n_corners ..."], separable: bool = False
    ) -> "DenseTensor":
        """Alternative constructor to wrap constructed dense tensors.

        Commonly used to construct a `DenseTensor` from factorized tensors
        after calling `to_dense()`, e.g. if `reconstructed` flag is `True`.

        Args:
            weights: Constructed dense tensors.
            separable: Whether the wrapped weights are separable.

        Returns:
            A new instance of `DenseTensor`.
        """
        # unstack first dim
        weights = tuple(weights)
        self = object.__new__(cls)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "separable", separable)
        return self

    def __call__(
        self, corner_idx: int, x_ft_slice: Complex[Array, "in_c ..."]
    ) -> Complex[Array, "..."]:
        """Performs dense tensor contraction.

        Args:
            corner_idx: Corner index of the n-dim FFT hypercube.
            x_ft_slice: Mode slice of fourier transformed input.

        Returns:
            Tensor contraction of respective weight and input.
        """
        if self.separable:
            return self.weights[corner_idx] * x_ft_slice
        else:
            return jnp.einsum("oi...,i...->o...", self.weights[corner_idx], x_ft_slice)
