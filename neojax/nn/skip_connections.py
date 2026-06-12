"""Implementation of various skip connections."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class SoftGating(eqx.Module):
    """Applies soft-gating by weighting the channels of the given input.

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Args:
        ndim: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
            If provided, must match `in_channels`.
        use_bias: Whether to include a learnable bias.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **weight** (`Float[Array, ...]`): Learnable channel-wise weights.
        * **bias** (`Float[Array, ...] | None`): Optional learnable channel-wise bias.
    """

    weight: Float[Array, "c ..."]
    bias: Float[Array, "c ..."] | None = None

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int | None = None,
        use_bias: bool = False,
    ) -> None:
        if out_channels is not None and in_channels != out_channels:
            raise ValueError(
                "Mismatch of in_channels and out_channels. Both must match. "
                f"Got {in_channels} in_channels and {out_channels} out_channels."
            )
        self.weight = jnp.ones((in_channels,) + (1,) * ndim)
        if use_bias:
            self.bias = jnp.ones((in_channels,) + (1,) * ndim)

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Applies soft-gating to input activations.

        Args:
            x: Input activations.

        Returns:
            Weighted activations.
        """
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class Flattened1dConv(eqx.Module):
    """Applies 1d convolution to flattened dimensions.

    Skip layer that flattens all dimensions except for the
    leading channel dimension and applies 1d convolution over them,
    then un-flattens result.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        use_bias: Whether to add a learnable bias to the output.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **conv** (`eqx.nn.Conv1d`): The underlying 1D convolution layer.
        * **out_channels** (`int`): Number of output channels.
    """

    conv: eqx.nn.Conv1d
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_bias: bool = False,
    ) -> None:
        self.conv = eqx.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=use_bias,
            key=key,
        )
        self.out_channels = out_channels

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Applies 1d convolution to flattened dimensions.

        Args:
            x: Input activations.

        Returns:
            Output activations.
        """
        shape = x.shape
        x = x.reshape(shape[0], -1)
        x = self.conv(x)
        return x.reshape(self.out_channels, *shape[1:])
