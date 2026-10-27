"""Implementation of various CNO blocks."""

from typing import Literal

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.nn.filtered_lrelu import FilteredLReLU
from neojax.nn.kaiser_fir_filter import KaiserFIRFilter

# Convolution padding mode matching the `pad_mode` of the activation
# All caps to match equinox conv padding modes
_CONV_PADDING_MODE = {"wrap": "CIRCULAR", "reflect": "REFLECT", "constant": "ZEROS"}


class CNOConvBlock(eqx.nn.StatefulLayer):
    r"""Convolution - batch normalization - filtered activation block of the CNO.

    Implements the general n-dimensional convolution operator $\mathcal{K}_l$
    followed by the alias-free activation $\Sigma_{s, s'}$ of the `CNO`
    reference, i.e. the `CNO` I-, U- and D-blocks. The resampling
    is part of the activation, see `FilteredLReLU`.

    The filters of the activation are designed as in the reference
    from the resolution ratio `s' / s` of the block type: With the input
    sampling rate $s$ and the output sampling rate $s'$ the activation is
    evaluated at $s_{\text{act}} = 2 \max(s, s')$. The up-/downsampling filters
    have cutoffs $s / c$ and $s' / c$, with the cutoff
    denominator $c$ (`cutoff_den`), and half-widths $m s - s / c$ and
    $m s' - s' / c$, with the half-width multiplier $m$ (`half_width_mult`).
    This only depends on the ratio of $s$ and $s'$, so a block works for any
    input resolution.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        ndim: Number of spatial dimensions.
        block_type: The resampling of the block. Must be one of
            `"invariant"` (same resolution), `"upsample"` (twice the
            resolution) or `"downsample"` (half the resolution).
            Default is `"invariant"`.
        conv_kernel: Odd size of the convolution kernel along each dimension.
            Default is 3.
        batch_norm: Whether to apply batch normalization after the convolution,
            before the activation, as in the reference. If `False`, there is no
            normalization. Default is `True`. Batch normalization makes the
            block stateful, see below.
        batch_axis_name: Name of the `jax.vmap` axis the batch statistics are
            computed over if `batch_norm=True`. Default is `"batch"`.
        batch_norm_momentum: Weight of the running statistics (not of the batch
            statistics) in their update if `batch_norm=True`.
            Default is `0.9`, which corresponds to PyTorch's `momentum=0.1`.
        cutoff_den: Denominator of the cutoff frequency of the filters,
            in `(2, inf)`. Default is `2.0001`, i.e. slightly above critical sampling.
        half_width_mult: Multiplier of the half-width of the transition band
            of the filters. Default is `0.8`.
        filter_size: Number of filter taps per phase of the resampling,
            i.e. the filters have `filter_size * factor` taps. Default is 6.
        pad_mode: Padding mode of the convolution and the activation.
            Use `"wrap"` for a periodic domain, otherwise `"reflect"` or
            `"constant"` (zero padding, as in the reference).
            Default is `"constant"`.
        negative_slope: Negative slope of the leaky ReLU. Default is `0.2`.

    Raises:
        ValueError: If `conv_kernel` is not odd.

    !!! info "Stateful batch normalization"
        The block is an `eqx.nn.StatefulLayer`. It is only stateful if
        `batch_norm=True` (see `is_stateful`). In this case, it has to be called as
        `out, state = block(x, state)` inside of a `jax.vmap` with
        `axis_name=batch_axis_name`, with the `state` returned by
        `eqx.nn.make_with_state`, see
        [Equinox docs](https://docs.kidger.site/equinox/api/nn/stateful/).
        Otherwise, it is called as `out = block(x)`.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **convolution** (`eqx.nn.Conv`): The convolution layer.
        * **batch_norm** (`eqx.nn.BatchNorm | None`): The batch normalization layer
            or `None`.
        * **activation** (`FilteredLReLU`): The filtered activation with resampling.

    Examples:
        ```python
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import CNOConvBlock

        key = jr.key(0)
        block = CNOConvBlock(
            key=key,
            in_channels=3,
            out_channels=8,
            ndim=2,
            block_type="downsample",
            batch_norm=False,
        )
        x = jnp.ones((3, 32, 32))
        # out.shape = (8, 16, 16)
        out = block(x)

        # Batch normalization: stateful block
        stateful_block, state = eqx.nn.make_with_state(CNOConvBlock)(
            key=key, in_channels=3, out_channels=8, ndim=2, batch_norm=True
        )
        xs = jnp.ones((4, 3, 32, 32))
        out, state = jax.vmap(
            stateful_block, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )(xs, state)
        ```
    """

    convolution: eqx.nn.Conv
    batch_norm: eqx.nn.BatchNorm | None
    activation: FilteredLReLU

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        ndim: int,
        block_type: Literal["invariant", "upsample", "downsample"] = "invariant",
        conv_kernel: int = 3,
        batch_norm: bool = True,
        batch_axis_name: str = "batch",
        batch_norm_momentum: float = 0.9,
        cutoff_den: float = 2.0001,
        half_width_mult: float = 0.8,
        filter_size: int = 6,
        pad_mode: Literal["wrap", "reflect", "constant"] = "constant",
        negative_slope: float = 0.2,
    ) -> None:
        if conv_kernel % 2 != 1:
            raise ValueError(f"`conv_kernel` must be odd, got {conv_kernel}.")
        if pad_mode not in _CONV_PADDING_MODE:
            raise ValueError(
                "Invalid pad_mode. Must be one of ['wrap', 'reflect', 'constant']."
            )
        self.convolution = eqx.nn.Conv(
            num_spatial_dims=ndim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2,
            padding_mode=_CONV_PADDING_MODE[pad_mode],
            key=key,
        )
        if batch_norm:
            self.batch_norm = eqx.nn.BatchNorm(
                input_size=out_channels,
                axis_name=batch_axis_name,
                momentum=batch_norm_momentum,
                mode="batch",
            )
        else:
            self.batch_norm = None

        # Filter design on a unit input grid, only the ratio of in and out matters
        rate_in = 1.0
        rate_out = {"invariant": 1.0, "upsample": 2.0, "downsample": 0.5}[block_type]
        rate_act = 2 * max(rate_in, rate_out)
        up = round(rate_act / rate_in)
        down = round(rate_act / rate_out)
        h_up = KaiserFIRFilter(
            cutoff_freq=rate_in / cutoff_den,
            num_taps=filter_size,
            half_band_width=half_width_mult * rate_in - rate_in / cutoff_den,
            sampling_fac=up,
            sampling_freq=rate_in,
        )
        h_down = KaiserFIRFilter(
            cutoff_freq=rate_out / cutoff_den,
            num_taps=filter_size,
            half_band_width=half_width_mult * rate_out - rate_out / cutoff_den,
            sampling_fac=down,
            sampling_freq=rate_out,
        )
        self.activation = FilteredLReLU(
            ndim=ndim,
            channels=out_channels,
            h_up=h_up,
            h_down=h_down,
            block_type=block_type,
            pad_mode=pad_mode,
            negative_slope=negative_slope,
        )

    def is_stateful(self) -> bool:
        """Whether the block is stateful, i.e. uses batch normalization.

        Returns:
            `True` if the block has to be called as `block(x, state)`
            and returns `(out, state)`, `False` if it is called as `block(x)`.
        """
        return self.batch_norm is not None

    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        state: eqx.nn.State | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Inexact[Array, "out_c ..."] | tuple[Inexact[Array, "out_c ..."], eqx.nn.State]:
        """Forward pass of the CNO convolution block.

        Args:
            x: Input array shaped (in_c, d1, ..., dN).
            state: The state holding the running statistics of the batch
                normalization. Required if (and only if) `is_stateful()`.
            key: Unused. Accepted for compatibility with `eqx.nn.Sequential`.
            inference: Whether to use the running statistics of the batch
                normalization instead of the batch statistics. If `None`, the
                value of the `BatchNorm` is used (see `eqx.nn.inference_mode`).
                Ignored if the block is not stateful.

        Returns:
            Output array shaped (out_c, d1', ..., dN'), with the resolution
            scaled according to the `block_type`. If `is_stateful()`, the output
            and the updated state.

        Raises:
            ValueError: If a state is required, but not given or vice versa.
        """
        if (state is not None) != self.is_stateful():
            raise ValueError(
                "A state has to be passed if (and only if) the block is stateful, "
                "i.e. `batch_norm=True`. Got "
                f"is_stateful()={self.is_stateful()} and state is None={state is None}."
            )
        x = self.convolution(x)
        if self.batch_norm is not None:
            x, state = self.batch_norm(x, state, inference=inference)
            return self.activation(x), state
        return self.activation(x)


class CNOResidualBlock(eqx.nn.StatefulLayer):
    r"""Residual block of the CNO.

    Computes `x + bn(conv(act(bn(conv(x)))))` with resolution preserving
    convolutions, optional batch normalizations `bn` and the
    filtered activation (an I-block, see `CNOConvBlock`).

    Args:
        key: PRNG key for parameter initialization.
        channels: Number of input and output channels.
        ndim: Number of spatial dimensions.
        conv_kernel: Odd size of the convolution kernel along each dimension.
            Default is 3.
        batch_norm: Whether to apply batch normalization after each
            convolution, as in the reference. If `False`, there is no
            normalization. Default is `True`. Batch normalization makes the
            block stateful, see below.
        batch_axis_name: Name of the `jax.vmap` axis the batch statistics are
            computed over if `batch_norm=True`. Default is `"batch"`.
        batch_norm_momentum: Weight of the running statistics (not of the batch
            statistics) in their update if `batch_norm=True`.
            Default is `0.9`, which corresponds to PyTorch's `momentum=0.1`.
        cutoff_den: Denominator of the cutoff frequency of the filters.
            See `CNOConvBlock`. Default is `2.0001`.
        half_width_mult: Multiplier of the half-width of the transition band
            of the filters. See `CNOConvBlock`. Default is `0.8`.
        filter_size: Number of filter taps per phase of the resampling.
            See `CNOConvBlock`. Default is 6.
        pad_mode: Padding mode of the convolutions and the activation.
            Use `"wrap"` for a periodic domain, otherwise `"reflect"` or
            `"constant"` (zero padding, as in the reference).
            Default is `"constant"`.
        negative_slope: Negative slope of the leaky ReLU. Default is `0.2`.

    Raises:
        ValueError: If `conv_kernel` is not odd.

    !!! info "Stateful batch normalization"
        The block is an `eqx.nn.StatefulLayer`. It is only stateful if
        `batch_norm=True` (see `is_stateful`). See `CNOConvBlock` for how to
        call stateful blocks.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **block** (`CNOConvBlock`): First convolution, batch normalization and activation.
        * **convolution** (`eqx.nn.Conv`): The second convolution layer.
        * **batch_norm** (`eqx.nn.BatchNorm | None`): The second batch normalization
            layer or `None`.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import CNOResidualBlock

        key = jr.key(0)
        block = CNOResidualBlock(key=key, channels=8, ndim=2, batch_norm=False)
        x = jnp.ones((8, 32, 32))
        out = block(x)
        ```
    """

    block: CNOConvBlock
    convolution: eqx.nn.Conv
    batch_norm: eqx.nn.BatchNorm | None

    def __init__(
        self,
        key: PRNGKeyArray,
        channels: int,
        ndim: int,
        conv_kernel: int = 3,
        batch_norm: bool = True,
        batch_axis_name: str = "batch",
        batch_norm_momentum: float = 0.9,
        cutoff_den: float = 2.0001,
        half_width_mult: float = 0.8,
        filter_size: int = 6,
        pad_mode: Literal["wrap", "reflect", "constant"] = "constant",
        negative_slope: float = 0.2,
    ) -> None:
        key1, key2 = jr.split(key)
        self.block = CNOConvBlock(
            key=key1,
            in_channels=channels,
            out_channels=channels,
            ndim=ndim,
            block_type="invariant",
            conv_kernel=conv_kernel,
            batch_norm=batch_norm,
            batch_axis_name=batch_axis_name,
            batch_norm_momentum=batch_norm_momentum,
            cutoff_den=cutoff_den,
            half_width_mult=half_width_mult,
            filter_size=filter_size,
            pad_mode=pad_mode,
            negative_slope=negative_slope,
        )
        self.convolution = eqx.nn.Conv(
            num_spatial_dims=ndim,
            in_channels=channels,
            out_channels=channels,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2,
            padding_mode=_CONV_PADDING_MODE[pad_mode],
            key=key2,
        )
        if batch_norm:
            self.batch_norm = eqx.nn.BatchNorm(
                input_size=channels,
                axis_name=batch_axis_name,
                momentum=batch_norm_momentum,
                mode="batch",
            )
        else:
            self.batch_norm = None

    def is_stateful(self) -> bool:
        """Whether the block is stateful, i.e. uses batch normalization.

        Returns:
            `True` if the block has to be called as `block(x, state)`
            and returns `(out, state)`, `False` if it is called as `block(x)`.
        """
        return self.block.is_stateful()

    def __call__(
        self,
        x: Inexact[Array, "c ..."],
        state: eqx.nn.State | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Inexact[Array, "c ..."] | tuple[Inexact[Array, "c ..."], eqx.nn.State]:
        """Forward pass of the residual block.

        Args:
            x: Input array shaped (c, d1, ..., dN).
            state: The state holding the running statistics of the batch
                normalizations. Required if (and only if) `is_stateful()`.
            key: Unused. Accepted for compatibility with `eqx.nn.Sequential`.
            inference: Whether to use the running statistics of the batch
                normalizations instead of the batch statistics. If `None`, the
                value of the `BatchNorm` is used (see `eqx.nn.inference_mode`).
                Ignored if the block is not stateful.

        Returns:
            Output array with the same shape as the input. If `is_stateful()`,
            the output and the updated state.

        Raises:
            ValueError: If a state is required, but not given or vice versa.
        """
        if self.batch_norm is not None:
            out, state = self.block(x, state, inference=inference)
            out = self.convolution(out)
            out, state = self.batch_norm(out, state, inference=inference)
            return x + out, state
        out = self.convolution(self.block(x, state))
        return x + out


class CNOLiftProjectBlock(eqx.Module):
    """Lifting or projection block of the CNO.

    Applies an un-normalized, `CNOConvBlock` that maps
    the input to `latent_channels` followed by a final convolution to
    `out_channels`. Used both to lift the inputs and to project the latent
    features to the outputs. As in the reference, there is no batch
    normalization (stateless).

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        ndim: Number of spatial dimensions.
        latent_channels: Number of channels between the two layers.
            Defaults to 64.
        conv_kernel: Odd size of the convolution kernel along each dimension.
            Default is 3.
        cutoff_den: Denominator of the cutoff frequency of the filters.
            See `CNOConvBlock`. Default is `2.0001`.
        half_width_mult: Multiplier of the half-width of the transition band
            of the filters. See `CNOConvBlock`. Default is `0.8`.
        filter_size: Number of filter taps per phase of the resampling.
            See `CNOConvBlock`. Default is 6.
        pad_mode: Padding mode of the convolutions and the activation.
            Use `"wrap"` for a periodic domain, otherwise `"reflect"` or
            `"constant"` (zero padding, as in the reference).
            Default is `"constant"`.
        negative_slope: Negative slope of the leaky ReLU. Default is `0.2`.

    Raises:
        ValueError: If `conv_kernel` is not odd.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **block** (`CNOConvBlock`): Convolution and filtered activation,
            without batch normalization.
        * **convolution** (`eqx.nn.Conv`): The final convolution layer.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import CNOLiftProjectBlock

        key = jr.key(0)
        lift = CNOLiftProjectBlock(key=key, in_channels=3, out_channels=8, ndim=2)
        x = jnp.ones((3, 32, 32))
        # out.shape = (8, 32, 32)
        out = lift(x)
        ```
    """

    block: CNOConvBlock
    convolution: eqx.nn.Conv

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        ndim: int,
        latent_channels: int = 64,
        conv_kernel: int = 3,
        cutoff_den: float = 2.0001,
        half_width_mult: float = 0.8,
        filter_size: int = 6,
        pad_mode: Literal["wrap", "reflect", "constant"] = "constant",
        negative_slope: float = 0.2,
    ) -> None:
        key1, key2 = jr.split(key)
        self.block = CNOConvBlock(
            key=key1,
            in_channels=in_channels,
            out_channels=latent_channels,
            ndim=ndim,
            block_type="invariant",
            conv_kernel=conv_kernel,
            batch_norm=False,
            cutoff_den=cutoff_den,
            half_width_mult=half_width_mult,
            filter_size=filter_size,
            pad_mode=pad_mode,
            negative_slope=negative_slope,
        )
        self.convolution = eqx.nn.Conv(
            num_spatial_dims=ndim,
            in_channels=latent_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2,
            padding_mode=_CONV_PADDING_MODE[pad_mode],
            key=key2,
        )

    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the lift/project block.

        Args:
            x: Input array shaped (in_c, d1, ..., dN).

        Returns:
            Output array shaped (out_c, d1, ..., dN).
        """
        return self.convolution(self.block(x))
