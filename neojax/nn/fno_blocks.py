"""Implementation of general Fourier Neural Operator (FNO) block(s)."""

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.skip_connections import Flattened1dConv, SoftGating
from neojax.nn.spectral_conv import SpectralConvNd


class FNOBlock(eqx.Module):
    """General FNO Block.

    FNO block composed of the global non-linear integral
    kernel, i.e., kappa = iFFT(R @ FFT(v)), and an optional linear
    transformation $W @ v$.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to retain
            across each spatial dimension.
            Must be a sequence of integers (e.g., `(16, 16)` for 2D).
        activation: Activation function to use.
            Defaults to `jax.nn.gelu`.
        fno_skip: Type of skip connection. Can be `"linear"`,
            `"soft-gating"`, `"identity"`, or `None`.
            Defaults to `"linear"`.
        use_skip_bias: Whether to use a bias term
            in the skip connection. Defaults to `False`.
        preactivation: Whether to apply the activation
            before the spectral convolution and skip connection.
            Defaults to `False`.

    Attributes:
        spectral_conv: The `SpectralConvNd` layer
            performing the operator integral.
        skip: The initialized skip connection layer or `None`.
        activation: The activation function.
        preactivation: Boolean flag indicating if preactivation is used.

    Examples:
        ```python
        import jax.random as jr
        from neojax.nn.fno_blocks import FNOBlock

        key = jr.PRNGKey(0)
        # Initialize a 2D FNO Block
        fno_block = FNOBlock(
            key=key,
            in_channels=3,
            out_channels=8,
            modes=(4, 4),
            fno_skip="linear"
        )
        # Input shape: (channels, height, width)
        x = jnp.ones((3, 32, 32))
        out = fno_block(x)
        ```
    """

    spectral_conv: SpectralConvNd
    skip: Flattened1dConv | SoftGating | eqx.nn.Identity | None = Flattened1dConv
    activation: Callable
    preactivation: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        fno_skip: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_skip_bias: bool = False,
        preactivation: bool = False,
    ) -> None:
        ndim = len(modes)
        fno_key, skip_key = jr.split(key, 2)
        if fno_skip == "linear":
            self.skip = Flattened1dConv(
                key=skip_key,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                use_bias=use_skip_bias,
            )
        elif fno_skip == "soft-gating":
            self.skip = SoftGating(
                ndim=ndim,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_skip_bias,
            )
        elif fno_skip == "identity":
            if in_channels != out_channels:
                raise ValueError(
                    "Identity skip connection requires in_channels == out_channels. "
                    f"Got {in_channels} and {out_channels}."
                )
            self.skip = eqx.nn.Identity()
        elif fno_skip is None:
            self.skip = None
        else:
            raise ValueError(f"'{fno_skip}' is not a valid skip connection.")
        self.spectral_conv = SpectralConvNd(
            key=fno_key, in_channels=in_channels, out_channels=out_channels, modes=modes
        )
        self.activation = activation
        self.preactivation = preactivation

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Forward pass of FNO block.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        if self.preactivation:
            x = self.activation(x)
        x_fft = self.spectral_conv(x)
        if self.skip is not None:
            x_skip = self.skip(x)
            if self.preactivation:
                return x_fft + x_skip
            return self.activation(x_fft + x_skip)
        else:
            if self.preactivation:
                return x_fft
            return self.activation(x_fft)


class FNOBlocks(eqx.Module):
    """General FNO Blocks with variable layer number.

    Implemented as in [[1]](#ref1) and [[2]](#ref2).
    Each block is an instance of FNOBlock with an
    optional channelwise MLP.

    Args:
        key: PRNG key for parameter initialization.
        n_layers: Number of consecutive FNO blocks.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dimension,
            e.g. (16, 16) for 2D inputs.
        activation: Activation function or sequence of
            activation functions used inside the FNO blocks.
            Default is `jax.nn.gelu`.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            wiwth 2 layers after each FNO block. Default is True.
        preactivation: Whether to apply the activation function before
            the spectral convolution and skip connections.
            Default is False. This is an exclusive flag.
            If activation is applied before, it isn't applied after.
        fno_skip: Type of skip connection used inside the FNO blocks.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"linear"`.
        use_fno_skip_bias: Whether to use a bias term
            in the FNO block's skip connection. Default is False.
        channel_mlp_skip: Type of skip connection used
            around the channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"soft-gating"`.
        channel_mlp_expansion: Expansion factor for computing the hidden
            channel dimension of the MLPs. Default is `0.5`.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is `jax.nn.gelu`.

    Attributes:
        fno_layers: The initialized `FNOBlock` layers.
        channel_mlps: The initialized `PointwiseMLP` layers,
            or None if `use_channel_mlp` is `False`.
        channel_mlp_skip: The skip connection instance or class
            applied alongside the MLPs or None.

    Examples:
        ```python
        import jax.random as jr
        from neojax.nn.fno_blocks import FNOBlocks

        key = jr.PRNGKey(0)

        # Initialize a sequence of 4 FNO Blocks
        fno_blocks = FNOBlocks(
            key=key,
            n_layers=4,
            in_channels=3,
            out_channels=16,
            modes=(8, 8),
            use_channel_mlp=True
        )
        # Input shape: (channels, height, width)
        x = jnp.ones((3, 64, 64))
        out = fno_blocks(x)
        ```

    References:
        1. <a name="ref1"></a> Li, Z. et al. "Fourier Neural Operator for Parametric
            Partial Differential Equations" (2021).
            ICLR 2021, https://arxiv.org/pdf/2010.08895.

        2. <a name="ref2"></a> Kovachki, N. et al. "Neural Operator: Learning Maps
            Between Function Spaces With Applications to PDEs"
            JMLR 2023, https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf.

    Notes:
        The current implementation doesn't support dropout
        for the channel-wise MLP or normalization layers.
        Both will be added in future releases.
    """

    fno_layers: tuple[FNOBlock, ...]
    channel_mlps: tuple[PointwiseMLP, ...] | None
    channel_mlp_skip: tuple | None = None

    def __init__(
        self,
        key: PRNGKeyArray,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        activation: Callable | Sequence[Callable] = jax.nn.gelu,
        use_channel_mlp: bool = True,
        preactivation: bool = False,
        fno_skip: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_fno_skip_bias: bool = False,
        channel_mlp_skip: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
    ) -> None:
        fno_layers = []
        if isinstance(activation, Callable):
            activations = [activation] * n_layers
        else:
            if len(activations) != n_layers:
                raise ValueError(
                    "Mismatch in the number of activations and layers: "
                    "Can only have the same num of activations and n_layers,"
                    f" but got {len(activations)} activations and "
                    f"{n_layers} layers!"
                )
        for i in range(n_layers):
            fno_key, key = jr.split(key, 2)
            if i > 0:
                in_channels = out_channels
            fno_layers.append(
                FNOBlock(
                    key=fno_key,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=modes,
                    activation=activations[i],
                    fno_skip=fno_skip,
                    use_skip_bias=use_fno_skip_bias,
                    preactivation=preactivation,
                )
            )
        self.fno_layers = tuple(fno_layers)

        if use_channel_mlp:
            mlp_keys = jr.split(key, n_layers)
            if channel_mlp_expansion is not None:
                hidden_channel = round(channel_mlp_expansion * out_channels)
            else:
                hidden_channel = out_channels
            channel_mlp_skips = []
            for _ in range(n_layers):
                skip_key, key = jr.split(key, 2)
                if channel_mlp_skip == "linear":
                    channel_mlp_skips.append(
                        Flattened1dConv(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            key=skip_key,
                        )
                    )
                elif channel_mlp_skip == "soft-gating":
                    channel_mlp_skips.append(
                        SoftGating(
                            ndim=len((modes,) if isinstance(modes, int) else modes),
                            in_channels=out_channels,
                        )
                    )
                elif channel_mlp_skip == "identity":
                    channel_mlp_skips.append(eqx.nn.Identity())
                elif channel_mlp_skip is None:
                    channel_mlp_skips.append(None)
                else:
                    raise ValueError(
                        f"'{channel_mlp_skip}' is not a valid skip connection."
                    )
            self.channel_mlp_skip = (
                tuple(channel_mlp_skips) if channel_mlp_skip is not None else None
            )
            channel_mlps = [
                PointwiseMLP(
                    key=mlp_keys[i],
                    layers=(out_channels, hidden_channel, out_channels),
                    activations=channel_mlp_activations,
                )
                for i in range(n_layers)
            ]
            self.channel_mlps = tuple(channel_mlps)
        else:
            self.channel_mlps = None

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Forward pass through n_layers of FNO Blocks.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        if self.channel_mlps is not None:
            skips = (
                self.channel_mlp_skip
                if self.channel_mlp_skip is not None
                else [None] * len(self.channel_mlps)
            )
            for fno_layer, mlp_layer, skip_layer in zip(
                self.fno_layers, self.channel_mlps, skips, strict=True
            ):
                x = fno_layer(x)
                if skip_layer is not None:
                    x = mlp_layer(x) + skip_layer(x)
                else:
                    x = mlp_layer(x)
            return x
        else:
            for layer in self.fno_layers:
                x = layer(x)
            return x
