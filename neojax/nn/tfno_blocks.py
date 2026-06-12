"""Implementation of general Tucker factorized FNO block(s)."""

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.normalization import InstanceNorm
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.skip_connections import Flattened1dConv, SoftGating
from neojax.nn.tucker_spectral_conv import TuckerSpectralConvNd


class TFNOBlock(eqx.Module):
    r"""General Tucker factorized FNO Block.

    TFNO block composed of the global non-linear integral kernel, i.e.,
    $\mathcal{K}(\mathbf{v}) = \mathcal{F}^{-1}(R \cdot \mathcal{F}(\mathbf{v}))$,
    and an optional local operator $W \mathbf{v}$.
    Here $R$ is decomposed into core tensor and factor matrices.
    A block can be further customized with Resnet-style residual
    connections around the TFNO and normalization
    before the non-linear activation.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to retain
            across each spatial dimension.
            Must be a sequence of integers (e.g., `(16, 16)` for 2D)
            or single integer for 1D inputs.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not,
            should be num_spatial_dims + num_channel_dims ranks,
            e.g. in 2D 4 ranks (out_channel, in_channel, 2 spatial ranks).
        share_factor_matrices: Whether to share the factor matrices
            across the weight tensors. This further decreases the number
            of weight tensors. Only the core tensor is not shared then.
            Default is True.
        activation: Activation function to use.
            Defaults to `jax.nn.gelu`.
        local_operator: Type of local operator to use.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or `None`.
            Defaults to `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the TFNO blocks local operator. Defaults to `False`.
        normalization: Type of normalization to use. Applied after
            the spectral_op + local_op sum, before activation.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is `"layer"`.
        use_fno_residual: Whether to use a Resnet-style residual
            connection around each TFNO block. Improves stability.
            Default True.
        preactivation: Whether to apply the activation
            before the spectral convolution and skip connection.
            Defaults to `False`. This is an exclusive flag.
            If activation is applied before, it isn't applied after.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **tucker_spectral_conv** (`TuckerSpectralConvNd`): The `TuckerSpectralConvNd` layer performing the operator integral.
        * **local_operator** (`Flattened1dConv | SoftGating | eqx.nn.Identity | None`): The initialized local operator layer or `None`.
        * **normalization** (`InstanceNorm | eqx.nn.GroupNorm | None`): Type of normalization to use. Applied after the spectral_op + local_op sum, before activation.
        * **activation** (`Callable`): The activation function.
        * **preactivation** (`bool`): Boolean flag indicating if preactivation is used.
        * **use_fno_residual** (`bool`): Whether to use a Resnet-style residual connection around each TFNO block. Improves stability.

    Examples:
        ```python
        import jax.random as jr
        from neojax.nn.tfno_blocks import TFNOBlock
        import jax.numpy as jnp

        key = jr.key(0)
        # Initialize a 2D TFNO Block
        tfno_block = TFNOBlock(
            key=key,
            in_channels=3,
            out_channels=8,
            modes=(4, 4),
            ranks=2,
            local_operator="linear"
        )
        # Input shape: (channels, height, width)
        x = jnp.ones((3, 32, 32))
        out = tfno_block(x)
        ```
    """

    tucker_spectral_conv: TuckerSpectralConvNd
    local_operator: Flattened1dConv | SoftGating | eqx.nn.Identity | None
    normalization: InstanceNorm | eqx.nn.GroupNorm | None
    activation: Callable
    preactivation: bool = eqx.field(static=True)
    use_fno_residual: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        ranks: int | Sequence[int],
        share_factor_matrices: bool = True,
        activation: Callable = jax.nn.gelu,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        use_fno_residual: bool = True,
        preactivation: bool = False,
    ) -> None:
        ndim = len(modes) if isinstance(modes, Sequence) else 1
        tfno_key, skip_key = jr.split(key, 2)
        if local_operator == "linear":
            self.local_operator = Flattened1dConv(
                key=skip_key,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                use_bias=use_local_operator_bias,
            )
        elif local_operator == "soft-gating":
            self.local_operator = SoftGating(
                ndim=ndim,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_local_operator_bias,
            )
        elif local_operator == "identity":
            if in_channels != out_channels:
                raise ValueError(
                    "Identity skip connection requires in_channels == out_channels. "
                    f"Got {in_channels} and {out_channels}."
                )
            self.local_operator = eqx.nn.Identity()
        elif local_operator is None:
            self.local_operator = None
        else:
            raise ValueError(f"'{local_operator}' is not a valid local operator.")
        self.tucker_spectral_conv = TuckerSpectralConvNd(
            key=tfno_key,
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            ranks=ranks,
            share_factor_matrices=share_factor_matrices,
        )
        # We use GroupNorm instead of eqx.nn.LayerNorm
        # because LayerNorm is very strict since equinox (v0.11+)
        if normalization == "layer":
            self.normalization = eqx.nn.GroupNorm(groups=1, channels=out_channels)
        elif normalization == "instance":
            self.normalization = InstanceNorm(shape=(out_channels, *([1] * ndim)))
        elif normalization == "group":
            self.normalization = eqx.nn.GroupNorm(
                groups=norm_groups, channels=out_channels
            )
        elif normalization is None:
            self.normalization = None
        else:
            raise ValueError(f"'{normalization}' is not a valid normalization.")
        self.activation = activation
        self.preactivation = preactivation
        self.use_fno_residual = use_fno_residual

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Forward pass of TFNO block.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        res = x
        if self.preactivation:
            x = self.activation(x)

        x_fft = self.tucker_spectral_conv(x)
        x_skip = self.local_operator(x) if self.local_operator is not None else 0

        x = x_fft + x_skip

        if self.normalization is not None:
            x = self.normalization(x)

        if not self.preactivation:
            x = self.activation(x)

        if self.use_fno_residual:
            if res.shape == x.shape:
                x = x + res

        return x


class TFNOBlocks(eqx.Module):
    """General TFNO Blocks with variable layer number.

    Each block is an instance of TFNOBlock with an
    optional channelwise MLP.

    Args:
        key: PRNG key for parameter initialization.
        n_layers: Number of consecutive TFNO blocks.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dimension,
            e.g. (16, 16) for 2D inputs.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not,
            should be num_spatial_dims + num_channel_dims ranks,
            e.g. in 2D 4 ranks (out_channel, in_channel, 2 spatial ranks).
        share_factor_matrices: Whether to share the factor matrices
            across the weight tensors. This further decreases the number
            of weight tensors. Only the core tensor is not shared then.
            Default is True.
        activation: Activation function or sequence of
            activation functions used inside the TFNO blocks.
            Default is `jax.nn.gelu`.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            with 2 layers after each TFNO block. Default is True.
        preactivation: Whether to apply the activation function before
            the spectral convolution and skip connections.
            Default is False. This is an exclusive flag.
            If activation is applied before, it isn't applied after.
        normalization: Type of normalization to use. Applied after
            the spectral_op + local_op sum, before activation.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is `"layer"`.
        norm_groups: Number of groups to use if `normalization="group"`.
            Defaults to 1.
        use_fno_residual: Whether to use a Resnet-style residual
            connection around each TFNO block. Improves stability.
            Default True.
        local_operator: Type of local operator to use in the TFNO block.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the TFNO blocks local operator. Default is `False`.
        channel_mlp_residual: Type of residual connection used
            around the channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"identity"`, which corresponds to Resnet-style
            residual connection.
        channel_mlp_expansion: Expansion factor for computing the hidden
            channel dimension of the MLPs. Default is `0.5`.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is `jax.nn.gelu`.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **tfno_layers** (`tuple[TFNOBlock, ...]`): The initialized `TFNOBlock` layers.
        * **channel_mlps** (`tuple[PointwiseMLP, ...] | None`): The initialized `PointwiseMLP` layers, or None if `use_channel_mlp` is `False`.
        * **channel_mlp_residuals** (`tuple | None`): The residual connection instances or None.

    Examples:
        ```python
        import jax.random as jr
        import jax.numpy as jnp
        from neojax.nn.tfno_blocks import TFNOBlocks

        key = jr.key(0)

        # Initialize a sequence of 4 TFNO Blocks
        tfno_blocks = TFNOBlocks(
            key=key,
            n_layers=4,
            in_channels=3,
            out_channels=16,
            modes=(8, 8),
            use_channel_mlp=True
        )
        # Input shape: (channels, height, width)
        x = jnp.ones((3, 64, 64))
        out = tfno_blocks(x)
        ```

    !!! info "Upcoming Features"
        The current implementation doesn't support dropout
        for the channel-wise MLP or normalization layers.
        Both will be added in future releases.
    """

    tfno_layers: tuple[TFNOBlock, ...]
    channel_mlps: tuple[PointwiseMLP, ...] | None
    channel_mlp_residuals: tuple | None = None

    def __init__(
        self,
        key: PRNGKeyArray,
        n_layers: int,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        ranks: int | Sequence[int],
        share_factor_matrices: bool = True,
        activation: Callable | Sequence[Callable] = jax.nn.gelu,
        use_channel_mlp: bool = True,
        preactivation: bool = False,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        use_fno_residual: bool = True,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "identity",
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
    ) -> None:
        tfno_layers = []
        if isinstance(activation, Callable):
            activations = [activation] * n_layers
        else:
            if len(activation) != n_layers:
                raise ValueError(
                    "Mismatch in the number of activations and layers: "
                    "Can only have the same num of activations and n_layers,"
                    f" but got {len(activation)} activations and "
                    f"{n_layers} layers!"
                )
            activations = activation

        for i in range(n_layers):
            tfno_key, key = jr.split(key, 2)
            if i > 0:
                in_channels = out_channels
            tfno_layers.append(
                TFNOBlock(
                    key=tfno_key,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=modes,
                    ranks=ranks,
                    share_factor_matrices=share_factor_matrices,
                    activation=activations[i],
                    local_operator=local_operator,
                    use_local_operator_bias=use_local_operator_bias,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    use_fno_residual=use_fno_residual,
                    preactivation=preactivation,
                )
            )
        self.tfno_layers = tuple(tfno_layers)

        if use_channel_mlp:
            mlp_keys = jr.split(key, n_layers)
            if channel_mlp_expansion is not None:
                hidden_channel = round(channel_mlp_expansion * out_channels)
            else:
                hidden_channel = out_channels
            channel_mlp_skips = []
            ndim = len(modes) if isinstance(modes, Sequence) else 1
            for _ in range(n_layers):
                skip_key, key = jr.split(key, 2)
                if channel_mlp_residual == "linear":
                    channel_mlp_skips.append(
                        Flattened1dConv(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            key=skip_key,
                        )
                    )
                elif channel_mlp_residual == "soft-gating":
                    channel_mlp_skips.append(
                        SoftGating(
                            ndim=ndim,
                            in_channels=out_channels,
                            out_channels=out_channels,
                        )
                    )
                elif channel_mlp_residual == "identity":
                    channel_mlp_skips.append(eqx.nn.Identity())
                elif channel_mlp_residual is None:
                    channel_mlp_skips.append(None)
                else:
                    raise ValueError(
                        f"'{channel_mlp_residual}' is not a valid skip connection."
                    )
            self.channel_mlp_residuals = tuple(channel_mlp_skips)
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
            self.channel_mlp_residuals = None

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Forward pass through n_layers of TFNO Blocks.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        if self.channel_mlps is not None:
            for tfno_layer, mlp_layer, res_op in zip(
                self.tfno_layers,
                self.channel_mlps,
                self.channel_mlp_residuals,
                strict=True,
            ):
                x = tfno_layer(x)
                if res_op is not None:
                    x = mlp_layer(x) + res_op(x)
                else:
                    x = mlp_layer(x)
            return x
        else:
            for layer in self.tfno_layers:
                x = layer(x)
            return x
