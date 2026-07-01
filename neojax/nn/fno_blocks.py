"""Implementation of general Fourier Neural Operator (FNO) block(s)."""

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.nn.normalization import InstanceNorm
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.resample import Resampler
from neojax.nn.skip_connections import Flattened1dConv, SoftGating
from neojax.nn.spectral_conv import SpectralConvNd
from neojax.tensor import BaseTensor


class FNOBlock(eqx.Module):
    r"""General FNO Block.

    FNO block composed of the global non-linear integral kernel, i.e.,
    $\mathcal{K}(\mathbf{v}) = \mathcal{F}^{-1}(R \cdot \mathcal{F}(\mathbf{v}))$,
    and an optional local operator $W \mathbf{v}$.
    A block can be further customized with Resnet-style residual
    connections around the FNO and normalization
    before the non-linear activation.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to retain
            across each spatial dimension.
            Must be a sequence of integers (e.g., `(16, 16)` for 2D)
            or single integer for 1D inputs.
        activation: Activation function to use.
            Defaults to `jax.nn.gelu`.
        local_operator: Type of local operator to use.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or `None`.
            Defaults to `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Defaults to `False`.
        normalization: Type of normalization to use. Applied after
            the spectral_op + local_op sum, before activation.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is `"layer"`.
        norm_groups: Number of groups to use if `normalization="group"`.
            Defaults to 1.
        use_fno_residual: Whether to use a Resnet-style residual
            connection around each FNO block. Improves stability.
            Default True.
        preactivation: Whether to apply the activation
            before the spectral convolution and skip connection.
            Defaults to `False`. This is an exclusive flag.
            If activation is applied before, it isn't applied after.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs of the spectral convolutions
            before calling `irfftn`.
            Default is True. If set to False, the outputs will
            not be hermitian-symmetric and the output will not
            be real-valued. Only considered for real-valued inputs.
        fft_norm: FFT normalization. Can be `None`, `"backward"`,
            `"ortho"` or `"forward"`. Default is `"forward"`.
        is_complex_data: Whether input data is complex valued.
            If True, uses full FFT. Default is False.
        resolution_scaling_factor: Factor by which to scale the domain
            resolution of the function. Default is None, no scaling.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not, should be a sequence of ranks.
            Default is None.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels * out_channels)) ** 0.5.
        factorization: Tensor factorization type. Can be a `BaseTensor`
            instance, a string ("tucker", "cp", "tt"), or None.
            Default is None.
        implementation: Weight reconstruction mode.
            Can be "reconstructed" or "factorized".
            Default is "factorized".
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **spectral_conv** (`SpectralConvNd`): The `SpectralConvNd` layer performing the operator integral.
        * **local_operator** (`Flattened1dConv | SoftGating | eqx.nn.Identity | None`): The initialized local operator layer or `None`.
        * **normalization** (`InstanceNorm | eqx.nn.GroupNorm | None`): Type of normalization to use. Applied after the spectral_op + local_op sum, before activation.
        * **activation** (`Callable`): The activation function.
        * **preactivation** (`bool`): Boolean flag indicating if preactivation is used.
        * **use_fno_residual** (`bool`): Whether to use a Resnet-style residual connection around each FNO block. Improves stability.

    Examples:
        ```python
        import jax.random as jr
        from neojax.nn.fno_blocks import FNOBlock
        import jax.numpy as jnp

        key = jr.key(0)
        # Initialize a 2D FNO Block
        fno_block = FNOBlock(
            key=key,
            in_channels=3,
            out_channels=8,
            modes=(4, 4),
            local_operator="linear"
        )
        # Input shape: (channels, height, width)
        x = jnp.ones((3, 32, 32))
        out = fno_block(x)
        ```
    """

    spectral_conv: SpectralConvNd
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
        activation: Callable = jax.nn.gelu,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        use_fno_residual: bool = True,
        preactivation: bool = False,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
        resolution_scaling_factor: float | int | None = None,
        ranks: int | Sequence[int] | None = None,
        init_std: float | Literal["auto"] = "auto",
        factorization: BaseTensor | Literal["tucker", "cp", "tt"] | None = None,
        implementation: Literal["reconstructed", "factorized"] = "factorized",
        separable: bool = False,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if not isinstance(separable, bool):
            raise ValueError("separable must be a boolean.")
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        if not isinstance(enforce_hermitian_symmetry, bool):
            raise ValueError("enforce_hermitian_symmetry must be a boolean.")
        if not isinstance(is_complex_data, bool):
            raise ValueError("is_complex_data must be a boolean.")

        if implementation not in ["reconstructed", "factorized"]:
            raise ValueError(
                "'implementation' must be one of ['reconstructed', 'factorized']."
            )

        if fft_norm not in ["forward", "backward", "ortho", None]:
            raise ValueError(
                "'fft_norm' must be one of ['forward', 'backward', 'ortho', None]."
            )

        if resolution_scaling_factor is not None:
            if not isinstance(resolution_scaling_factor, (int, float)):
                raise ValueError("resolution_scaling_factor must be an int, float, or None.")
            if resolution_scaling_factor <= 0:
                raise ValueError("resolution_scaling_factor must be positive.")

        if isinstance(modes, int):
            modes_tuple = (modes,)
        elif isinstance(modes, Sequence):
            modes_tuple = tuple(modes)
        else:
            raise ValueError("modes must be an int or a sequence of ints.")

        if len(modes_tuple) == 0:
            raise ValueError("modes sequence cannot be empty.")
        if not all(isinstance(m, int) and m > 0 for m in modes_tuple):
            raise ValueError("All modes must be positive integers.")

        if not isinstance(init_std, float) and init_std != "auto":
            raise ValueError("init_std must be a float or 'auto'.")

        if isinstance(factorization, str):
            if factorization not in ["tucker", "cp", "tt"]:
                raise ValueError("Passed 'factorization' string invalid.")
            if ranks is None:
                raise ValueError(
                    "ranks must be provided if factorization is specified as a string"
                )
            if isinstance(ranks, int):
                if ranks <= 0:
                    raise ValueError("ranks must be a positive integer.")
            elif isinstance(ranks, Sequence):
                if not all(isinstance(r, int) and r > 0 for r in ranks):
                    raise ValueError("All ranks must be positive integers.")
            else:
                raise ValueError("ranks must be an int, a sequence of ints, or None.")

        ndim = len(modes_tuple)
        fno_key, skip_key = jr.split(key, 2)
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

        if resolution_scaling_factor is None:
            resolution_scaling_factor = 1

        self.spectral_conv = SpectralConvNd(
            key=fno_key,
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            ranks=ranks,
            init_std=init_std,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            fft_norm=fft_norm,
            is_complex_data=is_complex_data,
            resolution_scaling_factor=resolution_scaling_factor,
            factorization=factorization,
            implementation=implementation,
            separable=separable,
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

    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Forward pass of FNO block.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        res = x
        if self.preactivation:
            x = self.activation(x)

        x_fft = self.spectral_conv(x)
        x_skip = self.local_operator(x) if self.local_operator is not None else 0

        if self.local_operator is not None and x_skip.shape != x_fft.shape:
            ndim = len(x_fft.shape) - 1
            axes = tuple(range(1, ndim + 1))
            resampler = Resampler(
                input_shape=x_skip.shape,
                res_scale=self.spectral_conv.resolution_scaling_factor,
                axes=axes,
                output_shape=x_fft.shape,
            )
            x_skip = resampler(x_skip)

        x = x_fft + x_skip

        if self.normalization is not None:
            x = self.normalization(x)

        if not self.preactivation:
            x = self.activation(x)

        if self.use_fno_residual:
            if res.shape == x.shape:
                x = x + res

        return x


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
            with 2 layers after each FNO block. Default is True.
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
            connection around each FNO block. Improves stability.
            Default True.
        local_operator: Type of local operator to use in the FNO block.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Default is `False`.
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
        channel_mlp_dropout: Dropout probability applied after each layer
            (except the last) of the channel MLPs. Defaults to 0.0.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs of the spectral convolutions
            before calling `irfftn`.
            Default is True.  If set to False, cuFFT on GPU may cause line artifacts
            when calling irfftn.
        fft_norm: FFT normalization. Can be `None`, `"backward"`,
            `"ortho"` or `"forward"`. Default is `"forward"`.
        is_complex_data: Whether input data is complex valued.
            If True, uses full FFT. Default is False.
        resolution_scaling_factor: Layerwise factor by which to scale the domain
            resolution of the function. Default is None. Passing a scalar scales the
            resolution by that value each layer. Passing a sequence, scales the i-th
            layer by the i-th value.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not, should be a sequence of ranks.
            Default is None.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels * out_channels)) ** 0.5.
        factorization: Tensor factorization type. Can be a `BaseTensor`
            instance, a string ("tucker", "cp", "tt"), or None.
            Default is None.
        implementation: Weight reconstruction mode.
            Can be "reconstructed" or "factorized".
            Default is "factorized".
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **fno_layers** (`tuple[FNOBlock, ...]`): The initialized `FNOBlock` layers.
        * **channel_mlps** (`tuple[PointwiseMLP, ...] | None`): The initialized `PointwiseMLP` layers, or None if `use_channel_mlp` is `False`.
        * **channel_mlp_residuals** (`tuple | None`): The residual connection instances or None.

    Examples:
        ```python
        import jax.random as jr
        import jax.numpy as jnp
        from neojax.nn.fno_blocks import FNOBlocks

        key = jr.key(0)

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

    ??? cite

        [Fourier Neural Operator for
        Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

        ```bibtex
        @inproceedings{
            li2021fourier,
            title={Fourier Neural Operator for Parametric Partial Differential Equations},
            author={Zongyi Li and Nikola Kovachki and
            Kamyar Azizzadenesheli and Burigede liu and
            Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
            booktitle={International Conference on Learning Representations},
            year={2021},
            url={https://openreview.net/forum?id=c8P9NQVtmnO}
        }
        ```

        [Neural Operator: Learning Maps Between
        Function Spaces With Applications to PDEs](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf)

        ```bibtex
        @article{kovachki2023neural,
            title={Neural operator: Learning maps between function spaces with applications to pdes},
            author={Kovachki, Nikola and Li, Zongyi and
            Liu, Burigede and Azizzadenesheli,
            Kamyar and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
            journal={Journal of Machine Learning Research},
            volume={24},
            number={89},
            pages={1--97},
            year={2023}
        }
        ```
    """

    fno_layers: tuple[FNOBlock, ...]
    channel_mlps: tuple[PointwiseMLP, ...] | None
    channel_mlp_residuals: tuple | None = None

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
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        use_fno_residual: bool = True,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "identity",
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        channel_mlp_dropout: float = 0.0,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
        resolution_scaling_factor: float | int | Sequence[float | int] | None = None,
        ranks: int | Sequence[int] | None = None,
        init_std: float | Literal["auto"] = "auto",
        factorization: BaseTensor | Literal["tucker", "cp", "tt"] | None = None,
        implementation: Literal["reconstructed", "factorized"] = "factorized",
        separable: bool = False,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if not isinstance(separable, bool):
            raise ValueError("separable must be a boolean.")
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        if not isinstance(enforce_hermitian_symmetry, bool):
            raise ValueError("enforce_hermitian_symmetry must be a boolean.")
        if not isinstance(is_complex_data, bool):
            raise ValueError("is_complex_data must be a boolean.")

        if implementation not in ["reconstructed", "factorized"]:
            raise ValueError(
                "'implementation' must be one of ['reconstructed', 'factorized']."
            )

        if fft_norm not in ["forward", "backward", "ortho", None]:
            raise ValueError(
                "'fft_norm' must be one of ['forward', 'backward', 'ortho', None]."
            )

        if isinstance(modes, int):
            modes_tuple = (modes,)
        elif isinstance(modes, Sequence):
            modes_tuple = tuple(modes)
        else:
            raise ValueError("modes must be an int or a sequence of ints.")

        if len(modes_tuple) == 0:
            raise ValueError("modes sequence cannot be empty.")
        if not all(isinstance(m, int) and m > 0 for m in modes_tuple):
            raise ValueError("All modes must be positive integers.")

        if not isinstance(init_std, float) and init_std != "auto":
            raise ValueError("init_std must be a float or 'auto'.")

        if isinstance(factorization, str):
            if factorization not in ["tucker", "cp", "tt"]:
                raise ValueError("Passed 'factorization' string invalid.")
            if ranks is None:
                raise ValueError(
                    "ranks must be provided if factorization is specified as a string"
                )
            if isinstance(ranks, int):
                if ranks <= 0:
                    raise ValueError("ranks must be a positive integer.")
            elif isinstance(ranks, Sequence):
                if not all(isinstance(r, int) and r > 0 for r in ranks):
                    raise ValueError("All ranks must be positive integers.")
            else:
                raise ValueError("ranks must be an int, a sequence of ints, or None.")

        fno_layers = []
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

        if isinstance(resolution_scaling_factor, (float, int)):
            resolution_scaling_factor = (resolution_scaling_factor,) * n_layers
        elif resolution_scaling_factor is None:
            resolution_scaling_factor = (1,) * n_layers
        elif isinstance(resolution_scaling_factor, Sequence):
            resolution_scaling_factor = tuple(resolution_scaling_factor)
        else:
            raise ValueError("Invalid 'resolution_scaling_factor'.")

        for i in range(n_layers):
            fno_key, key = jr.split(key, 2)
            if i > 0:
                in_channels_layer = out_channels
            else:
                in_channels_layer = in_channels
            fno_layers.append(
                FNOBlock(
                    key=fno_key,
                    in_channels=in_channels_layer,
                    out_channels=out_channels,
                    modes=modes,
                    activation=activations[i],
                    local_operator=local_operator,
                    use_local_operator_bias=use_local_operator_bias,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    use_fno_residual=use_fno_residual,
                    preactivation=preactivation,
                    enforce_hermitian_symmetry=enforce_hermitian_symmetry,
                    fft_norm=fft_norm,
                    is_complex_data=is_complex_data,
                    resolution_scaling_factor=resolution_scaling_factor[i],
                    ranks=ranks,
                    init_std=init_std,
                    factorization=factorization,
                    implementation=implementation,
                    separable=separable,
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
            ndim = len(modes_tuple)
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
                    dropout=channel_mlp_dropout,
                )
                for i in range(n_layers)
            ]
            self.channel_mlps = tuple(channel_mlps)
        else:
            self.channel_mlps = None
            self.channel_mlp_residuals = None

    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass through n_layers of FNO Blocks.

        Args:
            x: Input array.
            key: PRNG key used for dropout masks.
            inference: If True, dropout is disabled.

        Returns:
            Output array.
        """
        if self.channel_mlps is not None:
            n_layers = len(self.fno_layers)
            if key is not None and not inference:
                keys = jr.split(key, n_layers)
            else:
                keys = [None] * n_layers

            for i, (fno_layer, mlp_layer, res_op) in enumerate(
                zip(
                    self.fno_layers,
                    self.channel_mlps,
                    self.channel_mlp_residuals,
                    strict=True,
                )
            ):
                x = fno_layer(x)
                mlp_out = mlp_layer(x, key=keys[i], inference=inference)
                if res_op is not None:
                    x = mlp_out + res_op(x)
                else:
                    x = mlp_out
            return x
        else:
            for layer in self.fno_layers:
                x = layer(x)
            return x
