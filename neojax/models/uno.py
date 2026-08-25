"""Implementation of a U-shaped Neural Operator."""

from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.domain_padding import DomainPadding
from neojax.nn.fno_blocks import FNOBlock
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.nn.resample import Resampler
from neojax.nn.skip_connections import Flattened1dConv, SoftGating, make_skip_connection
from neojax.tensor import BaseTensor


@final
class UNO(BaseNO):
    """U-shaped Neural Operator.

    The architecture is described in the reference publication.
    It extends the standard FNO with a U-shaped encoder-decoder
    structure and horizontal skip connections between corresponding
    encoder and decoder layers.

    Each FNO layer has its own number of output channels, Fourier modes,
    and resolution scaling factor. Horizontal skip connections concatenate
    features from an encoder layer to a decoder layer along the channel
    dimension. An optional skip-connection projection is applied before
    concatenation.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels
            (e.g. coordinates + initial conditions).
        out_channels: Number of output channels (e.g. solution field).
        hidden_channels: Initial width of the UNO. This significantly affects
            the number of parameters of the UNO. Good starting point can be 64,
            and then increased if more expressivity is needed.
            (Update `lift_channel_ratio` and `proj_channel_ratio` accordingly).
        uno_out_channels: Number of output channels of each Fourier layer,
            e.g. for a 5 layer UNO `uno_out_channels` can be [32, 64, 64, 64, 32].
        uno_modes: Number of Fourier modes to use in integral operation of each
            Fourier layer (along each dimension). For example in a 5 layer UNO with 2D inputs:
            [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]].
        uno_scalings: Scaling factor for each Fourier layer. For e.g. a 5 layer UNO with 2D inputs,
            the `uno_scalings` can be [[0.5, 0.5], [0.5, 0.5], [1, 1], [2, 2], [2, 2]].
            If inner elements are scalars, the same scaling is applied to all spatial dims.
        n_fno_layers: Number of Fourier layers. Default is 4.
        n_lift_layers: Number of lifting layers. Default is 2.
        n_proj_layers: Number of projection layers. Default is 2.
        lift_channel_ratio: Ratio of lifting channels to hidden_channels.
            The number of lifting channels in the lifting block
            of the UNO is lifting_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        proj_channel_ratio: Ratio of projection channels to hidden_channels.
            The number of projection channels in the projection block
            of the UNO is projection_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        positional_embedding: Positional embedding to apply
            to last channels of raw input before passing through UNO.
            Default is None. Pass ``"grid"`` for a regular ``GridEmbeddingNd``.
        horizontal_skip_map: A dictionary ``{b: a, ...}`` denoting horizontal skip connections
            from the a-th layer to the b-th layer. If None, default symmetric skip connections
            are applied (mirroring encoder layers to decoder layers).
            Layer indices are zero-based.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            after each FNO block. Defaults to True.
        channel_mlp_expansion: Expansion factor for hidden dimension
            in the channel MLPs. Defaults to 0.5.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is ``jax.nn.gelu``.
        channel_mlp_dropout: Dropout probability applied after each layer
            (except the last) of the channel MLPs. Defaults to 0.0.
        channel_mlp_residual: Type of skip connection around channel MLPs.
            Can be ``"linear"``, ``"soft-gating"``, ``"identity"``, or None.
            Defaults to ``"soft-gating"``.
        use_fno_residual: Whether to use residual connection
            around FNO blocks. Default is True.
        activation: Activation function used within the layers.
            Defaults to ``jax.nn.gelu``.
        normalization: Type of normalization to use in FNO blocks.
            Can be ``"layer"``, ``"instance"``, ``"group"`` or None.
            Default is ``"layer"``.
        norm_groups: Number of groups for group normalization.
            Default is 1.
        preactivation: Whether to use pre-activation style blocks.
            Default is False.
        local_operator: Type of skip connection inside FNO blocks.
            Can be ``"linear"``, ``"soft-gating"``, ``"identity"``, or None.
            Defaults to ``"linear"``.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Default is ``False``.
        horizontal_residual: Type of skip connection to use for horizontal
            skip paths. Can be ``"linear"``, ``"soft-gating"``, ``"identity"``, or None.
            Defaults to ``"linear"``.
        domain_padding: Percentage of padding to use.
            If single float, this padding is used for all dims.
            Sequence of floats indicates padding percentage per dim.
            Default is None, no padding.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs of the spectral convolutions
            before calling ``irfftn``.
            Default is True. If set to False, cuFFT on GPU may cause line artifacts
            when calling ``irfftn``.
        fft_norm: FFT normalization. Can be ``None``, ``"backward"``,
            ``"ortho"`` or ``"forward"``. Default is ``"forward"``.
        is_complex_data: Whether the input data is complex valued. Default is False.
        ranks: Number of ranks to contract the spectral tensors to.
            Default is None.
        init_std: Standard deviation to use for weight initialization.
            Default is ``"auto"``.
        factorization: Tensor factorization type.
            Default is None.
        implementation: Weight reconstruction mode.
            Default is ``"factorized"``.
        separable: Whether to use separable implementation of contraction.
            Default is False.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **positional_embedding** (`GridEmbeddingNd | None`): Positional embedding.
        * **lifting** (`PointwiseMLP`): Maps inputs to hidden dimension.
        * **fno_blocks** (`tuple[FNOBlock, ...]`): Per-layer FNO blocks with layer-specific channels,
            modes, and scalings.
        * **channel_mlps** (`tuple[PointwiseMLP, ...] | None`): Per-layer channel MLPs.
        * **channel_mlp_residuals** (`tuple | None`): Per-layer channel MLP residual connections.
        * **horizontal_skip_connections** (`dict[int, Flattened1dConv | SoftGating | eqx.nn.Identity | None]`):
            Skip connections for horizontal paths.
        * **projection** (`PointwiseMLP`): Maps latent features to output channels.
        * **padding** (`DomainPadding | None`): Percentage of domain padding to use.
        * **n_fno_layers** (`int`): Number of FNO layers.
        * **end_to_end_scaling_factor** (`list[int | float]`): Cumulative scaling factor per spatial dim.
        * **horizontal_skip_map** (`dict[int, int] | None`): Maps decoder layer indices
            to encoder layer indices for skip connections.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.models.uno import UNO

        key = jr.key(0)
        model = UNO(
            key=key,
            in_channels=3,
            out_channels=1,
            hidden_channels=16,
            uno_out_channels=[16, 32, 32, 16],
            uno_modes=[[8, 8], [8, 8], [8, 8], [8, 8]],
            uno_scalings=[[0.5, 0.5], [0.5, 0.5], [2, 2], [2, 2]],
            n_fno_layers=4,
        )
        x = jnp.ones((3, 32, 32))
        out = model(x)
        ```

    ??? cite

        [U-no: U-shaped neural operators](https://arxiv.org/pdf/2204.11127)

        ```bibtex
        @article{rahman2022u,
            title={U-no: U-shaped neural operators},
            author={Rahman, Md Ashiqur and Ross,
            Zachary E and Azizzadenesheli, Kamyar},
            journal={arXiv preprint arXiv:2204.11127},
            year={2022}
        }
        ```
    """

    positional_embedding: GridEmbeddingNd | None
    lifting: PointwiseMLP
    fno_blocks: tuple[FNOBlock, ...]
    channel_mlps: tuple[PointwiseMLP, ...] | None
    channel_mlp_residuals: tuple | None
    horizontal_skip_connections: dict[
        int, Flattened1dConv | SoftGating | eqx.nn.Identity | None
    ]
    projection: PointwiseMLP
    padding: DomainPadding | None
    n_fno_layers: int = eqx.field(static=True)
    end_to_end_scaling_factor: list[int | float] = eqx.field(static=True)
    horizontal_skip_map: dict[int, int] = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        uno_out_channels: Sequence[int],
        uno_modes: Sequence[Sequence[int]],
        uno_scalings: Sequence[float | int] | Sequence[Sequence[float | int]],
        n_fno_layers: int = 4,
        n_lift_layers: int = 2,
        n_proj_layers: int = 2,
        lift_channel_ratio: float = 2.0,
        proj_channel_ratio: float = 2.0,
        positional_embedding: GridEmbeddingNd | Literal["grid"] | None = None,
        horizontal_skip_map: dict[int, int] | None = None,
        use_channel_mlp: bool = True,
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        channel_mlp_dropout: float = 0.0,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        use_fno_residual: bool = True,
        activation: Callable = jax.nn.gelu,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        preactivation: bool = False,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        horizontal_residual: Literal["linear", "soft-gating", "identity"]
        | None = "linear",
        domain_padding: float | Sequence[float] | None = None,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
        ranks: int | Sequence[int] | None = None,
        init_std: float | Literal["auto"] = "auto",
        factorization: BaseTensor | Literal["tucker", "cp", "tt"] | None = None,
        implementation: Literal["reconstructed", "factorized"] = "factorized",
        separable: bool = False,
    ) -> None:
        lkey, pkey, fno_key, skip_key = jr.split(key, 4)
        self.n_fno_layers = n_fno_layers
        dim = len(uno_modes[0])

        # Positional embedding
        if positional_embedding is None:
            self.positional_embedding = None
        elif positional_embedding == "grid":
            self.positional_embedding = GridEmbeddingNd(in_channels, dim)
        elif isinstance(positional_embedding, GridEmbeddingNd):
            self.positional_embedding = positional_embedding
        else:
            raise ValueError(f"Positional embedding {positional_embedding} invalid.")

        # Construct default horizontal skip map: mirror encoder to decoder
        if horizontal_skip_map is None:
            horizontal_skip_map = {}
            for i in range(0, n_fno_layers // 2):
                horizontal_skip_map[n_fno_layers - i - 1] = i
        self.horizontal_skip_map = horizontal_skip_map

        # Normalize uno_scalings to nested lists
        uno_scalings_list: list[list[float | int]]
        if isinstance(uno_scalings[0], float | int):
            uno_scalings_list = [[us] * dim for us in uno_scalings]
        else:
            uno_scalings_list = [list(s) for s in uno_scalings]

        # Compute end-to-end cumulative scaling factor per spatial dim
        self.end_to_end_scaling_factor = [1.0] * dim
        for k in uno_scalings_list:
            self.end_to_end_scaling_factor = [
                i * j for (i, j) in zip(self.end_to_end_scaling_factor, k, strict=True)
            ]

        cumulative_scaling = 1.0
        for f in self.end_to_end_scaling_factor:
            cumulative_scaling *= f
        if domain_padding is not None:
            self.padding = DomainPadding(
                domain_padding, resolution_scaling_factor=cumulative_scaling
            )
        else:
            self.padding = None

        lift_hidden_channels = int(hidden_channels * lift_channel_ratio)
        lifting_layers = (
            [in_channels]
            + [lift_hidden_channels] * (n_lift_layers - 1)
            + [hidden_channels]
        )
        self.lifting = PointwiseMLP(
            key=lkey,
            layers=tuple(lifting_layers),
            activations=activation,
        )

        final_fno_channels = uno_out_channels[-1]
        proj_hidden_channels = int(hidden_channels * proj_channel_ratio)
        projection_layers = (
            [final_fno_channels]
            + [proj_hidden_channels] * (n_proj_layers - 1)
            + [out_channels]
        )
        self.projection = PointwiseMLP(
            key=pkey,
            layers=tuple(projection_layers),
            activations=activation,
        )

        # Per-layer FNO blocks
        fno_blocks = []
        fno_keys = jr.split(fno_key, n_fno_layers)

        for i in range(n_fno_layers):
            # Input channels for this layer
            if i == 0:
                layer_in_channels = hidden_channels
            else:
                layer_in_channels = uno_out_channels[i - 1]

            # If horizontal skip, concatenate channels
            if i in self.horizontal_skip_map:
                source_layer = self.horizontal_skip_map[i]
                layer_in_channels += uno_out_channels[source_layer]

            layer_out_channels = uno_out_channels[i]

            fno_blocks.append(
                FNOBlock(
                    key=fno_keys[i],
                    in_channels=layer_in_channels,
                    out_channels=layer_out_channels,
                    modes=tuple(uno_modes[i]),
                    activation=activation,
                    local_operator=local_operator,
                    use_local_operator_bias=use_local_operator_bias,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    use_fno_residual=use_fno_residual,
                    preactivation=preactivation,
                    enforce_hermitian_symmetry=enforce_hermitian_symmetry,
                    fft_norm=fft_norm,
                    is_complex_data=is_complex_data,
                    resolution_scaling_factor=None,
                    ranks=ranks,
                    init_std=init_std,
                    factorization=factorization,
                    implementation=implementation,
                    separable=separable,
                )
            )
        self.fno_blocks = tuple(fno_blocks)

        # Horizontal skip connection projections
        horizontal_skip_connections: dict[
            int, Flattened1dConv | SoftGating | eqx.nn.Identity | None
        ] = {}
        skip_keys = jr.split(skip_key, max(len(self.horizontal_skip_map), 1))
        for idx, (target_layer, source_layer) in enumerate(
            self.horizontal_skip_map.items()
        ):
            horizontal_skip_connections[target_layer] = make_skip_connection(
                key=skip_keys[idx],
                kind=horizontal_residual,
                in_channels=uno_out_channels[source_layer],
                out_channels=uno_out_channels[source_layer],
                ndim=dim,
            )
        self.horizontal_skip_connections = horizontal_skip_connections

        if use_channel_mlp:
            mlp_keys = jr.split(jr.fold_in(key, 42), n_fno_layers)
            channel_mlps = []
            channel_mlp_skips = []
            for i in range(n_fno_layers):
                ch = uno_out_channels[i]
                if channel_mlp_expansion is not None:
                    hidden_ch = round(channel_mlp_expansion * ch)
                else:
                    hidden_ch = ch

                channel_mlps.append(
                    PointwiseMLP(
                        key=mlp_keys[i],
                        layers=(ch, hidden_ch, ch),
                        activations=channel_mlp_activations,
                        dropout=channel_mlp_dropout,
                    )
                )
                mlp_skip_key = jr.fold_in(mlp_keys[i], 99)
                channel_mlp_skips.append(
                    make_skip_connection(
                        key=mlp_skip_key,
                        kind=channel_mlp_residual,
                        in_channels=ch,
                        out_channels=ch,
                        ndim=dim,
                    )
                )
            self.channel_mlps = tuple(channel_mlps)
            self.channel_mlp_residuals = tuple(channel_mlp_skips)
        else:
            self.channel_mlps = None
            self.channel_mlp_residuals = None

    @override
    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the UNO model.

        The input is lifted, passed through the U-shaped FNO layers
        with horizontal skip connections, and projected to the output.

        Args:
            x: Input array of shape ``(in_channels, d1, ..., dN)``.
            key: PRNG key used for dropout masks in channel MLPs.
            inference: If True, dropout is disabled.

        Returns:
            The predicted field of shape ``(out_channels, d1', ..., dN')``,
            where ``d_i'`` depends on the cumulative scaling factors.
        """
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        original_shape = x.shape
        if self.padding is not None:
            x = self.padding.pad(x)

        x = self.lifting(x)

        # Outputs for horizontal skip connections
        layer_outputs: dict[int, Inexact[Array, "c ..."]] = {}

        # Source layers for horizontal skips
        source_layers = set(self.horizontal_skip_map.values())

        if self.channel_mlps is not None:
            if key is not None and not inference:
                mlp_keys = jr.split(key, self.n_fno_layers)
            else:
                mlp_keys = [None] * self.n_fno_layers

        for i in range(self.n_fno_layers):
            # If layer receives horizontal skip, resample and concatenate
            if i in self.horizontal_skip_map:
                source_layer = self.horizontal_skip_map[i]
                skip_features = layer_outputs[source_layer]

                # Apply horizontal skip connection projection
                skip_conn = self.horizontal_skip_connections[i]
                if skip_conn is not None:
                    skip_features = skip_conn(skip_features)

                # Resample skip features to match spatial resolution
                if skip_features.shape[1:] != x.shape[1:]:
                    ndim = len(x.shape) - 1
                    axes = tuple(range(1, ndim + 1))
                    resampler = Resampler(
                        input_shape=skip_features.shape,
                        res_scale=1,
                        axes=axes,
                        output_shape=x.shape,
                    )
                    skip_features = resampler(skip_features)

                # Concatenate with skip features
                x = jnp.concatenate([x, skip_features], axis=0)

            x = self.fno_blocks[i](x)

            # Save output for potential skip connections
            if i in source_layers:
                layer_outputs[i] = x

            if self.channel_mlps is not None:
                mlp_out = self.channel_mlps[i](x, key=mlp_keys[i], inference=inference)
                if self.channel_mlp_residuals[i] is not None:
                    x = mlp_out + self.channel_mlp_residuals[i](x)
                else:
                    x = mlp_out

        x = self.projection(x)

        if self.padding is not None:
            x = self.padding.unpad(x, original_shape)

        return x
