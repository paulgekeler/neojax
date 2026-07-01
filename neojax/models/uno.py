"""Implementation of a U-shaped Neural Operator."""

from collections.abc import Callable, Sequence
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.nn.domain_padding import DomainPadding
from neojax.nn.fno_blocks import FNOBlock, FNOBlocks
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.nn.spectral_conv import SpectralConvNd
from neojax.nn.tfno_blocks import TFNOBlock, TFNOBlocks
from neojax.nn.tucker_spectral_conv import TuckerSpectralConvNd
from neojax.models.baseno import BaseNO

@final
class UNO(BaseNO):
    """U-shaped Neural Operator.

    The architecture is described in the reference publication.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels
            (e.g. coordinates + initial conditions).
        out_channels: Number of output channels (e.g. solution field).
        hidden_channels: Initial width of the UNO. This significantly affects
            the number of parameters of the UNO. Good starting point can be 64,
            and then increased if more expressivity is needed.
            (Update `lift_channel_ratio` and `proj_channel_ratio` accordingly).
        uno_out_channels: Number of output channnels of each Fourier layer,
            e.g. for a 5 layer UNO `uno_out_channels` can be [32, 64, 64, 64, 32].
        uno_modes: Number of Fourier modes to use in integral operation of each
            Fourier layer (along each dimension). For example in a 3 layer UNO with 2D inputs:
            ([[5, 4], [5, 4], [5, 4]]).
        uno_scalings: Scaling factor for each Fourier layer. For e.g. a 3 layer UNO with 2D inputs,
            the `uno_scalings` can be [[0.5, 0.5], [1, 1], [2, 2]].
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
            Default is "grid", a regular `GridEmbeddingNd` on a ((0, 1), ...) grid.
            Passing `None` does nothing.
        horizontal_skip_map: A dictionary `{b: a, ...}` denoting horizontal skip connections
            from the a-th layer to the b-th layer. If None, default skip connections are applied.
            Layer indices are zero based.
        use_fno_residual: Whether to use residual connection
            around FNO blocks. Default is True.
        activation: Activation function used within the layers.
            Defaults to `jax.nn.gelu`.
        normalization: Type of normalization to use in FNO blocks.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is `"layer"`.
        norm_groups: Number of groups for group normalization.
            Default is 1.
        preactivation: Whether to use pre-activation style blocks.
            Default is False.
        local_operator: Type of skip connection inside FNO blocks.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Default is `False`.
        channel_mlp_expansion: Expansion factor for hidden dimension
            in the channel MLPs. Defaults to 0.5.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is `jax.nn.gelu`.
        channel_mlp_residual: Type of skip connection around channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"soft-gating"`.
        horizontal_residual: Type of skip connection to use in horizontal connections.
        integral_operator: The integral operator to use in the FNO layers.
            Default is standard spectral convolution.
        operator_block: The operator block to use in the FNO layers.
            Default is standard FNO block. Has to be compatible with `integral_operator`.
        additional_int_op_kwargs:
        additional_op_kwargs:
        domain_padding: Percentage of padding to use.
            If single float, this padding is used for all dims.
            Sequence of floats indicates padding percentage per dim.
            Default is None, no padding.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs of the spectral convolutions
            before calling `irfftn`.
            Default is True. If set to False, cuFFT on GPU may cause line artifacts
            when calling irfftn.
        fft_norm: FFT normalization. Can be `None`, `"backward"`,
            `"ortho"` or `"forward"`. Default is `"forward"`.
        is_complex_data: Whether the input data is complex valued. Default is False.

    ??? info "Internal Attributes"
        * ** ** ():

    Example:
        ```python
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
    fno_blocks: tuple[FNOBlock, ...] | tuple[TFNOBlocks, ...]
    projection: PointwiseMLP
    horizontal_residuals: dict[int, eqx.Module]
    padding: DomainPadding | None
    n_fno_layers: int = eqx.field(static=True)
    end_to_end_scaling_factor: float = eqx.field(static=True)
    horizontal_skip_map: dict[int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        uno_out_channels: Sequence[int],
        uno_modes: Sequence[Sequence[int]],
        uno_scalings: Sequence[float | int] | Sequence[Sequence[float | int]],
        n_fno_layers: int | None = 4,
        n_lift_layers: int = 2,
        n_proj_layers: int = 2,
        lift_channel_ratio: float = 2.0,
        proj_channel_ratio: float = 2.0,
        positional_embedding: GridEmbeddingNd | Literal["grid"] | None = None,
        horizontal_skip_map: dict[int, int] | None = None,
        use_fno_residual: bool = True,
        activation: Callable = jax.nn.gelu,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        preactivation: bool = False,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        horizontal_residual: Literal["linear", "soft-gating", "identity"]
        | None = "linear",
        integral_operator: SpectralConvNd | TuckerSpectralConvNd = SpectralConvNd,
        operator_block: FNOBlock | TFNOBlock = FNOBlock,
        additional_int_op_kwargs: dict[str, Any] | None = None,
        additional_op_kwargs: dict[str, Any] | None = None,
        domain_padding: float | Sequence[float] | None = None,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: str | None = "forward",
        is_complex_data: bool = False,
    ) -> None:
        lkey, pkey, fno_key = jr.split(key, 3)
        self.n_fno_layers = n_fno_layers
        dim = len(uno_modes[0])
        if positional_embedding is None:
            self.positional_embedding = None
        elif positional_embedding == "grid":
            self.positional_embedding = GridEmbeddingNd(in_channels, dim)
        elif isinstance(positional_embedding, GridEmbeddingNd):
            self.positional_embedding = positional_embedding
        else:
            raise ValueError(f"Positional embedding {positional_embedding} invalid.")
        # TODO: add input verification

        if domain_padding is not None:
            self.padding = DomainPadding(domain_padding)
        else:
            self.padding = None

        # construct default horizontal skip map
        if horizontal_skip_map is None:
            horizontal_skip_map = {}
            for i in range(0, n_fno_layers // 2):
                horizontal_skip_map[n_fno_layers - i - 1] = i
        self.horizontal_skip_map = horizontal_skip_map

        # compute end-to-end scaling factor
        # uno_scalings may be single or nested list
        if isinstance(uno_scalings[0], (float, int)):
            uno_scalings = [[us] * dim for us in uno_scalings]
        self.end_to_end_scaling_factor = [1] * dim
        for k in uno_scalings:
            self.end_to_end_scaling_factor = [i * j for (i, j) in zip(self.end_to_end_scaling_factor, k, strict=True)]


        lift_hidden_channels = int(hidden_channels * lift_channel_ratio)
        lifting_layers = (
            [in_channels]
            + [lift_hidden_channels] * (n_lift_layers - 1)
            + [hidden_channels]
        )
        self.lifting = PointwiseMLP(
            key=lkey,
            layers=lifting_layers,
            activations=activation,
        )
        proj_hidden_channels = int(hidden_channels * proj_channel_ratio)
        projection_layers = (
            [hidden_channels]
            + [proj_hidden_channels] * (n_proj_layers - 1)
            + [out_channels]
        )
        self.projection = PointwiseMLP(
            key=pkey,
            layers=tuple(projection_layers),
            activations=activation,
        )

        fno_blocks = []
        horizontal_residuals = {}
        for i in range(self.n_fno_layers):
            if i in self.horizontal_skip_map.keys():
                prev_out = prev_out + uno_out_channels[self.horizontal_skip_map[i]]

            if 
            fno_blocks.append(operator_block())

        # TODO: add resolution_scaling_factor to SpectralConvNd and (T)FNOBlocks

    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the UNO model.

        Args:
            x: Input array of shape `(in_channels, d1, ..., dN)`.

        Returns:
            The predicted field of shape `(out_channels, d1, ..., dN)`.
        """
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # TODO: check if lifting, proj and padding should be swapped
        original_shape = x.shape
        if self.padding is not None:
            x = self.padding.pad(x)

        x = self.lifting(x)
        # iterate over layers and horizontal skips
        x = self.fno_blocks(x)

        x = self.projection(x)

        if self.padding is not None:
            x = self.padding.unpad(x, original_shape)

        return x
