"""Implementation of a Tucker-factorized FNO.

Pre-configured TFNO for end users.
(Same as a FNO with configured tucker factorization).
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal, final

import jax
from jaxtyping import Array, Inexact, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.models.fno import FNO
from neojax.nn.positional_embedding import GridEmbeddingNd


@final
class TFNO(BaseNO):
    """General n-dimensional Tucker-factorized Fourier Neural Operator (TFNO).

    The model consists of a lifting layer that maps
    the input to a higher-dimensional latent space,
    a sequence of FNO blocks (spectral convolutions + skip connections)
    optionally interleaved with pointwise MLPs,
    and a final projection layer.
    The spectral convolutions use a Tucker-factorized weight tensor.

    The implementation is as in the references.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels
            (e.g., coordinates + initial conditions).
        out_channels: Number of output channels (e.g., solution field).
        hidden_channels: Hidden channel dimension (latent width)
            used throughout the FNO blocks. This significantly affects
            the number of parameters. Good starting point is 64 and
            increase if needed. Update `lift_channel_ratio` and
            `proj_channel_ratio` accordingly.
            They scale proportional to hidden_channels.
        n_layers: Number of consecutive FNO blocks.
        modes: Number of Fourier modes to retain
            across each spatial dimension.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not, should be a sequence of ranks.
        activation: Activation function used within the layers.
            Defaults to `jax.nn.gelu`.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            after each FNO block. Defaults to `True`.
        local_operator: Type of skip connection inside FNO blocks.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Default is `False`.
        channel_mlp_residual: Type of skip connection around channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"soft-gating"`.
        channel_mlp_expansion: Expansion factor for hidden dimension
            in the channel MLPs. Defaults to 0.5.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is `jax.nn.gelu`.
        channel_mlp_dropout: Dropout probability applied after each layer
            (except the last) of the channel MLPs. Defaults to 0.0.
        normalization: Type of normalization to use in FNO blocks.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is `"layer"`.
        norm_groups: Number of groups for group normalization.
            Default is 1.
        use_fno_residual: Whether to use residual connection
            around FNO blocks. Default is True.
        preactivation: Whether to use pre-activation style blocks.
            Default is False.
        n_lift_layers: Number of layers in the lifting MLP.
            Defaults to 2.
        n_proj_layers: Number of layers in the projection MLP.
            Defaults to 2.
        lift_channel_ratio: Ratio of lifting channels
            to hidden_channels.
            The number of lifting channels in the lifting block
            of the TFNO is lifting_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        proj_channel_ratio: Ratio of projection channels
            to hidden_channels.
            The number of projection channels in the projection block
            of the TFNO is projection_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        positional_embedding: Positional embedding to apply
            to last channels of raw input before passing through UNO.
            Default is "grid", a regular `GridEmbeddingNd` on a ((0, 1), ...) grid.
            Passing `None` does nothing.
        domain_padding: Percentage of domain padding to use.
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
        is_complex_data: Whether input data is complex valued.
            If True, uses full FFT. Default is False.
        resolution_scaling_factor: Layerwise factor by which to scale the domain
            resolution of the function. Default is None. Passing a scalar scales the
            resolution by that value each layer. Passing a sequence, scales the i-th
            layer by the i-th value.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'.
        implementation: Weight reconstruction mode.
            Can be "reconstructed" or "factorized".
            Default is "factorized".
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **positional_embedding** (`GridEmbeddingNd | None`): Positional embedding
            to apply to last channels of raw input before passing through TFNO.
        * **lifting** (`PointwiseMLP`): The `PointwiseMLP` used to lift inputs to the hidden `hidden_channels`.
        * **fno_blocks** (`FNOBlocks`): The `FNOBlocks` sequence containing spectral convolutions.
        * **projection** (`PointwiseMLP`): The `PointwiseMLP` used to project latent features to `out_channels`.
        * **padding** (`DomainPadding | None`): Percentage of domain padding to use.

    ??? cite

        [Fourier Neural Operator for Parametric Partial Differential Equations](
        https://arxiv.org/abs/2010.08895)

        ```bibtex
        @inproceedings{
            li2021fourier,
            title={Fourier Neural Operator for
            Parametric Partial Differential Equations},
            author={Zongyi Li and Nikola Kovachki and
            Kamyar Azizzadenesheli and Burigede liu and
            Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
            booktitle={International Conference on Learning Representations},
            year={2021},
            url={https://openreview.net/forum?id=c8P9NQVtmnO}
        }
        ```

        [Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](
        https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf)

        ```bibtex
        @article{kovachki2023neural,
            title={Neural operator: Learning maps between
            function spaces with applications to pdes},
            author={Kovachki, Nikola and Li, Zongyi and Liu,
            Burigede and Azizzadenesheli, Kamyar and Bhattacharya,
            Kaushik and Stuart, Andrew and Anandkumar, Anima},
            journal={Journal of Machine Learning Research},
            volume={24},
            number={89},
            pages={1--97},
            year={2023}
        }
        ```

    !!! info
        This model just wraps a FNO and configures the factorization as "tucker".
        For more details see `neojax.models.FNO`.
    """

    fno: FNO

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        modes: Sequence[int],
        ranks: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        use_channel_mlp: bool = True,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        channel_mlp_dropout: float = 0.0,
        normalization: Literal["layer", "instance", "group"] | None = "layer",
        norm_groups: int = 1,
        use_fno_residual: bool = True,
        preactivation: bool = False,
        n_lift_layers: int = 2,
        n_proj_layers: int = 2,
        lift_channel_ratio: float = 2.0,
        proj_channel_ratio: float = 2.0,
        positional_embedding: GridEmbeddingNd | Literal["grid"] | None = None,
        domain_padding: float | Sequence[float] | None = None,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
        resolution_scaling_factor: float | int | Sequence[float | int] | None = None,
        init_std: float | Literal["auto"] = "auto",
        implementation: Literal["reconstructed", "factorized"] = "factorized",
        separable: bool = False,
    ) -> None:
        self.fno = FNO(
            key=key,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            activation=activation,
            use_channel_mlp=use_channel_mlp,
            local_operator=local_operator,
            use_local_operator_bias=use_local_operator_bias,
            channel_mlp_residual=channel_mlp_residual,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_activations=channel_mlp_activations,
            channel_mlp_dropout=channel_mlp_dropout,
            normalization=normalization,
            norm_groups=norm_groups,
            use_fno_residual=use_fno_residual,
            preactivation=preactivation,
            n_lift_layers=n_lift_layers,
            n_proj_layers=n_proj_layers,
            lift_channel_ratio=lift_channel_ratio,
            proj_channel_ratio=proj_channel_ratio,
            positional_embedding=positional_embedding,
            domain_padding=domain_padding,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            fft_norm=fft_norm,
            is_complex_data=is_complex_data,
            resolution_scaling_factor=resolution_scaling_factor,
            ranks=ranks,
            init_std=init_std,
            factorization="tucker",
            implementation=implementation,
            separable=separable,
        )

    @override
    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the TFNO model."""
        return self.fno(x, key=key, inference=inference)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup to the underlying FNO instance."""
        return getattr(self.fno, name)
