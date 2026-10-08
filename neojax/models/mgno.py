"""Implementation of the Multigrid Neural Operator (MgNO)."""

from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.mg_conv import MgConv
from neojax.nn.pointwise_mlp import PointwiseMLP


@final
class MgNO(BaseNO):
    """Multigrid Neural Operator (MgNO).

    Based on the mathematical structure of multigrid methods.

    This class wraps all `MgNO` variants of the [reference
    implementation](https://github.com/xlliu2017/MgNO/tree/main)
    in a single class. Use the `MgNO.make_mgno_type` convenience
    function to use one of the pre-configured variants.

    Args:
        key: PRNG key for parameter initialization.
        n_layers: The number of `MgConv` blocks (and optionally 1x1 Convs) to use.
            Varies with model variant (some add an additional output or input `MgConv` block).
        n_its_per_level: Number of iterations per level. A sequence containing
            a pair `[n_pre, n_post]` (pre-smoothing, post-smoothing) per level
            in the V-cycle or a sequence of ints specifying only pre-smoothing iterations
            per level (as is the case in the `MgNO_helm` variant of the original publication).
        n_channels_u: Number of channels of the solution field.
        n_channels_f: Number of channels of the data field.
        proj_hidden_dim: Whether to use an additional hidden dimension
            in the projection MLP with this dimension and intermediate GeLU activation.
            Default is `None`, that is a single projection layer.
        use_local_pointwise: Whether to use a parallel local (pointwise), non-local (`MgConv`)
            architecture. Requires `True` for the `MgNO_DC...` variants of the original
            publication. Default is `False`.
        activation: The type of non-linear activation function to use after each layer.
            Default is `"gelu"`.
        use_residual_skip: Whether to use a residual skip connection around all layers.
            Requires `True` for the `MgNO_NS` variant of the original publication.
            Default is `True`.
        use_proj_mgconv: Whether to use another `MgConv` block as projection instead of
            a 1x1 Conv. Requires `True` for the `MgNO_helm2` variant of the
            original publication. Default is `False`.
        padding_mode: Padding mode for the convolutions. Requires `"reflect"`
            for the `MgNO_helm` variant of the original publication.
            Default is `"zeros"`.
        padding_mode_smoothing: Padding mode for the smoothing iterations. If set,
            overrides `padding_mode` for smoothing. Requires `"reflect"` for the
            `MgNO_helm3` variant of the original publication.
            Default is `None`, that is `padding_mode`.
        padding_mode_prolongation: Padding mode for the prolongations. If set,
            overrides `padding_mode` for prolongations.
            Default is `None`, that is `padding_mode`.
        padding_mode_restriction: Padding mode for the restrictions. If set,
            override `padding_mode` for restrictions. Default is `None`,
            that is `padding_mode`.
        prolongation_padding: The padding to use in the transposed convolutions
            of the prolongations. Requires `0` for the `MgNO_helm`
            variant of the original publication. Default is `1`.
        restriction_padding: The padding to use in the convolutions
            of the restrictions. Requires `0` for the `MgNO_helm`
            variant of the original publication. Default is `1`.
        smoothing_padding: The padding to use in the convolutions of the
            pre- and post-smoothing iterations. Is overwritten only if
            `padding_mode="circular"`. See the warnings below.
            No need to change this. Default is `1`.
        use_smoothing_bias: Whether to use bias in the smoothing convolutions. Requires
            `True` for the `MgNO_helm` variant of the original publication.
            Default is `False`.
        use_restriction_pde_bias: Whether to use bias in the approx PDE operator of
            the restrictions. Default is `False`.
        use_residual_restriction: Whether to use residual-based restrictions
            ("Full approximation scheme"-style). Default is `False`.
        use_layernorm: Whether to use layer normalization after each prolongation.
            Requires `False` for the `MgNO_DC` or `MgNO_helm` variant
            of the original publication. Default is `True`.
        use_layernorm_affine: Whether the layer normalization has learnable affine parameters.
            Default is `True`.
        layer_norm_type: The type of layer normalization spatial reduction to use. Requires `"helm"`
            for any `MgNO_helm...` using layer normalization. Default is `"halve"`.
        in_spatial_size: Optional size of the spatial dimensions. Required
            if `use_layernorm` is `True`. MgNO only accepts square
            grids so this should be the same across all spatial dimensions. Used to
            size the layer normalizations if applicable. Default is `None`.
        prolongation_kernel_type: The type of prolongation kernels to use.
            Use `"alternating"` for the `MgNO_DC_smooth` variant of the original publication
            or `"helm"` for the `MgNO_helm...` variants. Default is `"constant"`,
            which uses `kernel_size=4`.
        grow_channels_w_depth: Whether to increase the number of channels
            with depth (`num_channels * (l+1)` at level l). Requires `True`,
            for the `MgNO_helm` variant of the original publication.
            Default is `False`.
        use_xavier_restrict: Whether to use Xavier initialization for the
            restriction operators. Can be `True` for the `MgNO_helm` variant
            of the original publication. Default is `False`.

    Raises:
        ValueError: If an invalid activation function is passed.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **mg_convs** (`tuple[MgConv, ...]`): The `MgConv` blocks layers.
        * **local_layers** (`tuple[PointwiseMLP, ...] | None`): The optional
            local 1x1 Conv layers.
        * **activation_fn** (`Callable[[Array], Array]`): The non-linear
            activation function after each layer.
        * **projection** (`tuple[Callable, ...] | PointwiseMLP`): The output projection.
        * **n_layers** (`int`): The number of layers.
        * **use_residual_skip** (`bool`): Whether to use a ResNet-style residual
            skip connection.

    ??? cite

        [MgNO: Efficient Parameterization of Linear Operators
        via Multigrid](https://proceedings.iclr.cc/paper_files/paper/2024/file/
        eb3c8135137c8a60425a0320869ad87e-Paper-Conference.pdf)

        ```bibtex
        @inproceedings{he2024mgno,
            title={MgNO: Efficient parameterization of linear operators via multigrid},
            author={He, Juncai and Liu, Xinliang and Xu, Jinchao},
            booktitle={International Conference on Learning Representations},
            volume={2024},
            pages={53409--53428},
            year={2024}
        }
        ```

    !!! warning "Restricted Usability"
        `MgNO` is not a general n-dimensional implementation. This mirrors the
        [reference implementation](https://github.com/xlliu2017/MgNO/tree/main).
        In its current architecture, it is only usable with 2D inputs matching
        the experiments of the original publication (e.g. Helmholtz, Darcy or Navier Stokes)
        with their exact grid dimensions.
    """

    mg_convs: tuple[MgConv, ...]
    local_layers: tuple[PointwiseMLP, ...] | None
    activation_fn: Callable[[Array], Array]
    projection: tuple[Callable, ...] | PointwiseMLP
    n_layers: int = eqx.field(static=True)
    use_residual_skip: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        n_layers: int,
        n_its_per_level: Sequence[Sequence[int] | int],
        n_channels_u: int,
        n_channels_f: int,
        proj_hidden_dim: int | None = None,
        use_local_pointwise: bool = False,
        activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu",
        use_residual_skip: bool = True,
        use_proj_mgconv: bool = False,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        padding_mode_smoothing: Literal["zeros", "reflect", "replicate", "circular"]
        | None = None,
        padding_mode_prolongation: Literal["zeros", "reflect", "replicate", "circular"]
        | None = None,
        padding_mode_restriction: Literal["zeros", "reflect", "replicate", "circular"]
        | None = None,
        prolongation_padding: int = 1,
        restriction_padding: int = 1,
        smoothing_padding: int = 1,
        use_smoothing_bias: bool = False,
        use_restriction_pde_bias: bool = False,
        use_residual_restriction: bool = False,
        use_layernorm: bool = True,
        use_layernorm_affine: bool = True,
        layer_norm_type: Literal["helm", "halve"] = "halve",
        in_spatial_size: int | None = None,
        prolongation_kernel_type: Literal[
            "constant", "alternating", "helm"
        ] = "constant",
        grow_channels_w_depth: bool = False,
        use_xavier_restrict: bool = False,
    ) -> None:
        self.n_layers = n_layers
        self.use_residual_skip = use_residual_skip
        valid_activation_fns = ["relu", "gelu", "tanh", "silu"]
        if activation not in valid_activation_fns:
            raise ValueError(
                f"Invalid activation funtion. Has to be one of {valid_activation_fns}."
            )
        activation_fns = {a: getattr(jax.nn, a) for a in valid_activation_fns}
        self.activation_fn = activation_fns[activation]
        key, *layer_keys = jr.split(key, n_layers + 1)
        self.mg_convs = tuple(
            MgConv(
                lkey,
                n_its_per_level,
                n_channels_u,
                n_channels_f if l == 0 else n_channels_u,
                padding_mode,
                padding_mode_smoothing,
                padding_mode_prolongation,
                padding_mode_restriction,
                prolongation_padding,
                restriction_padding,
                smoothing_padding,
                use_smoothing_bias,
                use_restriction_pde_bias,
                use_residual_restriction,
                use_layernorm,
                use_layernorm_affine,
                layer_norm_type,
                in_spatial_size,
                prolongation_kernel_type,
                grow_channels_w_depth,
                use_xavier_restrict,
            )
            for l, lkey in zip(range(n_layers), layer_keys, strict=True)
        )
        if use_local_pointwise:
            key, *local_keys = jr.split(key, self.n_layers + 1)
            self.local_layers = tuple(
                PointwiseMLP(
                    lkey, (n_channels_f if l == 0 else n_channels_u, n_channels_u)
                )
                for l, lkey in zip(range(self.n_layers), local_keys, strict=True)
            )
        else:
            self.local_layers = None

        if not use_proj_mgconv:
            if proj_hidden_dim is not None:
                key, *proj_keys = jr.split(key, 3)
                self.projection = (
                    PointwiseMLP(
                        proj_keys[0], (n_channels_u, proj_hidden_dim), use_bias=False
                    ),
                    jax.nn.gelu,
                    PointwiseMLP(proj_keys[1], (proj_hidden_dim, 1), use_bias=False),
                )
            else:
                proj_key, key = jr.split(key)
                self.projection = PointwiseMLP(
                    proj_key, (n_channels_u, 1), use_bias=False
                )
        else:
            self.projection = MgConv(
                key,
                n_its_per_level,
                1,
                n_channels_u,
                padding_mode,
                padding_mode_smoothing,
                padding_mode_prolongation,
                padding_mode_restriction,
                prolongation_padding,
                restriction_padding,
                smoothing_padding,
                use_smoothing_bias,
                use_restriction_pde_bias,
                use_residual_restriction,
                use_layernorm,
                use_layernorm_affine,
                layer_norm_type,
                in_spatial_size,
                prolongation_kernel_type,
                grow_channels_w_depth,
                use_xavier_restrict,
            )

    @classmethod
    def make_mgno_type(
        self,
        key: PRNGKeyArray,
        mgno_type: Literal[
            "MgNO_NS", "MgNO_DC", "MgNO_DC_smooth", "MgNO_helm", "MgNO_helm2"
        ],
        n_layers: int,
        activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu",
        n_its_per_level: Sequence[Sequence[int] | int] | None = None,
    ) -> "MgNO":
        """Configures an MgNO instance for an experiment specific architecture.

        The non-specific parameters still have to passed (`n_layers` and `n_its_per_level`).
        The iterations per level may be omitted.

        Args:
            mgno_type: The type of MgNO architecture to use.
            n_layers: The number of `MgConv` blocks (and optionally 1x1 Convs) to use.
                Varies with model variant (some add an additional output or input `MgConv` block).
            activation: The type of non-linear activation function to use. Default is `"gelu"`.
            n_its_per_level: Number of iterations per level. A sequence containing
                a pair `[n_pre, n_post]` (pre-smoothing, post-smoothing) per level
                in the V-cycle or a sequence of ints specifying only pre-smoothing iterations
                per level (as is the case in the `MgNO_helm` variant of the original publication).
                Default is `None`, in which case the pre-configured value is used.

        Returns:
            A configured MgNO instance.

        Raises:
            ValueError: If an invalid `MgNO` type is passed.
        """
        match mgno_type:
            case "MgNO_NS":
                params = dict(
                    n_layers=n_layers,
                    activation=activation,
                    n_its_per_level=[[1, 0], [1, 0], [1, 0], [2, 0], [2, 0]]
                    if n_its_per_level is None
                    else n_its_per_level,
                    n_channels_u=32,
                    n_channels_f=1,
                    padding_mode="circular",
                    padding_mode_prolongation="zeros",
                    use_layernorm=True,
                    use_layernorm_affine=True,
                    in_spatial_size=64,
                    use_smoothing_bias=True,
                )
            case "MgNO_DC":
                params = dict(
                    n_layers=n_layers,
                    activation=activation,
                    n_its_per_level=[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [2, 0]]
                    if n_its_per_level is None
                    else n_its_per_level,
                    n_channels_u=32,
                    n_channels_f=1,
                    use_layernorm=False,
                    use_residual_restriction=True,
                    use_residual_skip=False,
                    use_local_pointwise=True,
                )
            case "MgNO_DC_smooth":
                params = dict(
                    n_layers=n_layers,
                    activation=activation,
                    n_its_per_level=[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [2, 0]]
                    if n_its_per_level is None
                    else n_its_per_level,
                    n_channels_u=24,
                    n_channels_f=1,
                    prolongation_kernel_type="alternating",
                    use_layernorm=False,
                    use_residual_restriction=True,
                    use_residual_skip=False,
                    use_local_pointwise=True,
                )
            case "MgNO_helm":
                params = dict(
                    n_layers=n_layers,
                    activation=activation,
                    n_channels_u=20,
                    n_channels_f=1,
                    n_its_per_level=[1, 1, 1, 1, 2]
                    if n_its_per_level is None
                    else n_its_per_level,
                    prolongation_padding=0,
                    restriction_padding=0,
                    padding_mode_smoothing="reflect",
                    prolongation_kernel_type="helm",
                    use_xavier_restrict=True,
                    grow_channels_w_depth=True,
                    use_smoothing_bias=True,
                    use_layernorm=False,
                    use_residual_skip=False,
                    use_proj_mgconv=True,
                )
            case "MgNO_helm2":
                params = dict(
                    n_layers=n_layers,
                    activation=activation,
                    n_channels_u=20,
                    n_channels_f=1,
                    n_its_per_level=[1, 1, 1, 1, 2]
                    if n_its_per_level is None
                    else n_its_per_level,
                    prolongation_padding=0,
                    restriction_padding=0,
                    padding_mode_smoothing="reflect",
                    prolongation_kernel_type="helm",
                    layer_norm_type="helm",
                    use_xavier_restrict=False,
                    grow_channels_w_depth=False,
                    use_smoothing_bias=True,
                    use_layernorm=True,
                    use_layernorm_affine=False,
                    in_spatial_size=101,
                    use_residual_skip=False,
                    use_proj_mgconv=False,
                )
            case _:
                raise ValueError(
                    "Passed invalid MgNO type. Must be one of "
                    "['MgNO_NS', 'MgNO_DC', 'MgNO_DC_smooth', 'MgNO_helm', 'MgNO_helm2']."
                )
        return MgNO(key=key, **params)

    @override
    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the MgNO.

        Args:
            x: Input field or forcing of shape `(in_c, ...)`.
            key: Ignored. Kept for API compatibility.
            inference: Ignored. Kept for API compatibility.

        Returns:
            The predicted field of shape `(out_c, ...)`.
        """
        x_res = x
        for i in range(self.n_layers):
            if self.local_layers is None:
                x = self.activation_fn(self.mg_convs[i](x))
            else:
                x = self.activation_fn(self.mg_convs[i](x) + self.local_layers[i](x))
        if isinstance(self.projection, tuple):
            for layer in self.projection:
                x = layer(x)
        else:
            x = self.projection(x)
        if self.use_residual_skip:
            return x + x_res
        return x
