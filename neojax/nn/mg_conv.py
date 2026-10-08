"""Implementation of Multigrid convolution as used in MgNO."""

from collections.abc import Sequence
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.nn.mg_iter import MgIter
from neojax.nn.mg_restrict import MgRestrict


class MgConv(eqx.Module):
    """Multigrid V-cycle convolutional block.

    Builds a hierarchy of `len(n_its_per_level)` levels. Each level
    applies pre-smoothing iterations `MgIter`, restricts to the
    next coarser grid, and then, after a prolongated coarse correction,
    applies post-smoothing on the upward pass.

    Args:
        key: PRNG key for parameter initialization.
        n_its_per_level: Number of iterations per level. A sequence containing
            a pair `[n_pre, n_post]` (pre-smoothing, post-smoothing) per level
            in the V-cycle or a sequence of ints specifying only pre-smoothing iterations
            per level (as is the case in the `MgConv_helm` variant of the original publication).
        n_channels_u: Number of channels of the solution field.
        n_channels_f: Number of channels of the data field.
        padding_mode: Padding mode for the convolutions. Requires `"reflect"`
            for the `MgConv_helm` variant of the original publication.
            Default is `"zeros"`.
        padding_mode_smoothing: Padding mode for the smoothing iterations. If set,
            overrides `padding_mode` for smoothing. Requires `"reflect"` for the
            `MgConv_helm3` variant of the original publication.
            Default is `None`, that is `padding_mode`.
        padding_mode_prolongation: Padding mode for the prolongations. If set,
            overrides `padding_mode` for prolongations.
            Default is `None`, that is `padding_mode`.
        padding_mode_restriction: Padding mode for the restrictions. If set,
            override `padding_mode` for restrictions. Default is `None`,
            that is `padding_mode`.
        prolongation_padding: The padding to use in the transposed convolutions
            of the prolongations. Requires `0` for the `MgConv_helm`
            variant of the original publication. Default is `1`.
        restriction_padding: The padding to use in the convolutions
            of the restrictions. Requires `0` for the `MgConv_helm`
            variant of the original publication. Default is `1`.
        smoothing_padding: The padding to use in the convolutions of the
            pre- and post-smoothing iterations. Is overwritten only if
            `padding_mode="circular"`. See the warnings below.
            No need to change this. Default is `1`.
        use_smoothing_bias: Whether to use bias in the smoothing convolutions. Requires
            `True` for the `MgConv_helm` variant of the original publication.
            Default is `False`.
        use_restriction_pde_bias: Whether to use bias in the approx PDE operator of
            the restrictions. Default is `False`.
        use_residual_restriction: Whether to use residual-based restrictions
            ("Full approximation scheme"-style). Default is `False`.
        use_layernorm: Whether to use layer normalization after each prolongation.
            Requires `False` for the `MgConv_DC` or `MgConv_helm` variant
            of the original publication. Default is `True`.
        use_layernorm_affine: Whether the layer normalization has learnable affine parameters.
            Default is `True`.
        layer_norm_type: The type of layer normalization spatial reduction to use. Requires `"helm"`
            for any `MgConv_helm...` using layer normalization. Default is `"halve"`.
        in_spatial_size: Optional size of the spatial dimensions. Required
            if `use_layernorm` is `True`. MgNO only accepts square
            grids so this should be the same across all spatial dimensions. Used to
            size the layer normalizations if applicable. Default is `None`.
        prolongation_kernel_type: The type of prolongation kernels to use.
            Use `"alternating"` for the `MgConv_DC_smooth` variant of the original publication
            or `"helm"` for the `MgConv_helm...` variants. Default is `"constant"`,
            which uses `kernel_size=4`.
        grow_channels_w_depth: Whether to increase the number of channels
            with depth (`num_channels * (l+1)` at level l). Requires `True`,
            for the `MgConv_helm` variant of the original publication.
            Default is `False`.
        use_xavier_restrict: Whether to use Xavier initialization for the
            restriction operators. Can be `True` for the `MgConv_helm` variant
            of the original publication. Default is `False`.

    Raises:
        ValueError: If `n_its_per_level` has zero length or per level sequence
            doesn't have 2 entries if `n_its_per_level` is sequence of sequences.
        ValueError: If `use_layernorm` is `True` but `in_spatial_size` is `None`.
        TypeError: If `n_its_per_level` is not a sequence of ints or sequence
            of sequence of ints.

    ??? info "Internal Attributes"
        These fields store the internal attributes and weights.

        * **layer_norms** (`tuple[eqx.Module, ...] | None`): The layer normalizations per level.
        * **prolongation_convs** (`tuple[eqx.Module, ...]`): The prolongation convolutions
            per level.
        * **layers** (`tuple[tuple[tuple[MgIter, ...], MgRestrict, tuple[MgIter, ...]], ...]`): Three element
            tuples per level containing the pre-smoothing, restriction and post-smoothing iterations.
        * **n_its_per_level** (`tuple[tuple[int, int] | int, ...]`): The number of iterations per level.
        * **n_levels** (`int`): The number of multigrid levels. Corresponds to `len(n_its_per_level)`.
        * **padding_mode_smoothing** (`str`): The padding mode for the smoothing convolutions.
        * **padding_mode_prolongation** (`str`): The padding mode for the prolongation convolutions.
        * **padding_mode_restriction** (`str`): The padding mode for the restriction convolutions.

    !!! warning "Using `padding_mode="circular"`"
        In order to be compatible with the underlying `equinox.nn.Conv`
        class, all `padding` parameters in `Conv2d` instances are set
        to `padding="SAME"` if `padding_mode="circular"` (in `MgNO_NS` variant).
        The `padding` of the transposed prolongation convolutions are kept as is.

    !!! warning "Restricted Usability"
        `MgConv` is not a general n-dimensional implementation. This mirrors the
        [reference implementation](https://github.com/xlliu2017/MgNO/tree/main).
        In its current architecture it is only usable with 2D inputs matching
        the experiments of the original publication (e.g. Helmholtz, Darcy or Navier Stokes)
        with their exact grid dimensions.

    !!! warning "Configuration of this Class"
        The configuration of the task specific architectures of the original publication
        is all done within this class. Have a look at the parameters to configure
        this class as needed. There are no exhaustive safety checks to validate all
        correct parameter combinations for each task architecture (e.g. `MgConv_helm`, ...).
        A model with an invalid configuration will just fail. That is the downside of
        a single modular class.
    """

    layer_norms: tuple[eqx.Module, ...] | None
    prolongation_convs: tuple[eqx.Module, ...]
    layers: tuple[tuple[tuple[MgIter, ...], MgRestrict, tuple[MgIter, ...]], ...]
    n_its_per_level: tuple[tuple[int, int] | int, ...] = eqx.field(static=True)
    n_levels: int = eqx.field(static=True)
    padding_mode_smoothing: str = eqx.field(static=True)
    padding_mode_prolongation: str = eqx.field(static=True)
    padding_mode_restriction: str = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        n_its_per_level: Sequence[Sequence[int] | int],
        n_channels_u: int,
        n_channels_f: int,
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
        if len(n_its_per_level) == 0:
            raise ValueError(
                "Passed length zero `n_its_per_level` but number of levels cannot be zero."
            )
        if all(isinstance(seq, Sequence) for seq in n_its_per_level):
            if any(len(seq) != 2 for seq in n_its_per_level):
                raise ValueError(
                    "Iterations per level has to be a sequence with 2 elements [n_pre, n_post] per level."
                )
            self.n_its_per_level = tuple(tuple(seq) for seq in n_its_per_level)
        elif all(isinstance(it, int) for it in n_its_per_level):
            # if iterations per level are ints, then there is no post-smoothing (MgConv_helm variant)
            self.n_its_per_level = tuple((seq, 0) for seq in n_its_per_level)
        else:
            raise TypeError(
                "`n_its_per_level` must be sequence of sequence of ints or sequence of ints."
            )

        self.n_levels = len(n_its_per_level)

        if padding_mode_smoothing is None:
            self.padding_mode_smoothing = padding_mode
        else:
            self.padding_mode_smoothing = padding_mode_smoothing
        if padding_mode_prolongation is None:
            self.padding_mode_prolongation = padding_mode
        else:
            self.padding_mode_prolongation = padding_mode_prolongation
        if padding_mode_restriction is None:
            self.padding_mode_restriction = padding_mode
        else:
            self.padding_mode_restriction = padding_mode_restriction
        # Override smoothing and restriction padding if circular, see warning above
        if padding_mode == "circular":
            smoothing_padding = "SAME"
            restriction_padding = "SAME"
        # Define LayerNorms
        if use_layernorm:
            if in_spatial_size is None:
                raise ValueError(
                    "Must pass `in_spatial_size` to size the layer normalization."
                )
            layer_norms = []
            for i in range(self.n_levels - 1):
                if layer_norm_type == "helm":
                    size_fac_fn = lambda s, i=i: (s + 1) // 2**i - 1
                else:
                    size_fac_fn = lambda s, i=i: s // 2**i

                channel_fac = i + 1 if grow_channels_w_depth else 1

                layer_norms.append(
                    eqx.nn.LayerNorm(
                        (
                            n_channels_u * channel_fac,
                            size_fac_fn(in_spatial_size, i),
                            size_fac_fn(in_spatial_size, i),
                        ),
                        use_weight=use_layernorm_affine,
                        use_bias=use_layernorm_affine,
                    )
                )
            self.layer_norms = tuple(layer_norms)
        else:
            self.layer_norms = None
        # Define prolongation convolutions (coarse to fine)
        key, pr_key = jr.split(key)
        pr_keys = jr.split(pr_key, self.n_levels - 1)
        prolongation_convs = []
        for subkey, k in zip(pr_keys, range(self.n_levels - 1), strict=True):
            in_channels_fac = k + 2 if grow_channels_w_depth else 1
            out_channels_fac = k + 1 if grow_channels_w_depth else 1
            if prolongation_kernel_type == "alternating":
                kernel_size = 3 if k in (0, 2, 3) else 4
            else:
                if prolongation_kernel_type == "helm" and k in (0, self.n_levels - 2):
                    kernel_size = 3
                else:
                    # Not helm version or correct layer or in correct alternating prolongation level
                    kernel_size = 4
            prolongation_convs.append(
                eqx.nn.ConvTranspose2d(
                    key=subkey,
                    in_channels=n_channels_u * in_channels_fac,
                    out_channels=n_channels_u * out_channels_fac,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=prolongation_padding,
                    use_bias=False,
                    padding_mode=self.padding_mode_prolongation,
                )
            )
        self.prolongation_convs = tuple(prolongation_convs)

        # Define per-level pre-smooth and post-smooth sequential modules
        layers = []
        level_keys = jr.split(key, self.n_levels)
        for l, (its_level, level_key) in enumerate(
            zip(self.n_its_per_level, level_keys, strict=True)
        ):
            post_smoothing_layers = []
            pre_smoothing_layers = []

            in_channels_fac = l + 1 if grow_channels_w_depth else 1
            out_channels_fac = l + 1 if grow_channels_w_depth else 1

            # Pre-smoothing its
            for i in range(its_level[0]):
                level_ps_key, level_key = jr.split(level_key)
                smoothing_op = eqx.nn.Conv2d(
                    key=level_ps_key,
                    in_channels=n_channels_f * in_channels_fac,
                    out_channels=n_channels_u * out_channels_fac,
                    kernel_size=3,
                    stride=1,
                    padding=smoothing_padding,
                    use_bias=use_smoothing_bias,
                    padding_mode=self.padding_mode_smoothing,
                )
                # Always use pde_op in case of grow_channels_w_depth
                if not (l == 0 and i == 0) or grow_channels_w_depth:
                    level_pde_key, level_key = jr.split(level_key)
                    pde_op = eqx.nn.Conv2d(
                        key=level_pde_key,
                        in_channels=n_channels_u * in_channels_fac,
                        out_channels=n_channels_f * out_channels_fac,
                        kernel_size=3,
                        stride=1,
                        padding=smoothing_padding,
                        use_bias=use_smoothing_bias,
                        padding_mode=self.padding_mode_smoothing,
                    )
                else:
                    # Initial step
                    pde_op = None
                pre_smoothing_layers.append(MgIter(smoothing_op, pde_op))

            # Post-smoothing its (after prolongation)
            # Don't need growing channels here (_helm has no post-smoothing)
            if its_level[1] != 0:
                for _ in range(its_level[1]):
                    level_ps_key, level_pde_key, level_key = jr.split(level_key, 3)
                    smoothing_op = eqx.nn.Conv2d(
                        key=level_ps_key,
                        in_channels=n_channels_f,
                        out_channels=n_channels_u,
                        kernel_size=3,
                        stride=1,
                        padding=smoothing_padding,
                        use_bias=use_smoothing_bias,
                        padding_mode=self.padding_mode_smoothing,
                    )
                    pde_op = eqx.nn.Conv2d(
                        key=level_pde_key,
                        in_channels=n_channels_u,
                        out_channels=n_channels_f,
                        kernel_size=3,
                        stride=1,
                        padding=smoothing_padding,
                        use_bias=use_smoothing_bias,
                        padding_mode=self.padding_mode_smoothing,
                    )
                    post_smoothing_layers.append(MgIter(smoothing_op, pde_op))
            else:
                post_smoothing_layers.append(eqx.nn.Identity())

            pre_smoothing_layers = tuple(pre_smoothing_layers)
            post_smoothing_layers = tuple(post_smoothing_layers)

            if use_xavier_restrict:
                xavier_init = jax.nn.initializers.xavier_uniform()
            # Restriction operators
            if 0 < l < self.n_levels:
                pde_key, sr_key, dr_key, level_key = jr.split(level_key, 4)

                in_channels_fac = l if grow_channels_w_depth else 1
                out_channels_fac = l + 1 if grow_channels_w_depth else 1

                pde_op = eqx.nn.Conv2d(
                    key=pde_key,
                    in_channels=n_channels_u * in_channels_fac,
                    out_channels=n_channels_f * out_channels_fac,
                    kernel_size=3,
                    stride=1,
                    padding=restriction_padding,
                    use_bias=use_restriction_pde_bias,
                    padding_mode=self.padding_mode_restriction,
                )
                sol_restrict_op = eqx.nn.Conv2d(
                    key=sr_key,
                    in_channels=n_channels_u * in_channels_fac,
                    out_channels=n_channels_u * out_channels_fac,
                    kernel_size=3,
                    stride=2,
                    padding=restriction_padding,
                    use_bias=False,
                    padding_mode=self.padding_mode_restriction,
                )
                data_restrict_op = eqx.nn.Conv2d(
                    key=dr_key,
                    in_channels=n_channels_f * in_channels_fac,
                    out_channels=n_channels_f * out_channels_fac,
                    kernel_size=3,
                    stride=2,
                    padding=restriction_padding,
                    use_bias=False,
                    padding_mode=self.padding_mode_restriction,
                )
                if use_xavier_restrict:
                    skey, wkey, level_key = jr.split(level_key, 3)
                    init_replace_fn = partial(
                        lambda w, k, init_fn=xavier_init: (
                            init_fn(k, w.shape, dtype=w.dtype) * (1 / n_channels_u**2)
                        ),
                        k=skey,
                    )
                    sol_restrict_op = eqx.tree_at(
                        lambda t: t.weight, sol_restrict_op, replace_fn=init_replace_fn
                    )
                    init_replace_fn = partial(
                        lambda w, k, init_fn=xavier_init: (
                            init_fn(k, w.shape, dtype=w.dtype) * (1 / n_channels_u**2)
                        ),
                        k=wkey,
                    )
                    data_restrict_op = eqx.tree_at(
                        lambda t: t.weight, data_restrict_op, replace_fn=init_replace_fn
                    )

                if use_residual_restriction:
                    restriction = MgRestrict(sol_restrict_op, data_restrict_op, pde_op)
                else:
                    restriction = MgRestrict(sol_restrict_op, data_restrict_op)
            else:
                # First layer only uses pre-smoothing
                restriction = None

            layers.append((restriction, pre_smoothing_layers, post_smoothing_layers))
        self.layers = tuple(layers)

    def __call__(self, f: Inexact[Array, "..."]) -> Inexact[Array, "..."]:
        """Runs multigrid V-cycle convolution.

        First computes the restriction steps, that is pre-smoothing
        followed by restricting at each level.
        Then computes the prolongated coarse corrections, (applies layer norm)
        and post-smoothing.

        Args:
            f: The input data field.

        Returns:
            The approximated solution field u.
        """
        out_list = [0] * self.n_levels
        out = (f,)

        # restriction steps (pre-smooth + restrict)
        for l in range(self.n_levels):
            restrict, pre_smooth = self.layers[l][:2]
            if restrict is not None:
                out = restrict(*out)
            for ps in pre_smooth:
                out = ps(*out)

            out_list[l] = out

        # prolongation steps
        for i in range(self.n_levels - 2, -1, -1):
            f, u = out_list[i]
            u_pro = self.prolongation_convs[i](out_list[i + 1][1])
            if self.layer_norms is not None:
                u_pro = self.layer_norms[i](u + u_pro)

            # apply post-smoothing
            post_smooth = self.layers[i][2]
            for ps in post_smooth:
                if isinstance(ps, eqx.nn.Identity):
                    f, u_pro = ps((f, u_pro))
                else:
                    f, u_pro = ps(f=f, u=u_pro)
            out_list[i] = (f, u_pro)

        return out_list[0][1]
