"""Implementation of a n-dimensional Geometry-Informed Neural Operator (GiNO)."""

from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.fno_blocks import FNOBlocks
from neojax.nn.gno_block import GNOBlock
from neojax.nn.mollifier_functions import make_weighting_fn
from neojax.nn.neighbor_search import Neighbors
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tensor import BaseTensor


@final
class GiNO(BaseNO):
    r"""General n-dimensional Geometry-Informed Neural Operator (GiNO).

    GiNO learns a map between functions given on arbitrary point clouds or
    meshes. It combines the flexibility of the Graph Neural Operator (GNO)
    with the efficiency of the FNO. The forward pass is

    1. an input `GNOBlock` that integrates the input function over the
       neighborhoods of the points of a regular latent grid,
    2. optional concatenation of additional latent grid features,
    3. a lifting `PointwiseMLP` and the `FNOBlocks` on the latent grid,
    4. an output `GNOBlock` that integrates the latent function over the
       neighborhoods of the output query points,
    5. a final projection `PointwiseMLP` to the output channels.

    The output GNO can be turned into a mollified GNO by
    selecting a weighting function (`gno_weighting_function`). The output
    is then differentiable in the query positions `x_out`.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of channels of the input function.
        out_channels: Number of channels of the output function.
        modes: Number of Fourier modes to retain across each dimension of
            the latent grid, e.g. (16, 16, 16). Its length is the dimension
            of the latent grid. Must be smaller than half of the latent grid
            resolution (Nyquist frequency).
        hidden_channels: Hidden channel dimension of the latent FNO.
        n_layers: Number of consecutive FNO blocks.
        coord_dim: Dimension of the physical coordinates of the input points
            and output queries. Default is 3.
        latent_feature_channels: Number of channels of an optional latent
            feature map, concatenated to the output of the input GNO before
            the lifting. Default is `None`, no latent features.
        in_gno_radius: Radius of the neighborhoods of the input GNO.
            Larger radii mean more interaction at a higher cost.
            Default is 0.033.
        out_gno_radius: Radius of the neighborhoods of the output GNO.
            Default is 0.033.
        in_gno_max_neighbors: Maximum number of neighbors per latent point
            in the input GNO. If more points are inside the radius, only the
            nearest are used. Default is 32.
        out_gno_max_neighbors: Maximum number of neighbors per output query
            in the output GNO. Default is 32.
        in_gno_transform_type: Integral transform of the input GNO,
            see `IntegralTransform`. Default is `"linear"`.
        out_gno_transform_type: Integral transform of the output GNO.
            Default is `"linear"`.
        in_gno_out_channels: Number of output channels of the input GNO,
            i.e. input channels of the latent FNO (without latent features).
            Must equal `in_channels` for the `"linear"` and `"nonlinear"`
            transforms. Default is `None`, which is `in_channels`.
        gno_weighting_function: Weighting function of the output GNO
            for a mollified GNO, see `make_weighting_fn`.
            Default is `None`.
        gno_weight_function_scale: Factor by which to scale the weights of
            `gno_weighting_function`. Default is 1.0.
        in_gno_pos_embedding_type: Type of the sinusoidal embedding of the
            points in the input GNO, `"transformer"`, `"nerf"` or `None`
            for no embedding. Default is `"transformer"`.
        out_gno_pos_embedding_type: Type of the sinusoidal embedding of the
            points in the output GNO. Default is `"transformer"`.
        gno_embed_channels: Number of frequencies of the sinusoidal
            embeddings per coordinate. Default is 32.
        gno_embed_max_positions: `max_positions` of the `"transformer"`
            embeddings. Default is 10000.
        in_gno_channel_mlp_hidden_layers: Widths of the hidden layers of the
            kernel network of the input GNO. Default is `(80, 80, 80)`.
        out_gno_channel_mlp_hidden_layers: Widths of the hidden layers of the
            kernel network of the output GNO. Default is `(512, 256)`.
        gno_activation: Activation function of the GNO kernel networks.
            Default is `jax.nn.gelu`.
        out_gno_tanh: Whether to apply `tanh` to the latent function before
            the output GNO, to stabilize it. Default is `False`.
        gno_query_chunk_size: Number of query points the GNO neighbor
            searches and integral transforms process at a time to limit
            peak memory. Default is `None`, all points at once.
        activation: Activation function of the lifting, FNO blocks
            and projection. Default is `jax.nn.gelu`.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            after each FNO block. Default is `True`.
        local_operator: Type of local operator inside FNO blocks.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"linear"`.
        use_local_operator_bias: Whether to use a bias term
            in the FNO blocks local operator. Default is `False`.
        channel_mlp_residual: Type of skip connection around channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Default is `"soft-gating"`.
        channel_mlp_expansion: Expansion factor for hidden dimension
            in the channel MLPs. Default is 0.5.
        channel_mlp_activations: Activation function or sequence
            of activation functions used inside the channel MLPs.
            Default is `jax.nn.gelu`.
        channel_mlp_dropout: Dropout probability applied after each layer
            (except the last) of the channel MLPs. Default is 0.0.
        normalization: Type of normalization to use in FNO blocks.
            Can be `"layer"`, `"instance"`, `"group"` or None.
            Default is None.
        norm_groups: Number of groups for group normalization.
            Default is 1.
        use_fno_residual: Whether to use a Resnet-style residual connection
            around FNO blocks. Default is `False`.
        preactivation: Whether to use pre-activation style blocks.
            Default is `False`.
        n_lift_layers: Number of layers in the lifting MLP. Default is 2.
        n_proj_layers: Number of layers in the projection MLP. Default is 2.
        lift_channel_ratio: Ratio of lifting channels to `hidden_channels`.
            Default is 2.0.
        proj_channel_ratio: Ratio of projection channels to `hidden_channels`.
            Default is 4.0.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs of the spectral convolutions
            before calling `irfftn`. Default is `True`.
        fft_norm: FFT normalization. Can be `None`, `"backward"`,
            `"ortho"` or `"forward"`. Default is `"forward"`.
        is_complex_data: Whether the latent data is complex valued.
            If True, uses full FFT. Default is `False`.
        ranks: Number of ranks to contract the spectral tensors to.
            Default is `None`.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels * out_channels)) ** 0.5.
        factorization: Tensor factorization type of the spectral weights.
            Can be a `BaseTensor` instance, a string ("tucker", "cp", "tt"),
            or None. Default is `None`.
        implementation: Weight reconstruction mode.
            Can be "reconstructed" or "factorized".
            Default is "factorized".
        separable: Whether to use separable implementation of contraction.
            Default is `False`.

    Raises:
        ValueError: If `modes` is empty or contains non-positive entries.
        ValueError: If `gno_weighting_function` is invalid.
        ValueError: If the channels of the input GNO don't match
            its transform type, see `GNOBlock`.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **in_gno** (`GNOBlock`): GNO from the input points to the latent grid.
        * **lifting** (`PointwiseMLP`): The `PointwiseMLP` lifting the latent
            function to `hidden_channels`.
        * **fno_blocks** (`FNOBlocks`): The `FNOBlocks` on the latent grid.
        * **out_gno** (`GNOBlock`): GNO from the latent grid to the output queries.
        * **projection** (`PointwiseMLP`): The `PointwiseMLP` projecting to `out_channels`.
        * **latent_feature_channels** (`int`): Number of latent feature channels.
        * **latent_dim** (`int`): Dimension of the latent grid.
        * **out_gno_tanh** (`bool`): Whether `tanh` is applied before the output GNO.

    ??? cite

        [Geometry-Informed Neural Operator for Large-Scale 3D PDEs](
        https://arxiv.org/abs/2309.00583)

        ```bibtex
        @inproceedings{li2023geometry,
            title={Geometry-Informed Neural Operator for Large-Scale 3D PDEs},
            author={Li, Zongyi and Kovachki, Nikola and Choy, Chris and Li, Boyi
                and Kossaifi, Jean and Otta, Shourya and Nabian, Mohammad Amin
                and Stadler, Maximilian and Hundt, Christian and
                Azizzadenesheli, Kamyar and Anandkumar, Anima},
            booktitle={Advances in Neural Information Processing Systems},
            year={2023}
        }
        ```

        [Neural Operator: Graph Kernel Network for Partial Differential Equations](
        https://arxiv.org/abs/2003.03485)

        ```bibtex
        @article{li2020neural,
            title={Neural operator: Graph kernel network for
                partial differential equations},
            author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli,
                Kamyar and Liu, Burigede and Bhattacharya, Kaushik
                and Stuart, Andrew and Anandkumar, Anima},
            journal={arXiv preprint arXiv:2003.03485},
            year={2020}
        }
        ```

        [Enabling Automatic Differentiation
        with Mollified Graph Neural Operators](https://arxiv.org/pdf/2504.08277)

        ```bibtex
        @article{lin2025enabling,
            title={Enabling automatic differentiation with mollified graph neural operators},
            author={Lin, Ryan Y and Berner, Julius and Duruisseaux,
            Valentin and Pitt, David and Leibovici, Daniel and Kossaifi,
            Jean and Azizzadenesheli, Kamyar and Anandkumar, Anima},
            journal={arXiv preprint arXiv:2504.08277},
            year={2025}
        }
        ```

    !!! info "`vmap` and fixed geometries"
        To batch the `vmap` call, map over the function values `u` (and latent_features)
        and share the geometry with `in_axes=None`

        ```python
        out = jax.vmap(
            model, in_axes=(0, None, None, None)
        )(u_batch, x_in, x_latent, x_out)
        ```

        The geometries can also be batched, since the neighbor search
        is jit- and vmap-compatible. For a fixed geometry, precompute the neighbors
        once (`radius_neighbor_search`) and pass them with `in_neighbors`
        and `out_neighbors` to avoid searching in every call.

    !!! warning "Neighbor truncation"
        The GNOs consider at most `in_gno_max_neighbors` and
        `out_gno_max_neighbors` neighbors per point (the nearest ones).
        Choose these large enough for your point densities and radii.
        With a `gno_weighting_function`, truncated neighbors can have nonzero
        weights, which breaks the smoothness of the output. Then choose the
        values large enough that truncation does not occur.

    !!! info "Extent of the latent grid"
        The mollified GNO paper pads the latent domain, so that the latent function
        is supported on the geometry extended by the radius,
        $D + B_r(0)$. Let `x_latent` extend about `in_gno_radius` and
        `out_gno_radius` beyond the geometry, such that the GNOs see a
        full neighborhood at the boundary of the domain.
    """

    in_gno: GNOBlock
    lifting: PointwiseMLP
    fno_blocks: FNOBlocks
    out_gno: GNOBlock
    projection: PointwiseMLP
    latent_feature_channels: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    out_gno_tanh: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        hidden_channels: int = 64,
        n_layers: int = 4,
        coord_dim: int = 3,
        latent_feature_channels: int | None = None,
        in_gno_radius: float = 0.033,
        out_gno_radius: float = 0.033,
        in_gno_max_neighbors: int = 32,
        out_gno_max_neighbors: int = 32,
        in_gno_transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        out_gno_transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        in_gno_out_channels: int | None = None,
        gno_weighting_function: Literal["bump", "half_cos", "quadr", "quartic", "octic"]
        | None = None,
        gno_weight_function_scale: float = 1.0,
        in_gno_pos_embedding_type: Literal["transformer", "nerf"]
        | None = "transformer",
        out_gno_pos_embedding_type: Literal["transformer", "nerf"]
        | None = "transformer",
        gno_embed_channels: int = 32,
        gno_embed_max_positions: int = 10000,
        in_gno_channel_mlp_hidden_layers: Sequence[int] = (80, 80, 80),
        out_gno_channel_mlp_hidden_layers: Sequence[int] = (512, 256),
        gno_activation: Callable = jax.nn.gelu,
        out_gno_tanh: bool = False,
        gno_query_chunk_size: int | None = None,
        activation: Callable = jax.nn.gelu,
        use_channel_mlp: bool = True,
        local_operator: Literal["linear", "soft-gating", "identity"] | None = "linear",
        use_local_operator_bias: bool = False,
        channel_mlp_residual: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        channel_mlp_expansion: float | None = 0.5,
        channel_mlp_activations: Callable | Sequence[Callable] = jax.nn.gelu,
        channel_mlp_dropout: float = 0.0,
        normalization: Literal["layer", "instance", "group"] | None = None,
        norm_groups: int = 1,
        use_fno_residual: bool = False,
        preactivation: bool = False,
        n_lift_layers: int = 2,
        n_proj_layers: int = 2,
        lift_channel_ratio: float = 2.0,
        proj_channel_ratio: float = 4.0,
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
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
        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if not isinstance(n_layers, int) or n_layers <= 0:
            raise ValueError("n_layers must be a positive integer.")
        if not isinstance(n_lift_layers, int) or n_lift_layers <= 0:
            raise ValueError("n_lift_layers must be a positive integer.")
        if not isinstance(n_proj_layers, int) or n_proj_layers <= 0:
            raise ValueError("n_proj_layers must be a positive integer.")
        if latent_feature_channels is not None and (
            not isinstance(latent_feature_channels, int) or latent_feature_channels <= 0
        ):
            raise ValueError("latent_feature_channels must be a positive integer.")

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

        self.latent_dim = len(modes_tuple)
        self.latent_feature_channels = latent_feature_channels or 0
        self.out_gno_tanh = out_gno_tanh

        k_in, k_lift, k_fno, k_out, k_proj = jr.split(key, 5)

        if in_gno_out_channels is None:
            in_gno_out_channels = in_channels
        self.in_gno = GNOBlock(
            key=k_in,
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=coord_dim,
            radius=in_gno_radius,
            max_neighbors=in_gno_max_neighbors,
            transform_type=in_gno_transform_type,
            reduction="mean",
            pos_embedding_type=in_gno_pos_embedding_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_hidden_layers=in_gno_channel_mlp_hidden_layers,
            activation=gno_activation,
            query_chunk_size=gno_query_chunk_size,
        )

        lift_in_channels = in_gno_out_channels + self.latent_feature_channels
        lift_hidden_channels = int(hidden_channels * lift_channel_ratio)
        self.lifting = PointwiseMLP(
            key=k_lift,
            layers=(
                lift_in_channels,
                *([lift_hidden_channels] * (n_lift_layers - 1)),
                hidden_channels,
            ),
            activations=activation,
        )

        self.fno_blocks = FNOBlocks(
            key=k_fno,
            n_layers=n_layers,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=modes_tuple,
            activation=activation,
            use_channel_mlp=use_channel_mlp,
            preactivation=preactivation,
            normalization=normalization,
            norm_groups=norm_groups,
            use_fno_residual=use_fno_residual,
            local_operator=local_operator,
            use_local_operator_bias=use_local_operator_bias,
            channel_mlp_residual=channel_mlp_residual,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_activations=channel_mlp_activations,
            channel_mlp_dropout=channel_mlp_dropout,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            fft_norm=fft_norm,
            is_complex_data=is_complex_data,
            ranks=ranks,
            init_std=init_std,
            factorization=factorization,
            implementation=implementation,
            separable=separable,
        )

        if gno_weighting_function is not None:
            weighting_fn = make_weighting_fn(
                gno_weighting_function,
                sq_radius=out_gno_radius**2,
                scale=gno_weight_function_scale,
            )
        else:
            weighting_fn = None
        self.out_gno = GNOBlock(
            key=k_out,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            coord_dim=coord_dim,
            radius=out_gno_radius,
            max_neighbors=out_gno_max_neighbors,
            transform_type=out_gno_transform_type,
            reduction="sum",
            weighting_fn=weighting_fn,
            pos_embedding_type=out_gno_pos_embedding_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_hidden_layers=out_gno_channel_mlp_hidden_layers,
            activation=gno_activation,
            query_chunk_size=gno_query_chunk_size,
        )

        proj_hidden_channels = int(hidden_channels * proj_channel_ratio)
        self.projection = PointwiseMLP(
            key=k_proj,
            layers=(
                hidden_channels,
                *([proj_hidden_channels] * (n_proj_layers - 1)),
                out_channels,
            ),
            activations=activation,
        )

    @override
    def __call__(
        self,
        u: Float[Array, "in_c n"] | None,
        *,
        x_in: Float[Array, "n d"],
        x_latent: Float[Array, "... d"],
        x_out: Float[Array, "m d"] | dict[str, Float[Array, "_ d"]] | None = None,
        latent_features: Float[Array, "lat_c ..."] | None = None,
        in_quadrature_weights: Float[Array, "n"] | None = None,
        latent_quadrature_weights: Float[Array, "..."] | None = None,
        in_neighbors: Neighbors | None = None,
        out_neighbors: Neighbors | dict[str, Neighbors] | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "out_c m"] | dict[str, Float[Array, "out_c _"]]:
        """Forward pass of the GINO model.

        Args:
            u: Input function sampled at `x_in`, shaped (in_channels, n).
                Can be `None` if the input GNO transform doesn't need it
                (`"linear"`, `"linear_kernelonly"`), in which case the model
                only sees the geometry.
            x_in: Coordinates of the `n` input points, shaped (n, coord_dim).
            x_latent: Coordinates of the regular latent grid,
                shaped (g1, ..., gk, coord_dim), where `k = len(modes)`.
                The grid is typically a uniform grid on the unit cube or on
                the bounding box of the geometry. The physical radii of the
                GNOs are in the units of these coordinates.
            x_out: Coordinates of the output queries, shaped (m, coord_dim).
                Pass a dict of such arrays to query at several sets of points,
                e.g. the volume and the surface of the domain, which returns
                a dict with the same keys. Default is `None`, which
                queries at `x_in`.
            latent_features: Optional features on the latent grid, shaped
                (latent_feature_channels, g1, ..., gk).
                Required iff `latent_feature_channels` was set.
            in_quadrature_weights: Optional quadrature weights of the `n` input
                points `x_in` for the input GNO, see `IntegralTransform`.
                Default is `None`.
            latent_quadrature_weights: Optional quadrature weights of the latent
                grid points for the output GNO, shaped (g1, ..., gk) like the
                latent grid, e.g. the cell volume. Default is `None`.
            in_neighbors: Optional precomputed neighbors in `x_in` of the
                flattened latent points `x_latent`. Default is `None`, the
                neighbors are searched in every call.
            out_neighbors: Optional precomputed neighbors in the flattened
                `x_latent` of the output queries. A dict with the keys of
                `x_out` if `x_out` is a dict. Default is `None`.
            key: PRNG key used for dropout masks in FNO blocks.
            inference: If True, dropout is disabled.

        Returns:
            The output function sampled at `x_out`, shaped (out_channels, m),
            or a dict of such arrays if `x_out` is a dict.

        Raises:
            ValueError: If `x_latent` doesn't have `len(modes)` grid axes.
            ValueError: If `latent_features` are missing, unexpected or have a
                shape other than the latent grid.
            ValueError: If `out_neighbors` is a dict but `x_out` is not
                or the keys of both differ.
            ValueError: If `latent_quadrature_weights` doesn't have the shape of
                the latent grid.
        """
        grid_shape = x_latent.shape[:-1]
        if len(grid_shape) != self.latent_dim:
            raise ValueError(
                f"x_latent must have {self.latent_dim} grid axes (and a trailing "
                f"coordinate axis), but got shape {x_latent.shape}."
            )
        if (latent_features is None) != (self.latent_feature_channels == 0):
            raise ValueError(
                "latent_features must be passed if and only if "
                "latent_feature_channels was set."
            )
        if latent_features is not None and latent_features.shape != (
            self.latent_feature_channels,
            *grid_shape,
        ):
            raise ValueError(
                f"Expected latent_features with shape "
                f"{(self.latent_feature_channels, *grid_shape)}, "
                f"but got {latent_features.shape}."
            )
        if _neighbors_structure_mismatch(x_out, out_neighbors):
            raise ValueError(
                "out_neighbors must be a dict with the keys of x_out "
                "if and only if x_out is a dict."
            )
        if (
            latent_quadrature_weights is not None
            and latent_quadrature_weights.shape != grid_shape
        ):
            raise ValueError(
                f"Expected latent_quadrature_weights with shape {grid_shape}, "
                f"but got {latent_quadrature_weights.shape}."
            )

        x_latent_flat = x_latent.reshape(-1, x_latent.shape[-1])

        h = self.in_gno(
            x_in,
            x_latent_flat,
            u,
            quadrature_weights=in_quadrature_weights,
            neighbors=in_neighbors,
        )
        h = h.reshape(h.shape[0], *grid_shape)
        if latent_features is not None:
            h = jnp.concatenate([h, latent_features], axis=0)

        h = self.lifting(h)
        h = self.fno_blocks(h, key=key, inference=inference)

        h = h.reshape(h.shape[0], -1)
        if self.out_gno_tanh:
            h = jnp.tanh(h)

        latent_weights_flat = (
            None
            if latent_quadrature_weights is None
            else latent_quadrature_weights.reshape(-1)
        )

        def query(
            x_query: Float[Array, "_ d"], neighbors: Neighbors | None
        ) -> Float[Array, "out_c _"]:
            return self.projection(
                self.out_gno(
                    x_latent_flat,
                    x_query,
                    h,
                    quadrature_weights=latent_weights_flat,
                    neighbors=neighbors,
                )
            )

        if x_out is None:
            x_out = x_in
        if isinstance(x_out, dict):
            return {
                name: query(
                    x_query, None if out_neighbors is None else out_neighbors[name]
                )
                for name, x_query in x_out.items()
            }
        return query(x_out, out_neighbors)


def _neighbors_structure_mismatch(
    x_out: object, out_neighbors: Neighbors | dict[str, Neighbors] | None
) -> bool:
    """Checks whether `x_out` and `out_neighbors` have incompatible structure."""
    if out_neighbors is None:
        return False
    if isinstance(x_out, dict):
        return not (
            isinstance(out_neighbors, dict) and x_out.keys() == out_neighbors.keys()
        )
    return isinstance(out_neighbors, dict)
