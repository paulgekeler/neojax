"""Implementation of a general n-dimensional Geometry-Aware Fourier Neural Operator (Geo-FNO)."""

from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Inexact, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.fno_blocks import FNOBlocks
from neojax.nn.geo_map import GeoMapNd
from neojax.nn.geo_spectral_conv import GeoSpectralConvNd
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tensor import BaseTensor


@final
class GeoFNO(BaseNO):
    r"""General n-dimensional Geometry-Aware Fourier Neural Operator (Geo-FNO).

    Geo-FNO extends the standard FNO to arbitrary geometries by learning a soft-diffeomorphism
    (using a coordinate deformation network `GeoMapNd`) that maps irregular physical
    domains to a uniform latent computational grid where the standard FFTs
    are computed.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input features.
        out_channels: Number of output features.
        hidden_channels: Hidden channel dimension.
        n_layers: Total number of operator layers (must be >= 2).
        modes: Number of Fourier modes to retain across each dimension.
        grid_resolution: Resolution of the uniform computational latent grid (e.g. (32, 32) for 2D).
        geomap_width: Width of the hidden layers in the GeoMap diffeomorphism network.
        code_dim: Optional dimensionality of domain/parameter codes.
        use_coord_projection: If True, projects coordinates and adds them as bias at each layer.
        activation: Activation function (default gelu).
        use_channel_mlp: Whether to use pointwise channel MLPs.
        local_operator: Type of skip connection inside FNO blocks.
        use_local_operator_bias: Whether to use a bias term.
        channel_mlp_residual: Type of skip connection around channel MLPs.
        channel_mlp_expansion: Expansion factor for hidden dimension in channel MLPs.
        channel_mlp_activations: Activation function(s) inside channel MLPs.
        channel_mlp_dropout: Dropout probability inside channel MLPs.
        normalization: Type of normalization to use (e.g. "layer").
        norm_groups: Number of groups for group normalization.
        use_fno_residual: Whether to use residual connection around FNO blocks.
        preactivation: Whether to use pre-activation style blocks.
        n_lift_layers: Number of layers in the lifting MLP.
        n_proj_layers: Number of layers in the projection MLP.
        lift_channel_ratio: Ratio of lifting channels to hidden_channels.
        proj_channel_ratio: Ratio of projection channels to hidden_channels.
        enforce_hermitian_symmetry: Whether to enforce Hermitian symmetry.
        fft_norm: FFT normalization.
        is_complex_data: Whether input data is complex-valued.
        resolution_scaling_factor: Layerwise scaling factor for domain resolution.
        ranks: Number of ranks for weight tensor factorization.
        init_std: Standard deviation for weight initialization.
        factorization: Tensor factorization type.
        implementation: Weight reconstruction mode.
        separable: Whether to use separable implementation.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **geomap** (`GeoMapNd`): Diffeomorphism coordinate mapping network.
        * **lifting** (`PointwiseMLP`): Pointwise MLP lifting input features.
        * **conv_in** (`GeoSpectralConvNd`): Continuous-to-grid input spectral conv.
        * **fno_blocks** (`FNOBlocks | None`): Intermediate grid FNO blocks.
        * **conv_out** (`GeoSpectralConvNd`): Grid-to-continuous output spectral conv.
        * **projection** (`PointwiseMLP`): Pointwise MLP projecting features to outputs.
        * **coord_projectors** (`tuple[PointwiseMLP, ...] | None`): Layerwise coordinate projection networks.

    ??? cite

        [Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](
        https://arxiv.org/pdf/2207.05209)

        ```bibtex
        @article{li2022fourier,
            title={Fourier Neural Operator with Learned Deformations for PDEs on General Geometries},
            author={Li, Zongyi and Huang, Daniel and Kovachki,
                Nikola and Azizzadenesheli, Kamyar and Bhattacharya,
                Kaushik and Anandkumar, Anima},
            journal={arXiv preprint arXiv:2207.05209},
            year={2022}
        }
        ```

    !!! warning
        The continuous Fourier map performed via direct discrete Fourier transform (DFT)
        summation is computationally very expensive, scaling as $\mathcal{O}(N \times M)$
        where $N$ is the number of input nodes and $M$ is the grid size. This is
        noticeable for larger spatial dimensions.
    """

    geomap: GeoMapNd
    lifting: PointwiseMLP
    conv_in: GeoSpectralConvNd
    fno_blocks: FNOBlocks | None
    conv_out: GeoSpectralConvNd
    projection: PointwiseMLP
    coord_projectors: tuple[PointwiseMLP, ...] | None

    grid_resolution: tuple[int, ...] = eqx.field(static=True)
    use_coord_projection: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        modes: int | Sequence[int],
        grid_resolution: Sequence[int],
        geomap_width: int = 32,
        code_dim: int | None = None,
        use_coord_projection: bool = True,
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
        if n_layers < 2:
            raise ValueError(
                "n_layers must be at least 2 for GeoFNO (1 input, 1 output layer)."
            )

        self.grid_resolution = tuple(grid_resolution)
        self.use_coord_projection = use_coord_projection
        dim = len(self.grid_resolution)

        k_geo, k_lift, k_conv_in, k_fno, k_conv_out, k_proj, k_b = jr.split(key, 7)

        self.geomap = GeoMapNd(
            key=k_geo, dim=dim, width=geomap_width, code_dim=code_dim
        )

        lift_hidden_channels = int(hidden_channels * lift_channel_ratio)
        lifting_layers = (
            [in_channels]
            + [lift_hidden_channels] * (n_lift_layers - 1)
            + [hidden_channels]
        )
        self.lifting = PointwiseMLP(
            key=k_lift,
            layers=tuple(lifting_layers),
            activations=activation,
        )

        self.conv_in = GeoSpectralConvNd(
            key=k_conv_in,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
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

        if n_layers > 2:
            # Handle resolution scaling sequence for FNOBlocks
            if isinstance(resolution_scaling_factor, Sequence):
                block_res_scaling = resolution_scaling_factor[1:-1]
            else:
                block_res_scaling = resolution_scaling_factor

            self.fno_blocks = FNOBlocks(
                key=k_fno,
                n_layers=n_layers - 2,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
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
                resolution_scaling_factor=block_res_scaling,
                ranks=ranks,
                init_std=init_std,
                factorization=factorization,
                implementation=implementation,
                separable=separable,
            )
        else:
            self.fno_blocks = None

        self.conv_out = GeoSpectralConvNd(
            key=k_conv_out,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
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

        if use_coord_projection:
            b_keys = jr.split(k_b, n_layers)
            projectors = []
            for i in range(n_layers):
                projectors.append(
                    PointwiseMLP(
                        key=b_keys[i],
                        layers=(dim, hidden_channels),
                        activations=activation,
                    )
                )
            self.coord_projectors = tuple(projectors)
        else:
            self.coord_projectors = None

        proj_hidden_channels = int(hidden_channels * proj_channel_ratio)
        projection_layers = (
            [hidden_channels]
            + [proj_hidden_channels] * (n_proj_layers - 1)
            + [out_channels]
        )
        self.projection = PointwiseMLP(
            key=k_proj,
            layers=tuple(projection_layers),
            activations=activation,
        )

    @override
    def __call__(
        self,
        u: Inexact[Array, "in_c ..."],
        *,
        x_in: Float[Array, "N d"] | None = None,
        x_out: Float[Array, "M d"] | None = None,
        code: Float[Array, "..."] | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the Geo-FNO model.

        If `x_in` is None, behaves like standard FNO on regular grids.
        Otherwise, maps unstructured input coordinates to latent uniform grid,
        runs spectral operations on it, and maps back to query coordinates.

        Args:
            u: Input signal. Shape is (in_channels, N) for meshes or (in_channels, ...) for grids.
            x_in: Physical coordinate mesh of the nodes in u. Shape (N, d).
            x_out: Physical query coordinate mesh. Shape (M, d). Defaults to x_in.
            code: Optional domain/conditioning features.
            key: PRNG key for dropout masks.
            inference: Disables dropout if True.

        Returns:
            Solution prediction field of shape (out_channels, M) or (out_channels, ...).
        """
        # If grid input
        if x_in is None:
            h = self.lifting(u)

            h = self.conv_in(h)
            if self.fno_blocks is not None:
                h = self.fno_blocks(h, key=key, inference=inference)
            h = self.conv_out(h)

            return self.projection(h)

        # If mesh input
        if x_out is None:
            x_out = x_in

        d = len(self.grid_resolution)

        h = self.lifting(u)  # (hidden_channels, N)

        # Deform input coordinate domain to latent regular coordinates
        xi_in = self.geomap(x_in, code)  # (N, d)

        # Create regular latent computational grid
        axes = [jnp.linspace(0.0, 1.0, r) for r in self.grid_resolution]
        grids = jnp.meshgrid(*axes, indexing="ij")
        grid = jnp.stack(grids, axis=0)  # (d, res0, res1, ...)
        grid_coords = grid.reshape(d, -1).T  # (res_prod, d)

        # Input Spectral Convolution mapping from continuous space to computational grid
        h_grid = self.conv_in(
            h, x_in=xi_in, x_out=grid_coords
        )  # (hidden_channels, res_prod)
        h_grid = h_grid.reshape(
            (h_grid.shape[0],) + self.grid_resolution
        )  # (hidden_channels, res0, ...)

        # Add Coordinate Embedding Bias (Layer 0)
        if self.coord_projectors is not None:
            h_grid = h_grid + self.coord_projectors[0](grid)
        h_grid = jax.nn.gelu(h_grid)

        # Intermediate FNO blocks on regular latent grid
        if self.fno_blocks is not None:
            n_blocks = len(self.fno_blocks.fno_layers)
            if key is not None and not inference:
                keys = jr.split(key, n_blocks)
            else:
                keys = [None] * n_blocks

            for i in range(n_blocks):
                h_grid = self.fno_blocks.fno_layers[i](h_grid)
                if self.coord_projectors is not None:
                    h_grid = h_grid + self.coord_projectors[i + 1](grid)

                # Apply optional pointwise channel MLP
                if self.fno_blocks.channel_mlps is not None:
                    mlp_out = self.fno_blocks.channel_mlps[i](
                        h_grid, key=keys[i], inference=inference
                    )
                    res_op = self.fno_blocks.channel_mlp_residuals[i]
                    if res_op is not None:
                        h_grid = mlp_out + res_op(h_grid)
                    else:
                        h_grid = mlp_out

        # Output Spectral Convolution mapping back from latent grid to query mesh
        xi_out = self.geomap(x_out, code)  # (M, d)
        h_flat = h_grid.reshape(h_grid.shape[0], -1)
        h_out = self.conv_out(
            h_flat, x_in=grid_coords, x_out=xi_out
        )  # (hidden_channels, M)

        # Add Query Coordinate Bias (Layer L-1)
        if self.coord_projectors is not None:
            h_out = h_out + self.coord_projectors[-1](x_out.T)

        return self.projection(h_out)
