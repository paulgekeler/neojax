# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the Geometry Aware Operator Transformer (GAOT)."""

from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.nn.magno import MAGNOBlock
from neojax.nn.neighbor_search import Neighbors
from neojax.nn.patching import patchify, unpatchify
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import SinusoidalEmbedding
from neojax.nn.uvit import UViT


@final
class GAOT(BaseNO):
    r"""Geometry Aware Operator Transformer (GAOT).

    GAOT learns a map between functions given on arbitrary point clouds or meshes
    in $d$ dimensions. It consists of

    1. a lifting `PointwiseMLP` of the input channels,
    2. a `MAGNOBlock` (encoder) that maps the function from the mesh to the points of
       a regular latent grid,
    3. a `PointwiseMLP` on the patches of the latent grid (`patchify`) with the
       positional encoding of the patches, absolute (sinusoidal) or rotary,
    4. a `UViT` transformer over the patches, unpatchified to the latent grid again,
    5. a `MAGNOBlock` (decoder) that maps the latent function to the output queries,
    6. a projection `PointwiseMLP` to the output channels.

    The `MAGNOBlock`s are multiscale attentional graph neural operators with a
    geometry embedding of the neighborhoods.

    For a fixed geometry, precompute the neighbors once
    with `build_neighbors` and pass them with `in_neighbors` and `out_neighbors`.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of channels of the input function.
        out_channels: Number of channels of the output function.
        latent_resolution: Number of points of the regular latent grid per axis,
            e.g. `(64, 64)`. Its length is the dimension of the coordinates.
            Every entry must be divisible by `patch_size`.
        domain_bounds: Bounds `((low, high), ...)` of the domain per axis.
            The coordinates `x_in` and `x_out` are affinely rescaled from these
            bounds to $[-1, 1]$ in the model, which is the domain of the latent
            grid. Default is `None`, the coordinates are already in $[-1, 1]^d$.
        lifting_channels: Number of channels of the latent function, i.e. of the
            lifted input and of the outputs of the encoder `MAGNOBlock`. Default is 32.
        patch_size: Number of latent grid points per axis of a patch.
            Default is 8.
        hidden_channels: Number of channels of the transformer. Default is 256.
        n_layers: Number of transformer blocks. Default is 3.
        n_heads: Number of attention heads of the transformer. Default is 8.
        n_kv_heads: Number of key and value heads of the transformer,
            see `MultiHeadAttention`. Default is `None`, which is `n_heads`.
        ffn_multiplier: Ratio of the hidden channels of the feed-forward networks of
            the transformer to `hidden_channels`. Default is 4.
        use_attn_norm: Whether to normalize the inputs of the attentions.
            Default is `True`.
        use_ffn_norm: Whether to normalize the inputs of the feed-forward networks.
            Default is `True`.
        norm_eps: Epsilon of the normalizations of the transformer. Default is 1e-6.
        use_long_range_skip: Whether the second half of the transformer blocks merges
            the outputs of the first half, see `UViT`. Default is `True`.
        attention_dropout: Dropout probability of the attention weights of the
            transformer. Default is 0.0.
        positional_embedding: Positional encoding of the patches, `"absolute"`
            (sinusoidal, added to the tokens) or `"rope"` (rotary embedding of the queries
            and keys of the transformer). Default is `"absolute"`.
        radius: Base radius of the neighborhoods of the `MAGNOBlock`s, in the units of
            the rescaled coordinates in $[-1, 1]$. It should be of the order of the
            spacing `2 / (resolution - 1)` of the latent grid, so that each latent point
            sees input points. Default is 0.033.
        scales: Factors of the radius for the multiscale neighborhoods of the
            `MAGNOBlock`s. Default is `(1.0,)`.
        max_neighbors: Width of the padded neighbor lists of each scale, i.e. the
            maximum number of neighbors per query. Default is 32.
        gno_hidden_channels: Width of the hidden layers of the kernel networks of the
            `MAGNOBlock`s. Default is 64.
        gno_n_layers: Number of hidden layers of the kernel networks. Default is 3.
        gno_transform_type: Integral transform of the `MAGNOBlock`s,
            see `IntegralTransform`. Default is `"linear"`.
        use_attention: Whether the `MAGNOBlock`s weigh the neighbors by attention.
            Default is `True`.
        attention_type: Type of the attention, `"cosine"` or `"dot_product"`,
            see `NeighborAttention`. Default is `"cosine"`.
        attention_dim: Dimension of the projections of the `"dot_product"` attention.
            Default is 64.
        use_geoembed: Whether the `MAGNOBlock`s use a `GeometricEmbedding`.
            Default is `True`.
        embedding_method: Method of the geometric embedding, `"statistical"` or
            `"pointnet"`. Default is `"statistical"`.
        pooling: Pooling of the `"pointnet"` geometric embedding, `"max"`
            or `"mean"`. Default is `"max"`.
        use_scale_weights: Whether the `MAGNOBlock`s learn weights to combine the scales,
            instead of averaging. Default is `False`.
        node_embedding: Whether the `MAGNOBlock`s embed the coordinates with a
            `"linear"` `SinusoidalEmbedding`. Default is `False`.
        node_embedding_frequencies: Number of frequencies per coordinate of the
            node embedding. Default is 4.
        edge_drop_strategy: How to randomly drop neighbors during training,
            `"ratio"` or `"max_neighbors"`, see `edge_drop`. Default is `None`.
        edge_drop_ratio: Probability to keep a neighbor for the `"ratio"` strategy.
        edge_drop_max_neighbors: Number of kept neighbors for the `"max_neighbors"`
            strategy.
        gno_activation: Activation function of the kernel networks of the `MAGNOBlock`s.
            Default is `jax.nn.gelu`.
        gno_query_chunk_size: Number of query points the neighbor searches and
            integral transforms of the `MAGNOBlock`s process at a time to limit peak
            memory. Default is `None`, all points at once.

    Raises:
        ValueError: If `in_channels`, `out_channels`, `lifting_channels` or `patch_size`
            is not a positive integer.
        ValueError: If `latent_resolution` is empty or an entry is not divisible by
            `patch_size`.
        ValueError: If `domain_bounds` does not have one pair `low < high` per axis.
        ValueError: If `positional_embedding` is invalid.
        ValueError: If `positional_embedding` is `"absolute"` and
            `patch_size ** len(latent_resolution) * lifting_channels` is not divisible
            by `2 * len(latent_resolution)`.
        ValueError: If `positional_embedding` is `"rope"` and the head dimension
            `hidden_channels // n_heads` is odd.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **lifting** (`PointwiseMLP`): Lifts the input channels to `lifting_channels`.
        * **encoder** (`MAGNOBlock`): Maps the mesh to the latent grid.
        * **patch_linear** (`PointwiseMLP`): Linear layer on the tokens of the patches.
        * **pos_embedding** (`SinusoidalEmbedding | None`): Absolute positional encoding of the patches.
        * **processor** (`UViT`): The transformer on the patches of the latent grid.
        * **decoder** (`MAGNOBlock`): Maps the latent grid to the queries.
        * **projection** (`PointwiseMLP`): Projects to `out_channels`.
        * **in_channels** (`int`): Number of channels of the input function.
        * **out_channels** (`int`): Number of channels of the output function.
        * **lifting_channels** (`int`): Number of channels of the latent function.
        * **latent_resolution** (`tuple[int, ...]`): Points of the latent grid per axis.
        * **domain_bounds** (`tuple[tuple[float, float], ...] | None`): Bounds of the domain.
        * **patch_size** (`int`): Latent grid points per axis of a patch.
        * **positional_embedding** (`Literal["absolute", "rope"]`): Positional encoding of the patches.

    !!! warning "Normalize the coordinates"
        The latent grid is on $[-1, 1]^d$ and the radii of the `MAGNOBlock`s are in
        these units. The cosine attention depends on the origin of the coordinates.
        Either pass coordinates in $[-1, 1]^d$ or the `domain_bounds` of the
        coordinates. The rescaling is per axis, which does not preserve the aspect ratio
        of the domain (as in the reference implementation).

    !!! warning "Neighbor truncation"
        The `MAGNOBlock`s consider at most `max_neighbors` neighbors per query and
        scale (the nearest ones), silently. Check with `neighbor_truncation` and choose
        `max_neighbors` large enough for your point densities, radii and scales.

    !!! info "Differences to the reference implementation"
        * The neighbors are padded fixed-width lists instead of ragged lists
          (CSR), to be compatible with `jax.jit`.
        * The random neighbor sampling is called `edge_drop_strategy`
          and `max_neighbors` is the width of the neighbor lists.
        * The learned scale weights of the `MAGNOBlock`s act on the rescaled coordinates,
          also with `node_embedding`.
        * The time-conditional normalization and the autoregressive rollout
          of the reference are not part of the model.

    ??? cite

        [Geometry Aware Operator Transformer as an Efficient and
        Accurate Neural Surrogate for PDEs on Arbitrary Domains](
        https://arxiv.org/abs/2505.18781)

        ```bibtex
        @article{wen2025goat,
            title={Geometry Aware Operator Transformer as an Efficient and
                Accurate Neural Surrogate for PDEs on Arbitrary Domains},
            author={Wen, Shizheng and Kumbhat, Arsh and Lingsch, Levi and
                Mousavi, Sepehr and Zhao, Yizhou and Chandrashekar, Praveen
                and Mishra, Siddhartha},
            journal={arXiv preprint arXiv:2505.18781},
            year={2025}
        }
        ```

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.models import GAOT

        key = jr.key(0)
        model_key, data_key = jr.split(key)
        model = GAOT(
            key=model_key,
            in_channels=2,
            out_channels=1,
            latent_resolution=(16, 16),
            patch_size=2,
            lifting_channels=8,
            hidden_channels=32,
            n_heads=4,
            radius=0.2,
        )
        x_in = jr.uniform(data_key, (500, 2), minval=-1.0, maxval=1.0)
        u = jnp.ones((4, 2, 500))  # batch of 4 functions on the same mesh
        out = jax.vmap(lambda u: model(u, x_in=x_in, inference=True))(u)
        ```
    """

    lifting: PointwiseMLP
    encoder: MAGNOBlock
    patch_linear: PointwiseMLP
    pos_embedding: SinusoidalEmbedding | None
    processor: UViT
    decoder: MAGNOBlock
    projection: PointwiseMLP
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    lifting_channels: int = eqx.field(static=True)
    latent_resolution: tuple[int, ...] = eqx.field(static=True)
    domain_bounds: tuple[tuple[float, float], ...] | None = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    positional_embedding: Literal["absolute", "rope"] = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        latent_resolution: Sequence[int],
        domain_bounds: Sequence[Sequence[int | float]] | None = None,
        lifting_channels: int = 32,
        patch_size: int = 8,
        hidden_channels: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        ffn_multiplier: int | float = 4,
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-6,
        use_long_range_skip: bool = True,
        attention_dropout: float = 0.0,
        positional_embedding: Literal["absolute", "rope"] = "absolute",
        radius: float = 0.033,
        scales: Sequence[float] = (1.0,),
        max_neighbors: int = 32,
        gno_hidden_channels: int = 64,
        gno_n_layers: int = 3,
        gno_transform_type: Literal[
            "linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"
        ] = "linear",
        use_attention: bool = True,
        attention_type: Literal["cosine", "dot_product"] = "cosine",
        attention_dim: int = 64,
        use_geoembed: bool = True,
        embedding_method: Literal["statistical", "pointnet"] = "statistical",
        pooling: Literal["max", "mean"] = "max",
        use_scale_weights: bool = False,
        node_embedding: bool = False,
        node_embedding_frequencies: int = 4,
        edge_drop_strategy: Literal["ratio", "max_neighbors"] | None = None,
        edge_drop_ratio: float | None = None,
        edge_drop_max_neighbors: int | None = None,
        gno_activation: Callable = jax.nn.gelu,
        gno_query_chunk_size: int | None = None,
    ) -> None:
        for name, value in [
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("lifting_channels", lifting_channels),
            ("patch_size", patch_size),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        resolution = tuple(int(r) for r in latent_resolution)
        ndim = len(resolution)
        if ndim == 0:
            raise ValueError("latent_resolution cannot be empty.")
        if any(r % patch_size != 0 or r <= 0 for r in resolution):
            raise ValueError(
                f"Every entry of latent_resolution {resolution} must be a positive "
                f"multiple of patch_size ({patch_size})."
            )

        if domain_bounds is None:
            bounds = None
        else:
            bounds = tuple((float(b[0]), float(b[1])) for b in domain_bounds)
            if len(bounds) != ndim:
                raise ValueError(
                    f"domain_bounds has {len(bounds)} entries but latent_resolution "
                    f"has {ndim}."
                )
            if any(low >= high for low, high in bounds):
                raise ValueError(
                    f"Every domain bound must satisfy low < high, but got {bounds}."
                )

        if positional_embedding not in ["absolute", "rope"]:
            raise ValueError(
                "positional_embedding must be one of ['absolute', 'rope'], "
                f"but got '{positional_embedding}'."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.latent_resolution = resolution
        self.domain_bounds = bounds
        self.patch_size = patch_size
        self.positional_embedding = positional_embedding

        token_channels = patch_size**ndim * lifting_channels
        if positional_embedding == "absolute":
            if token_channels % (2 * ndim) != 0:
                raise ValueError(
                    "The absolute positional embedding needs patch_size**ndim * "
                    f"lifting_channels ({token_channels}) to be divisible by "
                    f"2 * ndim ({2 * ndim})."
                )
            # 10000 ** (-i / F) = (1 / 100) ** (2 i / F), as in the reference
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=ndim,
                num_frequencies=token_channels // (2 * ndim),
                embedding_type="transformer",
                max_positions=100,
            )
        else:
            self.pos_embedding = None

        k_lift, k_enc, k_patch, k_proc, k_dec, k_proj = jr.split(key, 6)
        self.lifting = PointwiseMLP(k_lift, (in_channels, lifting_channels))

        magno_kwargs = dict(
            channels=lifting_channels,
            coord_dim=ndim,
            radius=radius,
            scales=scales,
            max_neighbors=max_neighbors,
            hidden_channels=gno_hidden_channels,
            n_layers=gno_n_layers,
            transform_type=gno_transform_type,
            use_attention=use_attention,
            attention_type=attention_type,
            attention_dim=attention_dim,
            use_geoembed=use_geoembed,
            embedding_method=embedding_method,
            pooling=pooling,
            use_scale_weights=use_scale_weights,
            node_embedding=node_embedding,
            node_embedding_frequencies=node_embedding_frequencies,
            edge_drop_strategy=edge_drop_strategy,
            edge_drop_ratio=edge_drop_ratio,
            edge_drop_max_neighbors=edge_drop_max_neighbors,
            activation=gno_activation,
            query_chunk_size=gno_query_chunk_size,
        )
        self.encoder = MAGNOBlock(key=k_enc, **magno_kwargs)
        self.decoder = MAGNOBlock(key=k_dec, **magno_kwargs)

        self.patch_linear = PointwiseMLP(k_patch, (token_channels, token_channels))
        self.processor = UViT(
            key=k_proc,
            in_channels=token_channels,
            out_channels=token_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_multiplier=ffn_multiplier,
            use_attn_norm=use_attn_norm,
            use_ffn_norm=use_ffn_norm,
            norm_eps=norm_eps,
            use_long_range_skip=use_long_range_skip,
            dropout=attention_dropout,
            use_rope=positional_embedding == "rope",
        )
        self.projection = PointwiseMLP(k_proj, (lifting_channels, out_channels))

    def _rescale(self, x: Float[Array, "n d"]) -> Float[Array, "n d"]:
        """Maps coordinates from the domain bounds to [-1, 1]."""
        if self.domain_bounds is None:
            return x
        low = jnp.asarray([b[0] for b in self.domain_bounds], dtype=jnp.float32)
        high = jnp.asarray([b[1] for b in self.domain_bounds], dtype=jnp.float32)
        return 2.0 * (x - low) / (high - low) - 1.0

    def _latent_points(self) -> Float[Array, "g d"]:
        """Points of the regular latent grid on [-1, 1], flattened row-major."""
        axes = [
            jnp.linspace(-1.0, 1.0, r, dtype=jnp.float32)
            for r in self.latent_resolution
        ]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        return grid.reshape(-1, len(self.latent_resolution))

    def _check_coordinates(self, x: Float[Array, "n d"], name: str) -> None:
        if x.ndim != 2 or x.shape[1] != len(self.latent_resolution):
            raise ValueError(
                f"{name} must be shaped (n, {len(self.latent_resolution)}), "
                f"but got {x.shape}."
            )

    def _patch_positions(self, dtype: DTypeLike) -> Float[Array, "t d"]:
        """Positions of the patches in the patch grid, flattened row-major."""
        grid_shape = [r // self.patch_size for r in self.latent_resolution]
        axes = [jnp.arange(n, dtype=jnp.float32).astype(dtype) for n in grid_shape]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        return grid.reshape(-1, len(grid_shape))

    def build_neighbors(
        self,
        x_in: Float[Array, "n d"],
        x_out: Float[Array, "m d"] | None = None,
    ) -> tuple[tuple[Neighbors, ...], tuple[Neighbors, ...]]:
        """Searches the neighbors of the encoder and the decoder.

        Useful to precompute the neighbors of a fixed geometry once and pass them
        to `__call__`. Vmap this method to precompute them for a batch of geometries.

        Args:
            x_in: Coordinates of the `n` input points, shaped (n, coord_dim).
            x_out: Coordinates of the `m` output queries, shaped (m, coord_dim).
                Default is `None`, which queries at `x_in`.

        Returns:
            The neighbors `(in_neighbors, out_neighbors)` with one `Neighbors` per scale
            each: the neighbors in `x_in` of the latent points and the neighbors in the
            latent points of `x_out`.

        Raises:
            ValueError: If the coordinates do not have `coord_dim` components.
        """
        self._check_coordinates(x_in, "x_in")
        if x_out is None:
            x_out = x_in
        self._check_coordinates(x_out, "x_out")
        x_in, x_out = self._rescale(x_in), self._rescale(x_out)
        x_latent = self._latent_points()
        return (
            self.encoder.build_neighbors(x_in, x_latent),
            self.decoder.build_neighbors(x_latent, x_out),
        )

    @override
    def __call__(
        self,
        u: Float[Array, "in_c n"],
        *,
        x_in: Float[Array, "n d"],
        x_out: Float[Array, "m d"] | None = None,
        in_neighbors: tuple[Neighbors, ...] | None = None,
        out_neighbors: tuple[Neighbors, ...] | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "out_c m"]:
        """Forward pass of the GAOT model.

        Args:
            u: Input function sampled at `x_in`, shaped (in_channels, n).
            x_in: Coordinates of the `n` input points, shaped (n, coord_dim),
                in `domain_bounds` or in [-1, 1] if there are none.
            x_out: Coordinates of the `m` output queries, shaped (m, coord_dim).
                Default is `None`, which queries at `x_in`.
            in_neighbors: Optional precomputed neighbors of the encoder, one per scale,
                see `build_neighbors`. Default is `None`, they are searched in every call.
            out_neighbors: Optional precomputed neighbors of the decoder, one per scale.
                Default is `None`.
            key: PRNG key for the dropout of the attention weights and the random
                neighbor sampling. Needed if either is active and `inference` is `False`.
            inference: If `True`, dropout and neighbor sampling are disabled.

        Returns:
            The output function sampled at `x_out`, shaped (out_channels, m).

        Raises:
            ValueError: If `u` does not have `in_channels` channels or `n` points.
            ValueError: If the coordinates do not have `coord_dim` components.
        """
        self._check_coordinates(x_in, "x_in")
        if x_out is None:
            x_out = x_in
        self._check_coordinates(x_out, "x_out")
        if u.ndim != 2 or u.shape[0] != self.in_channels or u.shape[1] != x_in.shape[0]:
            raise ValueError(
                f"Expected u shaped ({self.in_channels}, {x_in.shape[0]}), "
                f"but got {u.shape}."
            )

        if key is None:
            k_enc = k_proc = k_dec = None
        else:
            k_enc, k_proc, k_dec = jr.split(key, 3)

        # compute latent grid and neighbor search in float32
        # keep low precision for integral transform/attention in GNOBlock
        x_in, x_out = self._rescale(x_in), self._rescale(x_out)
        x_latent = self._latent_points()

        if in_neighbors is None:
            in_neighbors = self.encoder.build_neighbors(x_in, x_latent)
        if out_neighbors is None:
            out_neighbors = self.decoder.build_neighbors(x_latent, x_out)

        h = self.lifting(u)  # (c, n)
        h = self.encoder(
            x_in.astype(h.dtype),
            x_latent.astype(h.dtype),
            h,
            neighbors=in_neighbors,
            key=k_enc,
            inference=inference,
        )  # (c, g)

        h = h.reshape(self.lifting_channels, *self.latent_resolution)
        tokens = self.patch_linear(patchify(h, self.patch_size))
        if self.pos_embedding is not None:
            tokens = tokens + self.pos_embedding(self._patch_positions(tokens.dtype)).T
        tokens = self.processor(tokens, key=k_proc, inference=inference)
        h = unpatchify(tokens, self.patch_size, self.latent_resolution)

        h = self.decoder(
            x_latent.astype(h.dtype),
            x_out.astype(h.dtype),
            h.reshape(self.lifting_channels, -1),
            neighbors=out_neighbors,
            key=k_dec,
            inference=inference,
        )  # (c, m)
        return self.projection(h)
