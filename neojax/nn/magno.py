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

"""Implementation of the Multiscale Attentional Graph Neural Operator block."""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.edge_drop import edge_drop
from neojax.nn.geometric_embedding import GeometricEmbedding
from neojax.nn.gno_block import GNOBlock
from neojax.nn.neighbor_search import (
    Neighbors,
    RegularGrid,
    radius_neighbor_search,
    radius_neighbor_search_grid,
)
from neojax.nn.pointwise_mlp import PointwiseMLP


class MAGNOBlock(eqx.Module):
    r"""Multiscale Attentional Graph Neural Operator (MAGNO) block.

    Maps a function sampled on the points $y$ to a function sampled on the points $x$.
    It is the encoder (mesh to latent grid) and the decoder (latent grid to mesh)
    of `GAOT`. For each scale $s$ in `scales` it

    - computes the neighbors of each $x$ in $y$ within the radius `radius * s`,
    - applies one shared attentional `GNOBlock` over these neighbors,
    - optionally concatenates the `GeometricEmbedding` of the neighborhoods and
       merges the result back to `channels` with a linear "recovery" layer.

    The results of the scales are averaged or, with `use_scale_weights`, combined with
    weights $w_s(x)$ that are a softmax over the scales of a `PointwiseMLP` of $x$.

    During training, `edge_drop` randomly drops neighbors as a regularization,
    before they are used by the `GNOBlock` and the geometric embedding.

    If the input points are a `RegularGrid`, as in the decoder of `GAOT`, the
    faster `radius_neighbor_search_grid` is used.

    Args:
        key: PRNG key for parameter initialization.
        channels: Number of channels of the input and output function.
        coord_dim: Dimension of the domain of $x$ and $y$.
        radius: Base radius of the neighborhoods. The radius at scale $s$ is
            `radius * s`. The coordinates should be of order one, e.g. in $[-1, 1]$,
            as in `GAOT`. Default is 0.033.
        scales: Factors of the radius, one scale per entry. All scales share the
            parameters. Default is `(1.0,)`.
        mean_neighbors: Expected mean number of neighbors per query at the base
            radius. The capacity of the edge list of scale $s$ is `mean_neighbors`
            times $s^d$ times the number of queries, following the volume of the
            balls. A neighborhood can be larger, as long as the total fits. The
            search raises an error if it doesn't. Default is 32.
        hidden_channels: Width of the hidden layers of the kernel network of the
            `GNOBlock`. Default is 64.
        n_layers: Number of hidden layers of the kernel network. Default is 3.
        transform_type: Integral transform of the `GNOBlock`, see `IntegralTransform`.
            Default is `"linear"`.
        use_attention: Whether the `GNOBlock` weighs the neighbors by attention.
            Default is `True`.
        attention_type: Type of the attention, `"cosine"` or `"dot_product"`,
            see `NeighborAttention`. Default is `"cosine"`.
        attention_dim: Dimension of the projections of the `"dot_product"`
            attention. Default is 64.
        use_geoembed: Whether to add the `GeometricEmbedding` of the neighborhoods.
            Default is `True`.
        embedding_method: Method of the geometric embedding, `"statistical"` or
            `"pointnet"`. Default is `"statistical"`.
        pooling: Pooling of the `"pointnet"` geometric embedding, `"max"` or
            `"mean"`. Default is `"max"`.
        use_scale_weights: Whether to learn weights to combine the scales,
            instead of averaging them. Default is `False`.
        node_embedding: Whether to embed the coordinates with a `"linear"`
            `SinusoidalEmbedding` before they enter the kernel network and the
            attention. Default is `False`.
        node_embedding_frequencies: Number of frequencies per coordinate
            of the node embedding. Default is 4.
        edge_drop_strategy: How to randomly drop neighbors during training,
            `"ratio"` or `"max_neighbors"` (see `edge_drop`). `None` disables
            it. Default is `None`.
        edge_drop_ratio: Probability to keep a neighbor for the `"ratio"` strategy.
        edge_drop_max_neighbors: Number of kept neighbors for the `"max_neighbors"`
            strategy.
        activation: Activation function of the kernel network.
            Default is `jax.nn.gelu`.
        edge_chunk_size: Number of edges to process at a time in the integral
            transform to limit peak memory. Default is `None`, all edges at once.

    Raises:
        ValueError: If `channels` or `coord_dim` is not a positive integer.
        ValueError: If `radius`, `mean_neighbors` or an entry of `scales` is not
            positive or `scales` is empty.
        ValueError: If `edge_drop_strategy` is invalid or its parameter is missing or invalid.

    ??? info "Internal Attributes"
        * **gno** (`GNOBlock`): The (attentional) GNO shared by the scales.
        * **geoembed** (`GeometricEmbedding | None`): Geometric embedding of the neighborhoods.
        * **recovery** (`PointwiseMLP | None`): Linear layer merging the function and the geometric embedding.
        * **scale_weighting** (`PointwiseMLP | None`): Network computing the weights of the scales.
        * **channels** (`int`): Number of channels of the function.
        * **coord_dim** (`int`): Dimension of the domain.
        * **radius** (`float`): Base radius of the neighborhoods.
        * **scales** (`tuple[float, ...]`): Factors of the radius.
        * **mean_neighbors** (`float`): Expected mean number of neighbors at the base radius.
        * **edge_drop_strategy** (`Literal["ratio", "max_neighbors"] | None`): Neighbor dropping strategy.
        * **edge_drop_ratio** (`float | None`): Keep probability of the `"ratio"` strategy.
        * **edge_drop_max_neighbors** (`int | None`): Kept neighbors of the `"max_neighbors"` strategy.

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
    """

    gno: GNOBlock
    geoembed: GeometricEmbedding | None
    recovery: PointwiseMLP | None
    scale_weighting: PointwiseMLP | None
    channels: int = eqx.field(static=True)
    coord_dim: int = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    scales: tuple[float, ...] = eqx.field(static=True)
    mean_neighbors: float = eqx.field(static=True)
    edge_drop_strategy: Literal["ratio", "max_neighbors"] | None = eqx.field(
        static=True
    )
    edge_drop_ratio: float | None = eqx.field(static=True)
    edge_drop_max_neighbors: int | None = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        channels: int,
        coord_dim: int,
        radius: float = 0.033,
        scales: Sequence[float] = (1.0,),
        mean_neighbors: float | int = 32,
        hidden_channels: int = 64,
        n_layers: int = 3,
        transform_type: Literal[
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
        activation: Callable = jax.nn.gelu,
        edge_chunk_size: int | None = None,
    ) -> None:
        for name, value in [
            ("channels", channels),
            ("coord_dim", coord_dim),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not radius > 0:
            raise ValueError("radius must be positive.")
        if not mean_neighbors > 0:
            raise ValueError("mean_neighbors must be positive.")
        scales = tuple(float(s) for s in scales)
        if len(scales) == 0 or not all(s > 0 for s in scales):
            raise ValueError(
                f"scales must be a non-empty sequence of positive numbers, got {scales}."
            )
        if edge_drop_strategy is None:
            pass
        elif edge_drop_strategy == "ratio":
            if edge_drop_ratio is None or not 0.0 < edge_drop_ratio <= 1.0:
                raise ValueError(
                    "edge_drop_ratio must be in (0, 1] for the 'ratio' strategy."
                )
        elif edge_drop_strategy == "max_neighbors":
            if (
                not isinstance(edge_drop_max_neighbors, int)
                or edge_drop_max_neighbors <= 0
            ):
                raise ValueError(
                    "edge_drop_max_neighbors must be a positive integer "
                    "for the 'max_neighbors' strategy."
                )
        else:
            raise ValueError(
                "edge_drop_strategy must be one of [None, 'ratio', 'max_neighbors'], "
                f"but got '{edge_drop_strategy}'."
            )

        self.channels = channels
        self.coord_dim = coord_dim
        self.radius = float(radius)
        self.scales = scales
        self.mean_neighbors = float(mean_neighbors)
        self.edge_drop_strategy = edge_drop_strategy
        self.edge_drop_ratio = edge_drop_ratio
        self.edge_drop_max_neighbors = edge_drop_max_neighbors

        k_gno, k_geo, k_rec, k_scale = jr.split(key, 4)
        self.gno = GNOBlock(
            key=k_gno,
            in_channels=channels,
            out_channels=channels,
            coord_dim=coord_dim,
            radius=radius,
            mean_neighbors=mean_neighbors,
            transform_type=transform_type,
            reduction="mean",
            pos_embedding_type="linear" if node_embedding else None,
            pos_embedding_channels=node_embedding_frequencies,
            channel_mlp_hidden_layers=(hidden_channels,) * n_layers,
            activation=activation,
            edge_chunk_size=edge_chunk_size,
            attention_type=attention_type if use_attention else None,
            attention_dim=attention_dim,
        )

        if use_geoembed:
            self.geoembed = GeometricEmbedding(
                key=k_geo,
                coord_dim=coord_dim,
                out_channels=channels,
                method=embedding_method,
                pooling=pooling,
            )
            self.recovery = PointwiseMLP(key=k_rec, layers=(2 * channels, channels))
        else:
            self.geoembed = None
            self.recovery = None

        if use_scale_weights:
            self.scale_weighting = PointwiseMLP(
                key=k_scale,
                layers=(coord_dim, max(hidden_channels // 4, 1), len(scales)),
                activations=jax.nn.relu,
            )
        else:
            self.scale_weighting = None

    def build_neighbors(
        self, y: Float[Array, "n d"] | RegularGrid, x: Float[Array, "m d"]
    ) -> tuple[Neighbors, ...]:
        """Searches the neighbors in `y` of each point in `x` for every scale.

        Useful to precompute the neighbors for a fixed geometry
        and pass them to `__call__`.

        Args:
            y: Coordinates of the `n` input points, or a `RegularGrid`.
            x: Coordinates of the `m` output points.

        Returns:
            The edges, one `Neighbors` per scale.
        """
        neighbors = []
        for scale in self.scales:
            mean = self.mean_neighbors * scale**self.coord_dim
            max_edges = max(math.ceil(mean * x.shape[0]), 1)
            radius = self.radius * scale
            if isinstance(y, RegularGrid):
                neighbors.append(radius_neighbor_search_grid(y, x, radius, max_edges))
            else:
                neighbors.append(radius_neighbor_search(y, x, radius, max_edges))
        return tuple(neighbors)

    def __call__(
        self,
        y: Float[Array, "n d"] | RegularGrid,
        x: Float[Array, "m d"],
        f_y: Float[Array, "c n"],
        *,
        neighbors: tuple[Neighbors, ...] | None = None,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "c m"]:
        """Applies the block.

        Args:
            y: Coordinates of the `n` input points, or a `RegularGrid`.
            x: Coordinates of the `m` output points.
            f_y: Input function sampled at `y`.
            neighbors: Optional precomputed neighbors, one `Neighbors` per scale
                as returned by `build_neighbors`. Default is `None`, they are searched.
            key: PRNG key for the random edge dropping. Required if an
                `edge_drop_strategy` is set and `inference` is `False`.
            inference: If `True`, no edges are dropped.

        Returns:
            The output function sampled at `x`.

        Raises:
            ValueError: If `neighbors` doesn't have one entry per scale.
            ValueError: If edges are dropped but `key` is `None`.
        """
        if neighbors is None:
            neighbors = self.build_neighbors(y, x)
        elif len(neighbors) != len(self.scales):
            raise ValueError(
                f"Expected {len(self.scales)} neighbor lists (one per scale), "
                f"but got {len(neighbors)}."
            )
        if isinstance(y, RegularGrid):
            y = y.points().astype(x.dtype)

        drop = self.edge_drop_strategy is not None and not inference
        if drop:
            if key is None:
                raise ValueError(
                    "A PRNG key must be passed when edges are dropped "
                    "(edge_drop_strategy is set and inference=False)."
                )
            keys = jr.split(key, len(self.scales))

        outputs = []
        for i, nb in enumerate(neighbors):
            if drop:
                nb = edge_drop(
                    nb,
                    key=keys[i],
                    strategy=self.edge_drop_strategy,
                    ratio=self.edge_drop_ratio,
                    max_neighbors=self.edge_drop_max_neighbors,
                )
            out = self.gno(y, x, f_y, neighbors=nb)
            if self.geoembed is not None:
                geometry = self.geoembed(y, x, nb)
                out = self.recovery(jnp.concatenate([out, geometry], axis=0))
            outputs.append(out)

        stacked = jnp.stack(outputs)  # (num scales, c, m)
        if self.scale_weighting is None:
            return jnp.mean(stacked, axis=0)
        weights = jax.nn.softmax(self.scale_weighting(x.T), axis=0)  # (num scales, m)
        return jnp.sum(weights[:, None, :] * stacked, axis=0)
