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

"""Implementation of the geometric embedding of the neighborhoods of points."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.neighbor_search import Neighbors
from neojax.nn.pointwise_mlp import PointwiseMLP


class GeometricEmbedding(eqx.Module):
    r"""Embedding of the local geometry around query points.

    Describes the shape of the neighborhood of each query point $x_i$ in a point
    cloud $\{y_j\}$ by a feature vector that depends only on the geometry, not on
    any function values. Two methods are available:

    - `"statistical"`: Computes the number of neighbors $N_i$, the mean $\mu_i$ and
        the variance $\sigma_i^2$ of their distances to $x_i$, the offset
        $\bar{y}_i - x_i$ of their centroid and the eigenvalues of their covariance
        matrix (sorted in descending order), so $3 + 2d$ features in $d$ dimensions.
        The features are standardized over all queries (zero mean, unit unbiased
        standard deviation; constant features are only centered) and mapped to the
        embedding by a `PointwiseMLP`. Queries without neighbors have all zero
        features before the standardization.
    - `"pointnet"`: Applies a shared `PointwiseMLP` to the coordinates $y_j - x_i$
        of the neighbors relative to the query, pools the result over the
        neighbors and maps it to the embedding with a `PointwiseMLP`.
        Queries without neighbors have the embedding zero.

    Args:
        key: PRNG key for parameter initialization.
        coord_dim: Dimension $d$ of the points.
        out_channels: Number of channels of the embedding.
        method: How to embed the geometry, `"statistical"` or `"pointnet"`.
            Default is `"statistical"`.
        pooling: How to pool over the neighbors in the `"pointnet"` method,
            `"max"` or `"mean"`. Default is `"max"`.
        hidden_channels: Width of the hidden layer(s) of the `PointwiseMLP`s.
            Default is 64.

    Raises:
        ValueError: If `coord_dim`, `out_channels` or `hidden_channels`
            is not a positive integer.
        ValueError: If `method` or `pooling` is invalid.

    ??? info "Internal Attributes"
        * **mlp** (`PointwiseMLP`): Maps the features (`"statistical"`) or the pooled
            neighbor features (`"pointnet"`) to the embedding.
        * **point_mlp** (`PointwiseMLP | None`): Shared network applied to each
            neighbor in the `"pointnet"` method.
        * **coord_dim** (`int`): Dimension of the points.
        * **out_channels** (`int`): Number of channels of the embedding.
        * **method** (`Literal["statistical", "pointnet"]`): Embedding method.
        * **pooling** (`Literal["max", "mean"]`): Pooling of the `"pointnet"` method.

    !!! info "Gradients"
        The `"statistical"` features are computed with a `stop_gradient` on the
        coordinates: eigenvalue derivatives are ill-defined for degenerate covariance
        matrices. Hence the embedding is not differentiable w.r.t. the coordinates.
        The `"pointnet"` method is.

    !!! info "Dependence on all queries"
        The standardization of the `"statistical"` features uses the statistics of
        the given queries. The embedding of a point thus depends on the other queries.

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

        [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](
        https://arxiv.org/abs/1612.00593)

        ```bibtex
        @inproceedings{qi2017pointnet,
            title={PointNet: Deep Learning on Point Sets for 3D Classification
                and Segmentation},
            author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
            booktitle={Proceedings of the IEEE Conference on Computer Vision
                and Pattern Recognition},
            year={2017}
        }
        ```
    """

    mlp: PointwiseMLP
    point_mlp: PointwiseMLP | None
    coord_dim: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    method: Literal["statistical", "pointnet"] = eqx.field(static=True)
    pooling: Literal["max", "mean"] = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        coord_dim: int,
        out_channels: int,
        method: Literal["statistical", "pointnet"] = "statistical",
        pooling: Literal["max", "mean"] = "max",
        hidden_channels: int = 64,
    ) -> None:
        for name, value in [
            ("coord_dim", coord_dim),
            ("out_channels", out_channels),
            ("hidden_channels", hidden_channels),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if method not in ["statistical", "pointnet"]:
            raise ValueError(
                f"method must be one of ['statistical', 'pointnet'], but got '{method}'."
            )
        if pooling not in ["max", "mean"]:
            raise ValueError(
                f"pooling must be one of ['max', 'mean'], but got '{pooling}'."
            )

        self.coord_dim = coord_dim
        self.out_channels = out_channels
        self.method = method
        self.pooling = pooling

        k_mlp, k_point = jr.split(key)
        if method == "statistical":
            self.point_mlp = None
            self.mlp = PointwiseMLP(
                key=k_mlp,
                layers=(3 + 2 * coord_dim, hidden_channels, out_channels),
                activations=(jax.nn.relu, jax.nn.relu),
            )
        else:
            self.point_mlp = PointwiseMLP(
                key=k_point,
                layers=(coord_dim, hidden_channels, hidden_channels),
                activations=(jax.nn.relu, jax.nn.relu),
            )
            self.mlp = PointwiseMLP(
                key=k_mlp,
                layers=(hidden_channels, out_channels),
                activations=(jax.nn.relu,),
            )

    def _statistical_features(
        self,
        input_geom: Float[Array, "n d"],
        queries: Float[Array, "m d"],
        neighbors: Neighbors,
    ) -> Float[Array, "m f"]:
        """Standardized statistics of the neighborhoods, shaped (m, 3 + 2d)."""
        m = queries.shape[0]
        # promote to float32 to avoid cancellation
        # and NotImplementedError for (b)float16 (in linalg.eigvalsh)
        compute_dtype = jnp.promote_types(input_geom.dtype, jnp.float32)
        in_dtype = input_geom.dtype
        geom = jax.lax.stop_gradient(input_geom).astype(compute_dtype)
        q = jax.lax.stop_gradient(queries).astype(compute_dtype)

        query_index = neighbors.query_index
        safe_query = jnp.minimum(query_index, m - 1)

        def per_query(values: Array) -> Array:
            return jax.ops.segment_sum(values, query_index, m, indices_are_sorted=True)

        nbr = geom[neighbors.data_index]  # (e, d)
        count = per_query(jnp.ones(nbr.shape[0], compute_dtype))
        safe_count = jnp.maximum(count, 1.0)

        dist = jnp.sqrt(jnp.sum((nbr - q[safe_query]) ** 2, axis=-1))
        mean_dist = per_query(dist) / safe_count
        mean_sq_dist = per_query(dist**2) / safe_count
        var_dist = jnp.maximum(mean_sq_dist - mean_dist**2, 0.0)

        centroid = per_query(nbr) / safe_count[:, None]
        offset = centroid - q

        centered = nbr - centroid[safe_query]
        cov = per_query(centered[:, :, None] * centered[:, None, :])
        cov = cov / safe_count[:, None, None]
        eigenvalues = jnp.linalg.eigvalsh(cov)[:, ::-1]  # descending

        features = jnp.concatenate(
            [
                count[:, None],
                mean_dist[:, None],
                var_dist[:, None],
                offset,
                eigenvalues,
            ],
            axis=-1,
        )
        features = jnp.where((count > 0)[:, None], features, 0.0)

        mean = jnp.mean(features, axis=0, keepdims=True)
        std = jnp.std(features, axis=0, keepdims=True, ddof=1)
        # also covers the NaN of a single query
        std = jnp.where(std >= 1e-6, std, 1.0)
        # cast back to input dtype
        return ((features - mean) / std).astype(in_dtype)

    def _pointnet(
        self,
        input_geom: Float[Array, "n d"],
        queries: Float[Array, "m d"],
        neighbors: Neighbors,
    ) -> Float[Array, "out_c m"]:
        """PointNet embedding of the neighborhoods."""
        m = queries.shape[0]
        query_index = neighbors.query_index
        safe_query = jnp.minimum(query_index, m - 1)
        centered = input_geom[neighbors.data_index] - queries[safe_query]  # (e, d)
        features = self.point_mlp(centered.T).T  # (e, h) >= 0

        # imported here: `neojax.utils` imports the models, which import this module
        from neojax.utils.segment_ops import segment_max, segment_mean

        if self.pooling == "max":
            pooled = segment_max(features, query_index, m, indices_are_sorted=True)
        else:
            pooled = segment_mean(features, query_index, m, indices_are_sorted=True)

        has_neighbors = neighbors.neighbor_counts > 0
        return jnp.where(has_neighbors[None], self.mlp(pooled.T), 0.0)

    def __call__(
        self,
        input_geom: Float[Array, "n d"],
        queries: Float[Array, "m d"],
        neighbors: Neighbors,
    ) -> Float[Array, "out_c m"]:
        """Embeds the neighborhoods of the query points.

        Args:
            input_geom: The `n` points of the point cloud the neighbors index into.
            queries: The `m` query points.
            neighbors: The edges between the queries and their neighbors in
                `input_geom`.

        Returns:
            The geometric embedding of the queries.

        Raises:
            ValueError: If the points do not have `coord_dim` coordinates.
        """
        if (
            input_geom.shape[-1] != self.coord_dim
            or queries.shape[-1] != self.coord_dim
        ):
            raise ValueError(
                f"Expected points with {self.coord_dim} coordinates, but got shapes "
                f"{input_geom.shape} and {queries.shape}."
            )
        if self.method == "statistical":
            return self.mlp(
                self._statistical_features(input_geom, queries, neighbors).T
            )
        return self._pointnet(input_geom, queries, neighbors)
