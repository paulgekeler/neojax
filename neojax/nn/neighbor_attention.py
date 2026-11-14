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

"""Implementation of the attention weights over the neighbors of a point."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.neighbor_search import Neighbors
from neojax.nn.pointwise_mlp import PointwiseMLP


class NeighborAttention(eqx.Module):
    r"""Attention weights over the neighbors of each query point.

    Computes a normalized weight $\alpha(x, y_j)$ for each edge between
    a query point $x$ and a neighbor $y_j$ that sums to one over the neighbors of $x$:

    - `"cosine"`: $\alpha(x, y_j) = \operatorname{softmax}_j \, \left(\cos(x, y_j)\right)$,
        the cosine similarity of the (embedded) coordinates,
    - `"dot_product"`: $\alpha(x, y_j) = \operatorname{softmax}_j
        \, \left(\langle W_q x, W_k y_j \rangle / \sqrt{d_a}\right)$ with learnable
        projections $W_q$ and $W_k$ to the attention dimension $d_a$.

    Used by `IntegralTransform` to reweigh the kernel values of the neighbors,
    which turns a GNO into an Attentional GNO (AGNO).
    Padding edges do not take part in the softmax and get the weight zero.

    Args:
        key: PRNG key for the initialization of the projections.
            Not used for `"cosine"`.
        in_channels: Number of channels of the (embedded) coordinates.
        attention_type: Type of attention, `"cosine"` or `"dot_product"`.
            Default is `"cosine"`.
        attention_dim: Dimension $d_a$ of the query and key projections of
            the `"dot_product"` attention. Default is 64.

    Raises:
        ValueError: If `in_channels` or `attention_dim` is not a positive integer.
        ValueError: If `attention_type` is invalid.

    ??? info "Internal Attributes"
        * **query_proj** (`PointwiseMLP | None`): Query projection of the `"dot_product"` attention.
        * **key_proj** (`PointwiseMLP | None`): Key projection of the `"dot_product"` attention.
        * **in_channels** (`int`): Number of channels of the coordinates.
        * **attention_type** (`Literal["cosine", "dot_product"]`): Type of attention.
        * **attention_dim** (`int`): Dimension of the projections.

    !!! warning "Cosine attention depends on the origin"
        The cosine similarity compares the *position vectors* of the points, not
        the displacement between them. The weights thus depend on where the
        coordinate origin is, e.g. use coordinates centered at zero.

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

    query_proj: PointwiseMLP | None
    key_proj: PointwiseMLP | None
    in_channels: int = eqx.field(static=True)
    attention_type: Literal["cosine", "dot_product"] = eqx.field(static=True)
    attention_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        attention_type: Literal["cosine", "dot_product"] = "cosine",
        attention_dim: int = 64,
    ) -> None:
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(attention_dim, int) or attention_dim <= 0:
            raise ValueError("attention_dim must be a positive integer.")
        if attention_type not in ["cosine", "dot_product"]:
            raise ValueError(
                "attention_type must be one of ['cosine', 'dot_product'], "
                f"but got '{attention_type}'."
            )

        self.in_channels = in_channels
        self.attention_type = attention_type
        self.attention_dim = attention_dim

        if attention_type == "dot_product":
            k_query, k_key = jr.split(key)
            self.query_proj = PointwiseMLP(k_query, (in_channels, attention_dim))
            self.key_proj = PointwiseMLP(k_key, (in_channels, attention_dim))
        else:
            self.query_proj = None
            self.key_proj = None

    def __call__(
        self,
        x: Float[Array, "c e"],
        y: Float[Array, "c e"],
        neighbors: Neighbors,
    ) -> Float[Array, "e"]:
        """Computes the attention weights of the edges.

        Args:
            x: The (embedded) coordinates of the query point of each edge,
                shaped (in_channels, e).
            y: The (embedded) coordinates of the neighbor of each edge,
                shaped (in_channels, e).
            neighbors: The edges, which determine the neighborhoods.

        Returns:
            The attention weights per edge, shaped (e,). They sum to one over the
            edges of each query with at least one neighbor, and padding is zero.

        Raises:
            ValueError: If the number of coordinate channels doesn't match `in_channels`.
        """
        if x.shape[0] != self.in_channels or y.shape[0] != self.in_channels:
            raise ValueError(
                f"Expected coordinates with {self.in_channels} channels along axis 0, "
                f"but got shapes {x.shape} and {y.shape}."
            )

        if self.attention_type == "cosine":
            eps = 1e-6

            def normalize(a: Float[Array, "c e"]) -> Float[Array, "c e"]:
                # rsqrt(max(...)) instead of a division by the norm to have
                # finite gradients for points at the origin
                sq_norm = jnp.sum(a**2, axis=0, keepdims=True)
                return a * jax.lax.rsqrt(jnp.maximum(sq_norm, eps**2))

            logits = jnp.sum(normalize(x) * normalize(y), axis=0)
        else:
            q = self.query_proj(x)
            k = self.key_proj(y)
            logits = jnp.sum(q * k, axis=0) / jnp.sqrt(float(self.attention_dim))

        # imported here: `neojax.utils` imports the models, which import this module
        from neojax.utils.segment_ops import segment_softmax

        return segment_softmax(
            logits,
            neighbors.query_index,
            neighbors.num_queries,
            indices_are_sorted=True,
        )
