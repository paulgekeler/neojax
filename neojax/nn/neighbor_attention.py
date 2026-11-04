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
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP


class NeighborAttention(eqx.Module):
    r"""Attention weights over the neighbors of each query point.

    Computes for a query point $x$ and its neighbors $y_j$ a normalized weight
    $\alpha(x, y_j)$ that sums to one over the valid neighbors of $x$:

    - `"cosine"`: $\alpha(x, y_j) = \operatorname{softmax}_j \, \left(\cos(x, y_j)\right)$,
        the cosine similarity of the (embedded) coordinates,
    - `"dot_product"`: $\alpha(x, y_j) = \operatorname{softmax}_j
        \, \left(\langle W_q x, W_k y_j \rangle / \sqrt{d_a}\right)$ with learnable
        projections $W_q$ and $W_k$ to the attention dimension $d_a$.

    Used by `IntegralTransform` to reweigh the kernel values of the neighbors,
    which turns a GNO into an Attentional GNO (AGNO).
    Padded neighbors do not take part in the softmax and get the weight zero.
    A point without neighbors gets all zero weights.

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

    def _masked_softmax(
        self, logits: Float[Array, "c k"], mask: Bool[Array, "c k"]
    ) -> Float[Array, "c k"]:
        """Softmax over the last axis that ignores masked entries.

        Rows without any valid entry are all zero.

        Args:
            logits: The pre-softmax logits shaped per `c` query points
                and `k` neighbors.
            mask: `True` for valid neighbors, `False` for padding.

        Returns:
            Softmax output probabilities.
        """
        in_dtype = logits.dtype
        compute_dtype = jnp.promote_types(in_dtype, jnp.float32)
        logits = jnp.astype(logits, compute_dtype)
        logits = jnp.where(mask, logits, -jnp.inf)
        row_max = jnp.max(logits, axis=-1, keepdims=True)
        row_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(row_max), row_max, 0.0))
        # shift masked entries to zero before exponentiating: finite gradients
        exp = jnp.where(mask, jnp.exp(jnp.where(mask, logits, 0.0) - row_max), 0.0)
        total = jnp.sum(exp, axis=-1, keepdims=True)
        return (exp / jnp.where(total > 0, total, 1.0)).astype(in_dtype)

    def __call__(
        self,
        x: Float[Array, "e c k"],
        y: Float[Array, "e c k"],
        mask: Bool[Array, "c k"],
    ) -> Float[Array, "c k"]:
        """Computes the attention weights.

        Args:
            x: The (embedded) coordinates of the `c` query points, repeated
                along the `k` neighbors, shaped (in_channels, c, k).
            y: The (embedded) coordinates of the `k` neighbors of each
                query point, shaped (in_channels, c, k).
            mask: `True` for the valid neighbors, `False` for the padding.

        Returns:
            The attention weights, shaped (c, k). They sum to one over the
            neighbors of each query with at least one valid neighbor.

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

            def normalize(a: Float[Array, "e c k"]) -> Float[Array, "e c k"]:
                # rsqrt(max(...)) instead of a division by the norm to have
                # finite gradients for points at the origin
                sq_norm = jnp.sum(a**2, axis=0, keepdims=True)
                return a * jax.lax.rsqrt(jnp.maximum(sq_norm, eps**2))

            logits = jnp.sum(normalize(x) * normalize(y), axis=0)
        else:
            q = self.query_proj(x)
            k = self.key_proj(y)
            logits = jnp.sum(q * k, axis=0) / jnp.sqrt(float(self.attention_dim))

        return self._masked_softmax(logits, mask)
