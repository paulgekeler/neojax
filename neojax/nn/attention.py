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

"""Implementation of multi-head self-attention with grouped query heads."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import RotaryEmbedding


class MultiHeadAttention(eqx.Module):
    r"""Multi-head self-attention over a sequence of tokens.

    Computes $\operatorname{softmax}(Q K^\top / \sqrt{d_h}) V$ for each head
    of $h$ query heads with head dimension $d_h$ = `channels // n_heads`, followed by an
    output projection. All projections are linear layers without bias.

    With `n_kv_heads < n_heads` the layer is a grouped-query attention: groups of
    `n_heads // n_kv_heads` consecutive query heads share a key and value head.
    `n_kv_heads = 1` is multi-query attention. Optionally the queries and keys are
    rotated by a `RotaryEmbedding` of the token index.

    Args:
        key: PRNG key for parameter initialization.
        channels: Number of channels of the tokens (input and output).
            Must be divisible by `n_heads`.
        n_heads: Number of query heads. Default is 8.
        n_kv_heads: Number of key and value heads. Must divide `n_heads`.
            Default is `None`, which is `n_heads` (standard multi-head attention).
        dropout: Dropout probability of the attention weights. Default is 0.0.
        use_rope: Whether to apply a `RotaryEmbedding` to the queries and keys.
            Requires an even head dimension. Default is `False`.
        rope_theta: Base of the rotary frequencies. Default is 10000.

    Raises:
        ValueError: If `channels`, `n_heads` or `n_kv_heads` is not a positive integer.
        ValueError: If `channels` is not divisible by `n_heads` or `n_heads` is not
            divisible by `n_kv_heads`.
        ValueError: If `dropout` is not in [0, 1).
        ValueError: If `use_rope` is set and the head dimension is odd.

    ??? info "Internal Attributes"
        * **q_proj** (`PointwiseMLP`): Query projection.
        * **k_proj** (`PointwiseMLP`): Key projection.
        * **v_proj** (`PointwiseMLP`): Value projection.
        * **o_proj** (`PointwiseMLP`): Output projection.
        * **rope** (`RotaryEmbedding | None`): Rotary embedding of queries and keys.
        * **channels** (`int`): Number of channels of the tokens.
        * **n_heads** (`int`): Number of query heads.
        * **n_kv_heads** (`int`): Number of key and value heads.
        * **head_dim** (`int`): Dimension of a head.
        * **dropout** (`float`): Dropout probability of the attention weights.

    ??? cite

        [GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

        ```bibtex
        @inproceedings{ainslie2023gqa,
            title={GQA: Training Generalized Multi-Query Transformer Models
                from Multi-Head Checkpoints},
            author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel
                and Zemlyanskiy, Yury and Lebr{\'o}n, Federico and Sanghai, Sumit},
            booktitle={Proceedings of the 2023 Conference on Empirical Methods
                in Natural Language Processing},
            year={2023}
        }
        ```
    """

    q_proj: PointwiseMLP
    k_proj: PointwiseMLP
    v_proj: PointwiseMLP
    o_proj: PointwiseMLP
    rope: RotaryEmbedding | None
    channels: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        channels: int,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        if n_kv_heads is None:
            n_kv_heads = n_heads
        for name, value in [
            ("channels", channels),
            ("n_heads", n_heads),
            ("n_kv_heads", n_kv_heads),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if channels % n_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by n_heads ({n_heads})."
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})."
            )
        if not isinstance(dropout, int | float) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        head_dim = channels // n_heads
        if use_rope and head_dim % 2 != 0:
            raise ValueError(
                f"use_rope requires an even head dimension, but got {head_dim}."
            )

        self.channels = channels
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dropout = float(dropout)

        k_q, k_k, k_v, k_o = jr.split(key, 4)
        kv_channels = n_kv_heads * head_dim
        self.q_proj = PointwiseMLP(k_q, (channels, channels), use_bias=False)
        self.k_proj = PointwiseMLP(k_k, (channels, kv_channels), use_bias=False)
        self.v_proj = PointwiseMLP(k_v, (channels, kv_channels), use_bias=False)
        self.o_proj = PointwiseMLP(k_o, (channels, channels), use_bias=False)
        self.rope = RotaryEmbedding(head_dim, rope_theta) if use_rope else None

    def _attention_with_dropout(
        self,
        q: Float[Array, "h d t"],
        k: Float[Array, "g d t"],
        v: Float[Array, "g d t"],
        dropout: float,
        key: PRNGKeyArray,
    ) -> Float[Array, "h d t"]:
        """Explicit attention with dropout on the attention weights."""
        n_repeat = q.shape[0] // k.shape[0]
        k = jnp.repeat(k, n_repeat, axis=0)
        v = jnp.repeat(v, n_repeat, axis=0)
        scores = jnp.einsum("hdq,hdk->hqk", q, k) * q.shape[1] ** -0.5
        weights = jax.nn.softmax(scores, axis=-1)
        keep = jr.bernoulli(key, 1.0 - dropout, weights.shape)
        weights = jnp.where(keep, weights / (1.0 - dropout), 0.0)
        return jnp.einsum("hqk,hdk->hdq", weights, v)

    def __call__(
        self,
        x: Float[Array, "c t"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "c t"]:
        """Applies self-attention.

        Args:
            x: Tokens shaped (channels, num_tokens).
            key: PRNG key for the dropout of the attention weights.
            inference: If `True`, dropout is disabled.

        Returns:
            The attended tokens, shaped (channels, num_tokens).

        Raises:
            ValueError: If dropout is active but `key` is `None`.
        """
        t = x.shape[-1]
        q = self.q_proj(x).reshape(self.n_heads, self.head_dim, t)
        k = self.k_proj(x).reshape(self.n_kv_heads, self.head_dim, t)
        v = self.v_proj(x).reshape(self.n_kv_heads, self.head_dim, t)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError(
                    "A PRNG key must be passed to __call__ when dropout > 0 "
                    "and inference = False."
                )
            out = self._attention_with_dropout(q, k, v, self.dropout, key)
        else:
            # (t, heads, head_dim) with a batch axis for jax.nn.dot_product_attention
            out = jax.nn.dot_product_attention(
                q.transpose(2, 0, 1)[None],
                k.transpose(2, 0, 1)[None],
                v.transpose(2, 0, 1)[None],
            )[0]
            out = out.transpose(1, 2, 0)  # (heads, head_dim, t)

        return self.o_proj(out.reshape(self.channels, t))
