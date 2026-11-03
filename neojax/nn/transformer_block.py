"""Implementation of a transformer block with SwiGLU feed-forward network."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.attention import MultiHeadAttention
from neojax.nn.normalization import RMSNorm
from neojax.nn.pointwise_mlp import PointwiseMLP


class SwiGLU(eqx.Module):
    r"""Feed-forward network with a SiLU-gated linear unit.

    Computes $W_2 \left(\operatorname{SiLU}(W_1 x) \odot W_3 x\right)$ with
    linear layers without bias, which expands the channels to `hidden_channels`
    and contracts them back.

    Args:
        key: PRNG key for parameter initialization.
        channels: Number of channels of the input and output.
        hidden_channels: Number of channels of the expanded representation.

    Raises:
        ValueError: If `channels` or `hidden_channels` is not a positive integer.

    ??? info "Internal Attributes"
        * **gate** (`PointwiseMLP`): The layer $W_1$ whose output is passed through the SiLU.
        * **up** (`PointwiseMLP`): The layer $W_3$ producing the linear branch.
        * **down** (`PointwiseMLP`): The layer $W_2$ contracting to `channels`.

    ??? cite

        [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

        ```bibtex
        @article{shazeer2020glu,
            title={GLU variants improve transformer},
            author={Shazeer, Noam},
            journal={arXiv preprint arXiv:2002.05202},
            year={2020}
        }
        ```
    """

    gate: PointwiseMLP
    up: PointwiseMLP
    down: PointwiseMLP

    def __init__(self, key: PRNGKeyArray, channels: int, hidden_channels: int) -> None:
        for name, value in [
            ("channels", channels),
            ("hidden_channels", hidden_channels),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        k_gate, k_up, k_down = jr.split(key, 3)
        self.gate = PointwiseMLP(k_gate, (channels, hidden_channels), use_bias=False)
        self.up = PointwiseMLP(k_up, (channels, hidden_channels), use_bias=False)
        self.down = PointwiseMLP(k_down, (hidden_channels, channels), use_bias=False)

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Applies the feed-forward network.

        Args:
            x: Input array shaped (channels, d1, ..., dN).

        Returns:
            Output array of the same shape.
        """
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class TransformerBlock(eqx.Module):
    r"""Transformer block with `MultiHeadAttention` and a `SwiGLU` feed-forward network.

    With the RMS normalizations $N_1$ and $N_2$ it computes
    $$
    h = x + \operatorname{Attention}(N_1(x)), \qquad
    y = N_2(h) + \operatorname{FFN}(N_2(h)).
    $$
    Both normalizations can be disabled. Optionally the block first merges a
    `skip` input (e.g. the output of an earlier block) into `x` with a linear layer
    on the concatenated channels, as used by the decoder blocks of `UViT`.

    Args:
        key: PRNG key for parameter initialization.
        channels: Number of channels of the tokens.
        n_heads: Number of attention heads. Default is 8.
        n_kv_heads: Number of key and value heads, see `MultiHeadAttention`.
            Default is `None`, which is `n_heads`.
        ffn_multiplier: Ratio of the hidden channels of the feed-forward network
            to `channels`. Default is 4.
        use_attn_norm: Whether to normalize the input of the attention.
            Default is `True`.
        use_ffn_norm: Whether to normalize the input of the feed-forward network.
            Default is `True`.
        norm_eps: Epsilon of the `RMSNorm`s. Default is 1e-6.
        dropout: Dropout probability of the attention weights. Default is 0.0.
        use_rope: Whether to use rotary position embeddings in the attention.
            Default is `False`.
        skip_connection: Whether the block merges a `skip` input.
            Default is `False`.

    Raises:
        ValueError: If `ffn_multiplier` is not positive.
        ValueError: For the invalid arguments of `MultiHeadAttention`.

    ??? info "Internal Attributes"
        * **attn_norm** (`RMSNorm | None`): Normalization before the attention.
        * **attn** (`MultiHeadAttention`): The self-attention.
        * **ffn_norm** (`RMSNorm | None`): Normalization before the feed-forward network.
        * **ffn** (`SwiGLU`): The feed-forward network.
        * **skip_proj** (`PointwiseMLP | None`): Linear layer merging the `skip` input.

    !!! info "Position of the feed-forward residual"
        As in the reference GAOT implementation, the normalization $N_2$ is applied
        to the residual stream and the residual connection of the feed-forward
        network is taken *after* it. This differs from the usual pre-norm transformer
        block, where it would be $y = h + \operatorname{FFN}(N_2(h))$.

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

    attn_norm: RMSNorm | None
    attn: MultiHeadAttention
    ffn_norm: RMSNorm | None
    ffn: SwiGLU
    skip_proj: PointwiseMLP | None

    def __init__(
        self,
        key: PRNGKeyArray,
        channels: int,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        ffn_multiplier: int | float = 4,
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        use_rope: bool = False,
        skip_connection: bool = False,
    ) -> None:
        if not ffn_multiplier > 0:
            raise ValueError("ffn_multiplier must be positive.")
        k_attn, k_ffn, k_skip = jr.split(key, 3)
        self.attn_norm = RMSNorm(channels, norm_eps) if use_attn_norm else None
        self.attn = MultiHeadAttention(
            k_attn, channels, n_heads, n_kv_heads, dropout, use_rope
        )
        self.ffn_norm = RMSNorm(channels, norm_eps) if use_ffn_norm else None
        self.ffn = SwiGLU(k_ffn, channels, max(int(channels * ffn_multiplier), 1))
        self.skip_proj = (
            PointwiseMLP(k_skip, (2 * channels, channels)) if skip_connection else None
        )

    def __call__(
        self,
        x: Float[Array, "c t"],
        skip: Float[Array, "c t"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "c t"]:
        """Applies the block.

        Args:
            x: Tokens shaped (channels, num_tokens).
            skip: Optional tokens to merge into `x` first. Only used by blocks
                created with `skip_connection=True`. Default is `None`.
            key: PRNG key for the dropout of the attention weights.
            inference: If `True`, dropout is disabled.

        Returns:
            The output tokens, shaped (channels, num_tokens).
        """
        if self.skip_proj is not None and skip is not None:
            x = self.skip_proj(jnp.concatenate([x, skip], axis=0))

        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, key=key, inference=inference)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        return h + self.ffn(h)
