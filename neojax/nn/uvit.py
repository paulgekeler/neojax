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

"""Implementation of the U-shaped vision transformer."""

import equinox as eqx
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.transformer_block import TransformerBlock


class UViT(eqx.Module):
    r"""U-shaped vision transformer (U-ViT) over a sequence of tokens.

    A stack of `TransformerBlock`s, whose first half is the encoder, followed by the
    decoder as second half. With `n_layers` odd, an additional block sits in the middle.
    The input of every decoder block is merged with the output of the encoder block
    at the mirrored depth by a linear layer on the concatenated channels
    (long-range skip connections).
    Linear layers map the input to `hidden_channels` and the result to `out_channels`
    if the widths differ.

    The module acts on tokens and is agnostic of how they were formed.
    `GAOT` feeds it patches of the latent grid, see `patchify`.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of channels of the input tokens.
        out_channels: Number of channels of the output tokens.
        hidden_channels: Number of channels of the transformer blocks.
            Default is 256.
        n_layers: Number of transformer blocks. Default is 3.
        n_heads: Number of attention heads, see `MultiHeadAttention`. Default is 8.
        n_kv_heads: Number of key and value heads. Default is `None`, which
            is `n_heads`.
        ffn_multiplier: Ratio of the hidden channels of the feed-forward
            networks to `hidden_channels`. Default is 4.
        use_attn_norm: Whether to normalize the inputs of the attentions.
            Default is `True`.
        use_ffn_norm: Whether to normalize the inputs of the feed-forward networks.
            Default is `True`.
        norm_eps: Epsilon of the normalizations. Default is 1e-6.
        use_long_range_skip: Whether the decoder blocks merge the outputs of
            the encoder blocks. Default is `True`.
        dropout: Dropout probability of the attention weights. Default is 0.0.
        use_rope: Whether to use rotary position embeddings, see
            `MultiHeadAttention`. Without it, the positions have to be encoded in the
            tokens. Default is `False`.

    Raises:
        ValueError: If `in_channels`, `out_channels`, `hidden_channels`
            or `n_layers` is not a positive integer.
        ValueError: For the invalid arguments of `TransformerBlock`.

    ??? info "Internal Attributes"
        * **input_proj** (`PointwiseMLP | None`): Maps to `hidden_channels` if `in_channels` differs.
        * **output_proj** (`PointwiseMLP | None`): Maps to `out_channels` if `hidden_channels` differs.
        * **encoder_blocks** (`tuple[TransformerBlock, ...]`): The first `n_layers // 2` blocks.
        * **middle_block** (`TransformerBlock | None`): The middle block if `n_layers` is odd.
        * **decoder_blocks** (`tuple[TransformerBlock, ...]`): The last `n_layers // 2` blocks.
        * **use_long_range_skip** (`bool`): Whether the encoder outputs are merged into the decoder.

    ??? cite

        [All are Worth Words: A ViT Backbone for Diffusion Models](
        https://arxiv.org/abs/2209.12152)

        ```bibtex
        @inproceedings{bao2023all,
            title={All are Worth Words: A ViT Backbone for Diffusion Models},
            author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and
                Li, Chongxuan and Su, Hang and Zhu, Jun},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer
                Vision and Pattern Recognition},
            year={2023}
        }
        ```

        [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](
        https://arxiv.org/abs/2010.11929)

        ```bibtex
        @inproceedings{dosovitskiy2021image,
            title={An Image is Worth 16x16 Words: Transformers for Image
                Recognition at Scale},
            author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander
                and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas
                and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg
                and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
            booktitle={International Conference on Learning Representations},
            year={2021}
        }
        ```
    """

    input_proj: PointwiseMLP | None
    output_proj: PointwiseMLP | None
    encoder_blocks: tuple[TransformerBlock, ...]
    middle_block: TransformerBlock | None
    decoder_blocks: tuple[TransformerBlock, ...]
    use_long_range_skip: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        ffn_multiplier: int | float = 4,
        use_attn_norm: bool = True,
        use_ffn_norm: bool = True,
        norm_eps: float = 1e-6,
        use_long_range_skip: bool = True,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        for name, value in [
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("hidden_channels", hidden_channels),
            ("n_layers", n_layers),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        self.use_long_range_skip = use_long_range_skip
        k_in, k_out, k_blocks = jr.split(key, 3)
        self.input_proj = (
            PointwiseMLP(k_in, (in_channels, hidden_channels))
            if in_channels != hidden_channels
            else None
        )
        self.output_proj = (
            PointwiseMLP(k_out, (hidden_channels, out_channels))
            if hidden_channels != out_channels
            else None
        )

        n_half = n_layers // 2
        block_keys = jr.split(k_blocks, n_layers)

        def make_block(block_key: PRNGKeyArray, skip: bool) -> TransformerBlock:
            return TransformerBlock(
                block_key,
                hidden_channels,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ffn_multiplier=ffn_multiplier,
                use_attn_norm=use_attn_norm,
                use_ffn_norm=use_ffn_norm,
                norm_eps=norm_eps,
                dropout=dropout,
                use_rope=use_rope,
                skip_connection=skip,
            )

        self.encoder_blocks = tuple(make_block(k, False) for k in block_keys[:n_half])
        self.middle_block = (
            make_block(block_keys[n_half], False) if n_layers % 2 == 1 else None
        )
        self.decoder_blocks = tuple(
            make_block(k, use_long_range_skip) for k in block_keys[n_layers - n_half :]
        )

    def __call__(
        self,
        x: Float[Array, "in_c t"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Float[Array, "out_c t"]:
        """Applies the transformer.

        Args:
            x: Tokens shaped (in_channels, num_tokens).
            key: PRNG key for the dropout of the attention weights.
            inference: If `True`, dropout is disabled.

        Returns:
            The output tokens, shaped (out_channels, num_tokens).
        """
        n_blocks = len(self.encoder_blocks) + len(self.decoder_blocks)
        n_blocks += self.middle_block is not None
        keys = (
            jr.split(key, n_blocks)
            if key is not None and not inference
            else [None] * n_blocks
        )
        keys = list(keys)

        if self.input_proj is not None:
            x = self.input_proj(x)

        skips = []
        for block in self.encoder_blocks:
            x = block(x, key=keys.pop(0), inference=inference)
            skips.append(x)
        if self.middle_block is not None:
            x = self.middle_block(x, key=keys.pop(0), inference=inference)
        for block in self.decoder_blocks:
            skip = skips.pop() if self.use_long_range_skip else None
            x = block(x, skip, key=keys.pop(0), inference=inference)

        if self.output_proj is not None:
            x = self.output_proj(x)
        return x
