"""Implementation of a n-dimensional Convolutional Neural Operator (CNO)."""

from typing import Literal, final

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray
from typing_extensions import override

from neojax.models.baseno import BaseNO
from neojax.models.utils import apply_layer
from neojax.nn.cno_blocks import CNOConvBlock, CNOLiftProjectBlock, CNOResidualBlock
from neojax.nn.positional_embedding import GridEmbeddingNd


@final
class CNO(BaseNO):
    r"""Convolutional Neural Operator (CNO).

    A U-Net-style convolutional architecture, in which every convolution is
    paired with an alias-free activation $\Sigma_{s, s'}$ that
    upsamples, applies the leaky ReLU and downsamples again with
    low-pass filters, such that the network maps between function
    spaces instead of between fixed grids. Implements the classic
    variant of the reference (`CNO2d_classic`) for any number of spatial dimensions.

    The input is lifted to a latent space and passed through an encoder of
    `n_layers` levels. Each level consists of `n_res` residual blocks, whose
    output is kept for the skip connection, and a downsampling block (D-block).
    The bottleneck applies `n_res_neck` residual blocks. The decoder has
    `n_layers` levels, each consisting of an upsampling block (U-block)
    and optionally an invariant block (I-block) applied after the skip
    connection of the corresponding encoder level was concatenated
    (along the channel dimension).
    The channels double with every encoder level, starting
    at `channel_multiplier`. The output lives on the same grid as the input.

    All spatial sizes of the input must be divisible by `2 ** n_layers`.
    The model is otherwise resolution-agnostic, since the filters only depend on
    the ratio of the input and output resolution of the blocks.

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels
            (e.g. coordinates + initial conditions).
        out_channels: Number of output channels (e.g. solution field).
        ndim: Number of spatial dimensions of the input (e.g. 1, 2 or 3).
        n_layers: Number of encoder (and decoder) levels. Must be at least 1.
        n_res: Number of residual blocks per encoder level. Default is 1.
        n_res_neck: Number of residual blocks in the bottleneck. Default is 6.
        channel_multiplier: Number of channels after the first encoder level.
            The lifted input has `channel_multiplier // 2` channels and
            each further level doubles the channel count. Default is 32.
        latent_lift_proj_channels: Number of latent channels inside the lifting
            and projection blocks. Default is 64.
        conv_kernel: Odd size of the convolution kernels along each dimension.
            Default is 3.
        cutoff_den: Denominator of the cutoff frequency of the filters, see `CNOConvBlock`.
            Default is `2.0001`.
        half_width_mult: Multiplier of the half-width of the transition band
            of the filters, see `CNOConvBlock`. Default is `0.8`.
        filter_size: Number of filter taps per phase of the resampling, see `CNOConvBlock`.
            Default is 6.
        add_inv: Whether to add an invariant block after the concatenation
            of the skip connection in each decoder level. Default is True.
        positional_embedding: Positional embedding to apply
            to last channels of raw input before passing through the CNO.
            Default is None. Pass `"grid"` for a regular `GridEmbeddingNd`.
        batch_norm: Whether to apply batch normalization after the convolutions of
            the blocks (but not in the lifting and projection), as in the reference.
            If `False`, there is no normalization. Default is `True`. Batch normalization
            makes the model stateful, see below.
        batch_axis_name: Name of the `jax.vmap` axis the batch statistics are
            computed over if `batch_norm=True`. Default is `"batch"`.
        batch_norm_momentum: Weight of the running statistics (not of the batch
            statistics) in their update if `batch_norm=True`.
            Default is `0.9`, which corresponds to PyTorch's `momentum=0.1`.
        pad_mode: Padding mode of the convolutions and activations. Use `"wrap"`
            for a periodic domain, otherwise `"reflect"` or `"constant"`
            (zero padding, as in the reference). Default is `"constant"`.
        negative_slope: Negative slope of the leaky ReLU. Default is `0.2`.

    Raises:
        ValueError: If `n_layers` is smaller than 1.
        ValueError: If an invalid positional embedding is passed.

    !!! info "Stateful models (batch normalization)"
        With `batch_norm=True` (the default, as in the reference), the model has running
        statistics and is *stateful* (`is_stateful()` returns `True`), following the
        convention of [Equinox](https://docs.kidger.site/equinox/api/nn/stateful/):

        - The model and its initial state are created with
            `model, state = eqx.nn.make_with_state(CNO)(...)`.
        - The model is called as `out, state = model(x, state)`, and has to be called inside of a
            `jax.vmap` over the batch with `axis_name=batch_axis_name`, in which the
            state is not mapped: `in_axes=(0, None), out_axes=(0, None)`. The batch
            statistics are computed over this axis (see example).
        - `inference=True` uses the running statistics instead of the batch
            statistics and leaves the state unchanged.

        `CNO`'s with `batch_norm=False` are stateless and called as `out = model(x)`.

    !!! info "Differences to the reference"
        - The batch normalization uses the `"batch"` mode of `eqx.nn.BatchNorm`: Like
            PyTorch, the batch statistics are used for training. For inference, an
            (initialization-corrected) exponential moving average of the batch statistics is used.
        - The reference supports different input and output resolutions
            (`out_size`) and enlarging the input to a multiple of `2 ** n_layers`
            (`expand_input`). Neither are supported. The output has the resolution of the input.
        - The reference builds an unused invariant block in the last
            decoder level, which is omitted here.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **positional_embedding** (`GridEmbeddingNd | None`): Positional embedding.
        * **lift** (`CNOLiftProjectBlock`): Maps inputs to the latent space.
        * **encoder** (`tuple[CNOConvBlock, ...]`): Downsampling blocks of the encoder.
        * **res_nets** (`tuple[tuple[CNOResidualBlock, ...], ...]`): Residual blocks of each encoder level.
        * **res_net_neck** (`tuple[CNOResidualBlock, ...]`): Residual blocks of the bottleneck.
        * **ed_expansion** (`tuple[CNOConvBlock, ...]`): Invariant blocks applied to the
            features passed from the encoder to the decoder.
        * **decoder_inv** (`tuple[CNOConvBlock, ...] | None`): Invariant blocks of the decoder
            or `None`.
        * **decoder** (`tuple[CNOConvBlock, ...]`): Upsampling blocks of the decoder.
        * **project** (`CNOLiftProjectBlock`): Maps the latent features to the output channels.
        * **n_layers** (`int`): Number of encoder (and decoder) levels.
        * **ndim** (`int`): Number of spatial dimensions.

    Examples:
        ```python
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.models import CNO

        key = jr.key(0)
        model = CNO(
            key=key,
            in_channels=3,
            out_channels=1,
            ndim=2,
            n_layers=3,
            n_res=1,
            n_res_neck=2,
            channel_multiplier=8,
            batch_norm=False,
        )
        x = jnp.ones((3, 32, 32))
        # out.shape = (1, 32, 32)
        out = model(x)

        # Batch normalization (default): stateful model
        model, state = eqx.nn.make_with_state(CNO)(
            key=key,
            in_channels=3,
            out_channels=1,
            ndim=2,
            n_layers=3,
            n_res=1,
            n_res_neck=2,
            channel_multiplier=8,
            batch_norm=True,
        )
        xs = jnp.ones((4, 3, 32, 32))
        batched_model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        # outs.shape = (4, 1, 32, 32)
        outs, state = batched_model(xs, state)
        ```

    ??? cite
        [Convolutional Neural Operators for robust
        and accurate learning of PDEs](https://proceedings.neurips.cc/paper_files/
        paper/2023/file/f3c1951b34f7f55ffaecada7fde6bd5a-Paper-Conference.pdf)

        ```bibtex
        @inproceedings{NEURIPS2023_f3c1951b,
            author = {Raonic, Bogdan and Molinaro,
            Roberto and De Ryck, Tim and Rohner, Tobias and Bartolucci,
            Francesca and Alaifari, Rima and Mishra,
            Siddhartha and de B\'{e}zenac, Emmanuel},
            booktitle = {Advances in Neural Information Processing Systems},
            doi = {10.52202/075280-3376},
            editor = {A. Oh and T. Naumann and A. Globerson
            and K. Saenko and M. Hardt and S. Levine},
            pages = {77187--77200},
            title = {Convolutional Neural Operators for
            robust and accurate learning of PDEs},
            volume = {36},
            year = {2023}
        }
        ```
    """

    positional_embedding: GridEmbeddingNd | None
    lift: CNOLiftProjectBlock
    encoder: tuple[CNOConvBlock, ...]
    res_nets: tuple[tuple[CNOResidualBlock, ...], ...]
    res_net_neck: tuple[CNOResidualBlock, ...]
    ed_expansion: tuple[CNOConvBlock, ...]
    decoder_inv: tuple[CNOConvBlock, ...] | None
    decoder: tuple[CNOConvBlock, ...]
    project: CNOLiftProjectBlock
    n_layers: int = eqx.field(static=True)
    ndim: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        ndim: int,
        n_layers: int,
        n_res: int = 1,
        n_res_neck: int = 6,
        channel_multiplier: int = 32,
        latent_lift_proj_channels: int = 64,
        conv_kernel: int = 3,
        cutoff_den: float = 2.0001,
        half_width_mult: float = 0.8,
        filter_size: int = 6,
        add_inv: bool = True,
        positional_embedding: GridEmbeddingNd | Literal["grid"] | None = None,
        batch_norm: bool = True,
        batch_axis_name: str = "batch",
        batch_norm_momentum: float = 0.9,
        pad_mode: Literal["wrap", "reflect", "constant"] = "constant",
        negative_slope: float = 0.2,
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"`n_layers` must be at least 1, got {n_layers}.")
        self.n_layers = n_layers
        self.ndim = ndim

        # Positional embedding
        if positional_embedding is None:
            self.positional_embedding = None
        elif positional_embedding == "grid":
            self.positional_embedding = GridEmbeddingNd(in_channels, ndim)
        elif isinstance(positional_embedding, GridEmbeddingNd):
            self.positional_embedding = positional_embedding
        else:
            raise ValueError(f"Positional embedding {positional_embedding} invalid.")
        lift_in_channels = in_channels
        if self.positional_embedding is not None:
            lift_in_channels += self.positional_embedding.n_extra_channels

        # Evolution of the number of channels
        encoder_features = [channel_multiplier // 2] + [
            2**i * channel_multiplier for i in range(n_layers)
        ]
        decoder_features_in = encoder_features[1:][::-1]
        decoder_features_out = encoder_features[:-1][::-1]
        # Decoder levels beyond the first receive the concatenated skip features
        for i in range(1, n_layers):
            decoder_features_in[i] *= 2

        block_kwargs = {
            "ndim": ndim,
            "conv_kernel": conv_kernel,
            "cutoff_den": cutoff_den,
            "half_width_mult": half_width_mult,
            "filter_size": filter_size,
            "pad_mode": pad_mode,
            "negative_slope": negative_slope,
        }
        norm_kwargs = {
            "batch_norm": batch_norm,
            "batch_axis_name": batch_axis_name,
            "batch_norm_momentum": batch_norm_momentum,
        }

        (
            lift_key,
            proj_key,
            enc_key,
            ed_key,
            inv_key,
            dec_key,
            res_key,
            neck_key,
        ) = jr.split(key, 8)

        self.lift = CNOLiftProjectBlock(
            key=lift_key,
            in_channels=lift_in_channels,
            out_channels=encoder_features[0],
            latent_channels=latent_lift_proj_channels,
            **block_kwargs,
        )
        self.project = CNOLiftProjectBlock(
            key=proj_key,
            in_channels=encoder_features[0] + decoder_features_out[-1],
            out_channels=out_channels,
            latent_channels=latent_lift_proj_channels,
            **block_kwargs,
        )

        enc_keys = jr.split(enc_key, n_layers)
        self.encoder = tuple(
            CNOConvBlock(
                key=enc_keys[i],
                in_channels=encoder_features[i],
                out_channels=encoder_features[i + 1],
                block_type="downsample",
                **norm_kwargs,
                **block_kwargs,
            )
            for i in range(n_layers)
        )

        ed_keys = jr.split(ed_key, n_layers + 1)
        self.ed_expansion = tuple(
            CNOConvBlock(
                key=ed_keys[i],
                in_channels=encoder_features[i],
                out_channels=encoder_features[i],
                block_type="invariant",
                **norm_kwargs,
                **block_kwargs,
            )
            for i in range(n_layers + 1)
        )

        if add_inv:
            inv_keys = jr.split(inv_key, n_layers)
            self.decoder_inv = tuple(
                CNOConvBlock(
                    key=inv_keys[i],
                    in_channels=decoder_features_in[i],
                    out_channels=decoder_features_in[i],
                    block_type="invariant",
                    **norm_kwargs,
                    **block_kwargs,
                )
                for i in range(n_layers)
            )
        else:
            self.decoder_inv = None

        dec_keys = jr.split(dec_key, n_layers)
        self.decoder = tuple(
            CNOConvBlock(
                key=dec_keys[i],
                in_channels=decoder_features_in[i],
                out_channels=decoder_features_out[i],
                block_type="upsample",
                **norm_kwargs,
                **block_kwargs,
            )
            for i in range(n_layers)
        )

        res_keys = jr.split(res_key, (n_layers, n_res))
        self.res_nets = tuple(
            tuple(
                CNOResidualBlock(
                    key=res_keys[level, j],
                    channels=encoder_features[level],
                    **norm_kwargs,
                    **block_kwargs,
                )
                for j in range(n_res)
            )
            for level in range(n_layers)
        )
        neck_keys = jr.split(neck_key, n_res_neck)
        self.res_net_neck = tuple(
            CNOResidualBlock(
                key=neck_keys[j],
                channels=encoder_features[n_layers],
                **norm_kwargs,
                **block_kwargs,
            )
            for j in range(n_res_neck)
        )

    def is_stateful(self) -> bool:
        """Whether the model is stateful, i.e. uses batch normalization.

        Returns:
            `True` if the model has to be called as `model(x, state)` and returns
            `(out, state)`, `False` if it is called as `model(x)`.
        """
        return self.encoder[0].is_stateful()

    @override
    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        state: eqx.nn.State | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."] | tuple[Inexact[Array, "out_c ..."], eqx.nn.State]:
        """Forward pass of the CNO model.

        Args:
            x: Input array of shape `(in_channels, d1, ..., dN)`,
                where every `d_i` is divisible by `2 ** n_layers`.
            state: The state holding the running statistics of the batch
                normalizations. Required if (and only if) `is_stateful()`.
            key: Unused. Accepted for API compatibility with other neojax models.
            inference: Whether to use the running statistics of the batch normalizations
                instead of the batch statistics. Has no effect if the model is stateless.
                Default is `False`.

        Returns:
            The predicted field of shape `(out_channels, d1, ..., dN)`.
            If `is_stateful()`, the prediction and the updated state.

        Raises:
            ValueError: If the number of spatial dimensions of `x` does not match `ndim`.
            ValueError: If a spatial dimension of `x` is not divisible by `2 ** n_layers`.
            ValueError: If a state is required, but not given or vice versa.
        """
        if (state is not None) != self.is_stateful():
            raise ValueError(
                "A state has to be passed if (and only if) the model is stateful, "
                "i.e. `batch_norm=True`. Got "
                f"is_stateful()={self.is_stateful()} and state is None={state is None}."
            )
        if x.ndim - 1 != self.ndim:
            raise ValueError(
                f"Expected input with {self.ndim} spatial dimensions, "
                f"but got shape {x.shape}."
            )
        factor = 2**self.n_layers
        if any(s % factor != 0 for s in x.shape[1:]):
            raise ValueError(
                f"Spatial dimensions must be divisible by 2 ** n_layers = {factor}, "
                f"but got {x.shape[1:]}."
            )

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lift(x)

        # Encoder
        skips = []
        for res_blocks, down in zip(self.res_nets, self.encoder, strict=True):
            skip = x
            for res_block in res_blocks:
                skip, state = apply_layer(res_block, skip, state, inference=inference)
            skips.append(skip)
            x, state = apply_layer(down, x, state, inference=inference)

        # Bottleneck
        for res_block in self.res_net_neck:
            x, state = apply_layer(res_block, x, state, inference=inference)

        # Decoder
        for i, up in enumerate(self.decoder):
            if i == 0:
                x, state = apply_layer(
                    self.ed_expansion[self.n_layers], x, state, inference=inference
                )
            else:
                skip, state = apply_layer(
                    self.ed_expansion[self.n_layers - i],
                    skips[-i],
                    state,
                    inference=inference,
                )
                x = jnp.concatenate([x, skip], axis=0)
            if self.decoder_inv is not None:
                x, state = apply_layer(
                    self.decoder_inv[i], x, state, inference=inference
                )
            x, state = apply_layer(up, x, state, inference=inference)

        skip, state = apply_layer(
            self.ed_expansion[0], skips[0], state, inference=inference
        )
        x = self.project(jnp.concatenate([x, skip], axis=0))
        return (x, state) if self.is_stateful() else x
