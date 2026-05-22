"""Implementation of a n-dimensional FNO."""

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.nn.domain_padding import DomainPadding
from neojax.nn.fno_blocks import FNOBlocks
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.nn.positional_embedding import GridEmbeddingNd


class FNO(eqx.Module):
    """General n-dimensional Fourier Neural Operator (FNO).

    The model consists of a lifting layer that maps
    the input to a higher-dimensional latent space,
    a sequence of FNO blocks (spectral convolutions + skip connections)
    optionally interleaved with pointwise MLPs,
    and a final projection layer.

    The implementation is in accordance with [^1] and [^2].

    Args:
        key: PRNG key for parameter initialization.
        in_channels: Number of input channels
            (e.g., coordinates + initial conditions).
        out_channels: Number of output channels (e.g., solution field).
        hidden_channels: Hidden channel dimension (latent width)
            used throughout the FNO blocks. This significantly affects
            the number of parameters. Good starting point is 64 and
            increase if needed. Update `lift_channel_ratio` and
            `proj_channel_ratio` accordingly.
            They scale proportional to hidden_channels.
        n_layers: Number of consecutive FNO blocks.
        modes: Number of Fourier modes to retain
            across each spatial dimension.
        activation: Activation function used within the layers.
            Defaults to `jax.nn.gelu`.
        use_channel_mlp: Whether to apply a pointwise channel MLP
            after each FNO block. Defaults to `True`.
        fno_skip: Type of skip connection inside FNO blocks.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"linear"`.
        channel_mlp_skip: Type of skip connection around channel MLPs.
            Can be `"linear"`, `"soft-gating"`, `"identity"`, or None.
            Defaults to `"soft-gating"`.
        n_lift_layers: Number of layers in the lifting MLP.
            Defaults to 2.
        n_proj_layers: Number of layers in the projection MLP.
            Defaults to 2.
        lift_channel_ratio: Ratio of lifting channels
            to hidden_channels.
            The number of lifting channels in the lifting block
            of the FNO is lifting_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        proj_channel_ratio: Ratio of projection channels
            to hidden_channels.
            The number of projection channels in the projection block
            of the FNO is projection_channel_ratio * hidden_channels
            (e.g. default 2 * hidden_channels).
        positional_embedding: Positional embedding to apply
            to last channels of raw input before passing through FNO.
            Defaults to regular GridEmbeddingNd on a ((0, 1), ...) grid.
        domain_padding: Percentage of padding to use.
            If single float, this padding is used for all dims.
            Sequence of floats indicates padding percentage per dim.
            Default is None, no padding.

    Attributes:
        lifting: The `PointwiseMLP` used to lift inputs
            to the hidden `width`.
        fno_blocks: The `FNOBlocks` sequence
            containing spectral convolutions.
        projection: The `PointwiseMLP` used to project
            latent features to `out_channels`.

    Notes: Current implementation doesn't support Tucker factorization,
        different FNO block precisions, resolution scaling, stabilizers,
        separable spectral convolutions or enforcing hermitian symmetry.
        These will be added in future releases.

    References:
        [^1]: Li, Z. et al. "Fourier Neural Operator for Parametric
            Partial Differential Equations" (2021).
            ICLR 2021, https://arxiv.org/pdf/2010.08895.
        [^2]: Kovachki, N. et al. "Neural Operator: Learning Maps
            Between Function Spaces With Applications to PDEs"
            JMLR 2023, https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf.
    """

    positional_embedding: GridEmbeddingNd | None
    lifting: PointwiseMLP
    fno_blocks: FNOBlocks
    projection: PointwiseMLP
    padding: DomainPadding | None

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
        modes: int | Sequence[int],
        activation: Callable = jax.nn.gelu,
        use_channel_mlp: bool = True,
        fno_skip: Literal["linear", "soft-gating", "identity"] | None = "linear",
        channel_mlp_skip: Literal["linear", "soft-gating", "identity"]
        | None = "soft-gating",
        channel_mlp_expansion: float | None = 0.5,
        preactivation: bool = False,
        n_lift_layers: int = 2,
        n_proj_layers: int = 2,
        lift_channel_ratio: float = 2.0,
        proj_channel_ratio: float = 2.0,
        positional_embedding: GridEmbeddingNd | None = None,
        domain_padding: float | Sequence[float] | None = None,
    ) -> None:
        lkey, pkey, fno_key = jr.split(key, 3)

        self.positional_embedding = positional_embedding
        if domain_padding is not None:
            self.padding = DomainPadding(domain_padding)
        else:
            self.padding = None

        lift_hidden_channels = int(hidden_channels * lift_channel_ratio)
        self.lifting = PointwiseMLP(
            key=lkey,
            layers=(in_channels, lift_hidden_channels, hidden_channels),
            activations=activation,
        )
        proj_hidden_channels = int(hidden_channels * proj_channel_ratio)
        self.projection = PointwiseMLP(
            key=pkey,
            layers=(hidden_channels, proj_hidden_channels, out_channels),
            activations=activation,
        )
        self.fno_blocks = FNOBlocks(
            key=fno_key,
            n_layers=n_layers,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            modes=modes,
            use_channel_mlp=use_channel_mlp,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            channel_mlp_expansion=channel_mlp_expansion,
        )

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Forward pass of the FNO model.

        Args:
            x: Input array of shape `(in_channels, d1, ..., dn)`,
                where `n` matches the dimensionality of `modes`.

        Returns:
            The predicted field of shape `(out_channels, d1, ..., dn)`.
        """
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        original_shape = x.shape
        if self.padding is not None:
            x = self.padding.pad(x)

        x = self.lifting(x)
        x = self.fno_blocks(x)
        x = self.projection(x)

        if self.padding is not None:
            x = self.padding.unpad(x, original_shape)

        return x
