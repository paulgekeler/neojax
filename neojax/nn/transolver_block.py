"""Implementation of the Transolver block."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.nn.normalization import ChannelLayerNorm
from neojax.nn.physics_attention import PhysicsAttention
from neojax.nn.pointwise_mlp import PointwiseMLP


class TransolverBlock(eqx.Module):
    r"""Transolver block.

    A pre-norm transformer encoder block whose self-attention is replaced by
    `PhysicsAttention`, i.e.
    $$
    f' = f + \mathrm{PhysicsAttention}(\mathrm{LN}(f)), \quad
    f'' = f' + \mathrm{MLP}(\mathrm{LN}(f')).
    $$

    Args:
        key: PRNG key for parameter initialization.
        hidden_channels: Number of input and output channels.
        n_heads: Number of attention heads. Default is 8.
        slice_num: Number of slices of the `PhysicsAttention`. Default is 32.
        mlp_ratio: Ratio of the MLP hidden width to `hidden_channels`. Default is 4.
        dropout: Dropout probability inside the `PhysicsAttention`. Default is 0.0.
        activation: Activation function of the MLP. Default is `jax.nn.gelu`.
        ndim: Number of spatial dimensions for structured grid inputs
            (convolutional attention projections) or `None`
            for pointwise projections (e.g. unstructured meshes).
            See `PhysicsAttention`. Default is `None`.
        kernel_size: Kernel size of the convolutional attention projections.
            Default is 3.
        temperature_range: Clipping interval of the slice temperature.
            See `PhysicsAttention`. Default is `(0.1, 5.0)`.

    Raises:
        ValueError: If `hidden_channels` is not divisible by `n_heads`,
            or `mlp_ratio` is not positive.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **norm_1** (`ChannelLayerNorm`): Layer norm before the attention.
        * **attn** (`PhysicsAttention`): The physics attention.
        * **norm_2** (`ChannelLayerNorm`): Layer norm before the MLP.
        * **mlp** (`PointwiseMLP`): The pointwise MLP.

    """

    norm_1: ChannelLayerNorm
    attn: PhysicsAttention
    norm_2: ChannelLayerNorm
    mlp: PointwiseMLP

    def __init__(
        self,
        key: PRNGKeyArray,
        hidden_channels: int,
        n_heads: int = 8,
        slice_num: int = 32,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: Callable = jax.nn.gelu,
        ndim: int | None = None,
        kernel_size: int = 3,
        temperature_range: tuple[float, float] | None = (0.1, 5.0),
    ) -> None:
        if mlp_ratio <= 0:
            raise ValueError("`mlp_ratio` must be positive.")
        k_attn, k_mlp = jr.split(key)
        self.norm_1 = ChannelLayerNorm(hidden_channels)
        self.attn = PhysicsAttention(
            key=k_attn,
            dim=hidden_channels,
            n_heads=n_heads,
            slice_num=slice_num,
            dropout=dropout,
            ndim=ndim,
            kernel_size=kernel_size,
            temperature_range=temperature_range,
        )
        self.norm_2 = ChannelLayerNorm(hidden_channels)
        self.mlp = PointwiseMLP(
            key=k_mlp,
            layers=(
                hidden_channels,
                int(hidden_channels * mlp_ratio),
                hidden_channels,
            ),
            activations=activation,
        )

    def __call__(
        self,
        x: Inexact[Array, "c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "c ..."]:
        """Forward pass of the Transolver block.

        Args:
            x: Input array shaped (hidden_channels, N) or (hidden_channels, d1, ..., dN).
            key: PRNG key used for dropout masks.
            inference: If `True`, dropout is disabled.

        Returns:
            Output array with the same shape as `x`.
        """
        x = x + self.attn(self.norm_1(x), key=key, inference=inference)
        return x + self.mlp(self.norm_2(x))
