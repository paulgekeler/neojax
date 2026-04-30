import jax
from jax import Array
import equinox as eqx


class InstanceNorm1d(eqx.Module):
    ln: eqx.nn.LayerNorm

    def __init__(self, shape_per_channel: int) -> None:
        # shape_per_channel is the length of the 1d signal
        # we disable learnable affine parameters to match torch neuralop config
        self.ln = eqx.nn.LayerNorm(
            shape=(shape_per_channel,), use_weight=False, use_bias=False
        )

    def __call__(self, x: Array) -> Array:
        return jax.vmap(self.ln)(x)
