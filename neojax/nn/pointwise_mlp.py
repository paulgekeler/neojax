"""Implementation of a general pointwise MLP."""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class PointwiseMLP(eqx.Module):
    """General pointwise MLP.

    Args:
        key: PRNG key for weight initialization.
        layers: Sequence of dimensions for each layer.
        activations: Sequence of activations between layers
            or single activation for all layers.
            If single activation, MLP is initialized with given l layers
            and l - 1 instances of type `activation`.
            Otherwise, a final activation
            after the last layer can be passed.
            Default is GeLu activation for all layers.

    Attributes:
        weights: Learnable weights.
        biases: Learnable biases.
        activations: Activation functions between layers.

    Example:
        ```python
        import jax.random as jr
        from neojax.nn import PointwiseMLP

        key = jr.PRNGKey(0)

        mlp = PointwiseMLP(
            key, [64, 128, 128], [jax.nn.gelu, jax.nn.gelu]
        )
        ```
    """

    weights: tuple[Float[Array, "out_c in_c"], ...]
    biases: tuple[Float[Array, "out_c"], ...]
    activations: tuple[Callable, ...]

    def __init__(
        self,
        key: PRNGKeyArray,
        layers: Sequence[int],
        activations: Callable | Sequence[Callable] = jax.nn.gelu,
    ) -> None:
        if isinstance(activations, Callable):
            self.activations = tuple(activations for _ in range(len(layers) - 1))
        else:
            if len(activations) > len(layers) or len(activations) < len(layers) - 2:
                raise ValueError(
                    "Mismatch in the number of activations and layers: "
                    "Can only have one less than or "
                    "the same num of activations than num layers,"
                    f" but got {len(activations)} activations and "
                    f"{len(layers)} layers!"
                )
            self.activations = tuple(activations)

        weights, biases = [], []
        for i in range(len(layers) - 1):
            wkey, key = jr.split(key, 2)
            scale = 1.0 / jnp.sqrt(layers[i])
            weights.append(jr.normal(wkey, shape=(layers[i], layers[i + 1])) * scale)
            biases.append(jnp.zeros((layers[i + 1],)))
        self.weights = tuple(weights)
        self.biases = tuple(biases)

    def __call__(self, x: Float[Array, "... in_c"]) -> Float[Array, "out_c ..."]:
        """MLP forward pass.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        for w, b, a in zip(self.weights, self.biases, self.activations, strict=False):
            x = jnp.einsum("...i,ji->...j", x, w) + b
            if a:
                x = a(x)
        return x
