"""Implementation of a general pointwise MLP."""

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Inexact, PRNGKeyArray


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
        dropout: Dropout probability applied after each layer (except the last).
            If 0, no dropout is applied. Defaults to 0.0.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **weights** (`tuple[Float[Array, ...], ...]`): Learnable weights.
        * **biases** (`tuple[Float[Array, ...], ...]`): Learnable biases.
        * **activations** (`tuple[Callable, ...]`): Activation functions between layers.
        * **dropout** (`float`): Dropout probability.

    Example:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import PointwiseMLP

        key = jr.key(0)

        mlp = PointwiseMLP(
            key, [64, 128, 128], [jax.nn.gelu, jax.nn.gelu], dropout=0.1
        )
        x = jnp.ones((64, 32, 32))
        out = mlp(x, key=key)
        ```
    """

    weights: tuple[Float[Array, "out_c in_c"], ...]
    biases: tuple[Float[Array, "out_c"], ...]
    activations: tuple[Callable, ...]
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        layers: Sequence[int],
        activations: Callable | Sequence[Callable] = jax.nn.gelu,
        dropout: float = 0.0,
    ) -> None:
        if not isinstance(dropout, (int, float)):
            raise ValueError("dropout must be a float.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")
        self.dropout = float(dropout)

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
            weights.append(jr.normal(wkey, shape=(layers[i + 1], layers[i])) * scale)
            biases.append(jnp.zeros((layers[i + 1],)))

        self.weights = tuple(weights)
        self.biases = tuple(biases)

    def __call__(
        self,
        x: Inexact[Array, "in_c ..."],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
    ) -> Inexact[Array, "out_c ..."]:
        """MLP forward pass.

        Args:
            x: Input array.
            key: PRNG key used for dropout masks.
            inference: If True, dropout is disabled.

        Returns:
            Output array.
        """
        n_layers = len(self.weights)
        if key is not None and self.dropout > 0.0 and not inference:
            keys = jr.split(key, n_layers - 1)
        else:
            keys = [None] * (n_layers - 1)

        for i, (w, b, a) in enumerate(zip(self.weights, self.biases, self.activations, strict=False)):
            x = jnp.einsum("i...,ji->j...", x, w)
            x = x + b.reshape(-1, *([1] * (x.ndim - 1)))
            if a:
                x = a(x)

            if i < n_layers - 1 and self.dropout > 0.0 and not inference:
                drop_key = keys[i]
                if drop_key is None:
                    raise ValueError("A PRNG key must be passed to __call__ when dropout > 0 and inference = False.")
                mask = jr.bernoulli(drop_key, 1.0 - self.dropout, shape=(x.shape[0],))
                mask = mask.reshape(-1, *([1] * (x.ndim - 1)))
                x = jnp.where(mask, x / (1.0 - self.dropout), 0.0)

        return x
