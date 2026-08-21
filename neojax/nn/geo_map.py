"""Implementation of a Geo-FNO-like diffeomorphism NN."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class GeoMapNd(eqx.Module):
    r"""Physical to latent space map NN.

    Learns a soft-diffeomorphism between a
    mesh-like physical domain and a regular
    grid latent domain
    $$
    \xi = \phi^{-1}(x),
    $$
    where $\xi \in D_L$ are the coordinates of a uniform,
    regular latent space and $x \in D_P$ are the coordinates of
    the irregular physical domain.

    Args:
        key: PRNG key for parameter initialization.
        dim: Dimensionality of the coordinates.
        width: Width of the hidden linear layers.
            Default is 32.
        code_dim: Optional dimensionality of parameter/domain codes.

    ??? info "Internal Attributes"
        * **lin0** (`eqx.Module`): First linear layer.
        * **lin1** (`eqx.Module`): Hidden linear layer 1.
        * **lin2** (`eqx.Module`): Hidden linear layer 2.
        * **lin3** (`eqx.Module`): Output projection layer.
        * **lin_code** (`eqx.Module | None`): Encoder layer for condition code.
        * **lin_no_code** (`eqx.Module`): Layer to transform features when no code is present.
        * **center** (`Float[Array, "1 d"]`): Center of coordinate transformation.
        * **b** (`Float[Array, "1 1 w"]`): Frequency scales for NeRF feature expansion.
        * **dim** (`int`): Dimensionality of the coordinates.
        * **width** (`int`): Base width parameter.
        * **code_dim** (`int | None`): Optional dimensionality of parameter/domain codes.
    """

    lin0: eqx.nn.Linear
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear
    lin_code: eqx.nn.Linear | None
    lin_no_code: eqx.nn.Linear
    center: Float[Array, "1 d"]
    b: Float[Array, "1 1 w"]
    dim: int = eqx.field(static=True)
    width: int = eqx.field(static=True)
    code_dim: int | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int = 2,
        width: int = 32,
        code_dim: int | None = None,
    ) -> None:
        self.dim = dim
        self.width = width
        self.code_dim = code_dim

        k0, k1, k2, k3, k_code, k_no_code = jax.random.split(key, 6)

        # 2*dim features: Cartesian (dim) + Angles (dim-1) + Radius (1)
        self.lin0 = eqx.nn.Linear(2 * dim, width, key=k0)

        hidden_width = (dim + 2) * width

        if code_dim is not None:
            self.lin_code = eqx.nn.Linear(code_dim, width, key=k_code)
        else:
            self.lin_code = None

        self.lin_no_code = eqx.nn.Linear((dim + 1) * width, hidden_width, key=k_no_code)
        self.lin1 = eqx.nn.Linear(hidden_width, hidden_width, key=k1)
        self.lin2 = eqx.nn.Linear(hidden_width, hidden_width, key=k2)
        self.lin3 = eqx.nn.Linear(hidden_width, dim, key=k3)
        self.center = jnp.ones((1, dim)) * 0.5
        self.b = jnp.pi * (2.0 ** jnp.arange(0, width // 4)).reshape(1, 1, width // 4)

    def __call__(
        self, x: Float[Array, "N d"], code: Float[Array, "..."] | None = None
    ) -> Float[Array, "N d"]:
        """Maps mesh coordinates to latent regular coordinates.

        Args:
            x: Physical coordinates of shape (N, d).
            code: Optional conditioning code.

        Returns:
            Latent coordinates of shape (N, d).
        """
        d = self.dim
        n = x.shape[0]
        y = x - self.center

        # Get hyperspherical angles in Nd
        if d == 1:
            angles = jnp.zeros((n, 0))
        elif d == 2:
            angles = jnp.atan2(y[:, 1], y[:, 0])[:, None]
        else:
            y_sq = jnp.square(y)
            suffix_sum = jnp.cumsum(y_sq[:, ::-1], axis=-1)[:, ::-1]

            num = jnp.sqrt(suffix_sum[:, 1:-1])
            den = y[:, :-2]
            angles_first = jnp.atan2(num, den)

            angle_last = jnp.atan2(y[:, -1], y[:, -2])[:, None]
            angles = jnp.concat([angles_first, angle_last], axis=-1)

        radius = jnp.linalg.norm(y, axis=-1)
        xd = jnp.concat([x, angles, radius[:, None]], axis=-1)

        # NeRF features
        x_sin = jnp.sin(self.b * xd[..., None]).reshape(n, 2 * d * self.width // 4)
        x_cos = jnp.cos(self.b * xd[..., None]).reshape(n, 2 * d * self.width // 4)

        # Vectorize the linear layers across nodes
        xd = jax.vmap(self.lin0)(xd)
        xd = jnp.concat([xd, x_sin, x_cos], axis=-1)  # shape: (n, (d + 1)*width)

        if code is None:
            xd = jax.vmap(self.lin_no_code)(xd)
        else:
            if self.lin_code is None:
                raise ValueError("code_dim was not provided during initialization.")
            cd = self.lin_code(code)
            if cd.ndim == 1:
                cd = jnp.tile(cd, (n, 1))
            elif cd.ndim == 2 and cd.shape[0] == 1:
                cd = jnp.tile(cd, (n, 1))
            xd = jnp.concat([cd, xd], axis=-1)

        xd = jax.vmap(self.lin1)(xd)
        xd = jax.nn.gelu(xd)
        xd = jax.vmap(self.lin2)(xd)
        xd = jax.nn.gelu(xd)
        xd = jax.vmap(self.lin3)(xd)

        # Output deformation phi(x) = x + x * deformation
        return x + x * xd
