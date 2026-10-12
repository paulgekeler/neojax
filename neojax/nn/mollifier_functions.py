"""Implementation of weighting (mollifier) functions (as used by the mollified GNO)."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float


def bump_cutoff(
    x: Float[Array, "..."],
    radius: float = 1.0,
    scale: float = 1.0,
    eps: float = 1e-7,
) -> Float[Array, "..."]:
    r"""Bump cutoff function with exponential decay.

    $w(x) = s \cdot e \cdot \exp\left(-1 / (1 - d^2 + \varepsilon)\right)$
    with $d = x / r$ clipped to $[0, 1]$ where $e$ is Euler's number.

    Args:
        x: Values in $[0, r]$.
        radius: Cutoff radius $r$.
        scale: Scaling factor $s$ of the weights.
        eps: Small constant for numerical stability.

    Returns:
        The weights, same shape as `x`.
    """
    d = jnp.clip(x, 0.0, radius) / radius
    return jnp.exp(-1.0 / ((1.0 - d**2) + eps)) * jnp.e * scale


def half_cos_cutoff(
    x: Float[Array, "..."], radius: float = 1.0, scale: float = 1.0
) -> Float[Array, "..."]:
    r"""Half-cosine cutoff function with smooth decay.

    $w(x) = s \, (0.5 \cos(\pi d) + 0.5)$ with $d = x / r$.

    Args:
        x: Values in $[0, r]$.
        radius: Cutoff radius $r$.
        scale: Scaling factor $s$ of the weights.

    Returns:
        The weights, same shape as `x`.
    """
    d = x / radius
    return scale * (0.5 * jnp.cos(jnp.pi * d) + 0.5)


def quadr_cutoff(
    x: Float[Array, "..."], radius: float = 1.0, scale: float = 1.0
) -> Float[Array, "..."]:
    r"""Piecewise quadratic cutoff function.

    $w(x) = s \cdot \begin{cases} 1 - 2d^2 & d < 0.5 \\ 2(1 - d)^2 & d \geq 0.5
    \end{cases}$ with $d = x / r$.

    Args:
        x: Values in $[0, r]$.
        radius: Cutoff radius $r$.
        scale: Scaling factor $s$ of the weights.

    Returns:
        The weights, same shape as `x`.
    """
    d = x / radius
    return scale * jnp.where(d < 0.5, 1.0 - 2.0 * d**2, 2.0 * (1.0 - d) ** 2)


def quartic_cutoff(
    x: Float[Array, "..."], radius: float = 1.0, scale: float = 1.0
) -> Float[Array, "..."]:
    r"""Quartic cutoff function.

    $w(x) = s \, (1 - d^2)^2$ with $d = x / r$.

    Args:
        x: Values in $[0, r]$.
        radius: Cutoff radius $r$.
        scale: Scaling factor $s$ of the weights.

    Returns:
        The weights, same shape as `x`.
    """
    d = x / radius
    return scale * (d**4 - 2.0 * d**2 + 1.0)


def octic_cutoff(
    x: Float[Array, "..."], radius: float = 1.0, scale: float = 1.0
) -> Float[Array, "..."]:
    r"""Octic cutoff function with eighth-order polynomial.

    $w(x) = s \, (-3d^8 + 8d^6 - 6d^4 + 1)$ with $d = x / r$.

    Args:
        x: Values in $[0, r]$.
        radius: Cutoff radius $r$.
        scale: Scaling factor $s$ of the weights.

    Returns:
        The weights, same shape as `x`.
    """
    d = x / radius
    return scale * (-3.0 * d**8 + 8.0 * d**6 - 6.0 * d**4 + 1.0)


WEIGHTING_FN_REGISTRY: dict[str, Callable[..., Float[Array, "..."]]] = {
    "bump": bump_cutoff,
    "half_cos": half_cos_cutoff,
    "quadr": quadr_cutoff,
    "quartic": quartic_cutoff,
    "octic": octic_cutoff,
}


def make_weighting_fn(
    name: Literal["bump", "half_cos", "quadr", "quartic", "octic"],
    sq_radius: float,
    scale: float = 1.0,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    """Selects a weighting function for the output GNO of a mollified GNO.

    The weighting functions take the squared radius
    of the neighborhoods because they are evaluated on the squared distances
    between query and neighbor. They replace the hard cutoff of the neighborhood
    with a smooth weight, which makes the GNO differentiable in the query position.

    Args:
        name: Name of the weighting function.
            One of `"bump"`, `"half_cos"`, `"quadr"`, `"quartic"` or `"octic"`.
        sq_radius: Squared radius of the neighborhoods.
        scale: Factor by which to scale all weights. Default is 1.0.

    Returns:
        A function mapping squared distances to weights.

    Raises:
        ValueError: If `name` is not a registered weighting function.
    """
    if name not in WEIGHTING_FN_REGISTRY:
        raise ValueError(
            f"Weighting function must be one of {list(WEIGHTING_FN_REGISTRY)}, "
            f"but got '{name}'."
        )
    return partial(WEIGHTING_FN_REGISTRY[name], radius=sq_radius, scale=scale)
