"""Tests for the physical scales."""

import jax.numpy as jnp

from neojax.data.scales import (
    CharacteristicLengthScale,
    GridBasedScale,
    ReynoldsScale,
)


def test_characteristic_length_scale():
    """Tests the constant length scale."""
    l_ref = 2.5
    scale_provider = CharacteristicLengthScale(L_ref=l_ref)
    x = jnp.ones((10, 32, 32))

    scale = scale_provider.get_scale(x)
    assert jnp.allclose(scale, l_ref)
    assert scale.shape == ()


def test_reynolds_scale():
    """Tests the Reynolds number scale."""
    u, l, nu = 10.0, 1.0, 0.1
    # Re = 10 * 1 / 0.1 = 100
    scale_provider = ReynoldsScale(U=u, L=l, nu=nu)
    x = jnp.ones((5, 16))

    scale = scale_provider.get_scale(x)
    assert jnp.allclose(scale, 100.0)
    assert scale.shape == ()


def test_grid_based_scale():
    """Tests the spatially varying scale."""
    # 1D grid
    grid = jnp.linspace(1.0, 2.0, 10)
    scale_provider = GridBasedScale(scale_field=grid)

    # Matching shape
    x = jnp.zeros((10,))
    scale = scale_provider.get_scale(x)
    assert jnp.allclose(scale, grid)
    assert scale.shape == (10,)

    # Batch dimension (broadcasting)
    x_batch = jnp.zeros((5, 10))
    scale_batch = scale_provider.get_scale(x_batch)
    assert scale_batch.shape == (5, 10)
    assert jnp.allclose(scale_batch[0], grid)

    # Small epsilon check
    grid_with_zero = jnp.array([0.0, 1.0])
    scale_provider_eps = GridBasedScale(scale_field=grid_with_zero)
    scale_eps = scale_provider_eps.get_scale(jnp.zeros((2,)), eps=1e-3)
    assert scale_eps[0] == 1e-3
