import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.mollifier_functions import make_weighting_fn
from neojax.tests.conftest import assert_filter_jittable


@pytest.mark.parametrize("name", ["bump", "half_cos", "quadr", "quartic", "octic"])
def test_mollifier_jittable(name):
    assert_filter_jittable(make_weighting_fn(name, 1.0), jnp.linspace(0.0, 2.0, 20))


def test_wrong_name_raises():
    with pytest.raises((ValueError, TypeCheckError)):
        _ = make_weighting_fn("invalid", sq_radius=1.0)
