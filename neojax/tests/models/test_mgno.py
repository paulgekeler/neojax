import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import TypeCheckError

from neojax.models.mgno import MgNO
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def mgno():
    k = jr.key(0)
    return MgNO.make_mgno_type(k, "MgNO_NS", n_layers=5)


@pytest.mark.usefixtures("mgno")
class TestMgNO:
    def test_invalid_activation_raises(self):
        with pytest.raises((ValueError, TypeCheckError)):
            k = jr.key(3)
            MgNO(
                k,
                n_layers=2,
                n_its_per_level=(1, 1, 1),
                n_channels_u=20,
                n_channels_f=1,
                activation="invalid",
            )

    def test_ns_variant(self, mgno):
        k = jr.key(2)
        in_field = jr.normal(k, (1, 64, 64))
        out = mgno(in_field)
        assert out.shape == in_field.shape

    def test_helm_variant(self):
        k = jr.key(4)
        in_field = jr.normal(k, (1, 101, 101))
        mgno_helm = MgNO.make_mgno_type(k, "MgNO_helm", 4)
        out = mgno_helm(in_field)
        assert out.shape == in_field.shape

    def test_helm2_variant(self):
        k = jr.key(4)
        in_field = jr.normal(k, (1, 101, 101))
        mgno_helm = MgNO.make_mgno_type(k, "MgNO_helm2", 4)
        out = mgno_helm(in_field)
        assert out.shape == in_field.shape

    def test_darcy_variant(self):
        k = jr.key(4)
        in_field = jr.normal(k, (1, 128, 128))
        mgno_dc = MgNO.make_mgno_type(k, "MgNO_DC", 4)
        out = mgno_dc(in_field)
        assert out.shape == in_field.shape

    def test_darcy_smooth_variant(self):
        k = jr.key(4)
        in_field = jr.normal(k, (1, 211, 211))
        mgno_dcs = MgNO.make_mgno_type(k, "MgNO_DC_smooth", 4)
        out = mgno_dcs(in_field)
        assert out.shape == in_field.shape

    def test_batched_grad(self, mgno):
        import equinox as eqx

        k = jr.key(1)
        in_fields = jr.normal(k, (2, 1, 64, 64))

        def loss_fn(m):
            outs = jax.vmap(m)(in_fields)
            return jnp.sum(outs)

        grad = eqx.filter_grad(loss_fn)(mgno)
        grad_leaves = jtu.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
        assert grad_leaves
        assert all(bool(jnp.all(jnp.isfinite(l))) for l in grad_leaves)

    def test_jittable(self, mgno):
        k = jr.key(2)
        in_field = jr.normal(k, (1, 64, 64))
        assert_filter_jittable(mgno, in_field)
