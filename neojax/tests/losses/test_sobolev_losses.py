import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.losses.sobolev_losses import SobolevLoss
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture(scope="class")
def mlp():
    key = jr.key(0)
    return PointwiseMLP(key, layers=(2, 16, 2), activations=jax.nn.tanh)


@pytest.mark.usefixtures("mlp")
class TestSobolevLoss:
    def test_instantiation(self):
        # Valid instantiation
        loss = SobolevLoss(k=1, p=2.0, method="exact", diff_mode="fwd")
        assert loss.k == 1
        assert loss.p == 2.0

        # Invalid method
        with pytest.raises((ValueError, TypeCheckError)):
            SobolevLoss(method="invalid")

        # Invalid diff_mode
        with pytest.raises((ValueError, TypeCheckError)):
            SobolevLoss(diff_mode="invalid")

        # Invalid random_type
        with pytest.raises((ValueError, TypeCheckError)):
            SobolevLoss(random_type="invalid")

    def test_exact_sobolev(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4, 4))
        targets = jr.normal(key, (10, 2, 4, 4))

        loss_fn = SobolevLoss(k=1, method="exact", diff_mode="fwd")
        loss_val = loss_fn(mlp, x=inputs, target=targets)

        assert loss_val.shape == ()
        assert not jnp.isnan(loss_val)

    def test_interpolated_sobolev(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (5, 2, 4))
        targets = jr.normal(key, (5, 2, 4))

        loss_fn = SobolevLoss(k=2, method="interpolate")
        loss_val = loss_fn(mlp, x=inputs, target=targets)

        assert loss_val.shape == ()
        assert not jnp.isnan(loss_val)

    def test_stochastic_sobolev(self, mlp):
        key = jr.key(0)
        mkey, xkey, ykey, lkey = jr.split(key, 4)
        inputs = jr.normal(xkey, (10, 2, 4))
        targets = jr.normal(ykey, (10, 2, 4))

        loss_fn = SobolevLoss(k=2, method="stochastic", n_random_samples=2)
        loss_val = loss_fn(mlp, x=inputs, target=targets, key=lkey)

        assert loss_val.shape == ()
        assert not jnp.isnan(loss_val)

    def test_missing_key_stochastic(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4))
        targets = jr.normal(key, (10, 2, 4))

        loss_fn = SobolevLoss(k=2, method="stochastic")
        with pytest.raises(ValueError):
            loss_fn(mlp, x=inputs, target=targets)

    def test_diff_modes(self):
        key = jr.key(0)
        mlp_fwd = PointwiseMLP(key, layers=(2, 8, 4), activations=jax.nn.tanh)
        inputs_fwd = jr.normal(key, (5, 2, 3))
        targets_fwd = jr.normal(key, (5, 4, 3))

        loss_fn_auto = SobolevLoss(k=1, method="exact", diff_mode="auto")

        # This triggers fwd mode since in_size (2) <= out_size (4)
        loss_fwd = loss_fn_auto(mlp_fwd, x=inputs_fwd, target=targets_fwd)
        assert not jnp.isnan(loss_fwd)

        mlp_bwd = PointwiseMLP(key, layers=(4, 8, 2), activations=jax.nn.tanh)
        inputs_bwd = jr.normal(key, (5, 4, 3))
        targets_bwd = jr.normal(key, (5, 2, 3))

        # This triggers bwd mode since in_size (4) > out_size (2)
        loss_bwd = loss_fn_auto(mlp_bwd, x=inputs_bwd, target=targets_bwd)
        assert not jnp.isnan(loss_bwd)

    def test_jittability(self, mlp):
        key = jr.key(0)
        mkey, xkey, ykey, lkey = jr.split(key, 4)
        inputs = jr.normal(xkey, (5, 2, 3))
        targets = jr.normal(ykey, (5, 2, 3))

        loss_fn_exact = SobolevLoss(k=1, method="exact")
        loss_fn_interp = SobolevLoss(k=2, method="interpolate")
        loss_fn_stoch = SobolevLoss(k=2, method="stochastic", n_random_samples=2)

        # We wrap the loss logic because `assert_filter_jittable` doesn't pass **kwargs natively
        def wrap_exact(m, x, t):
            return loss_fn_exact(model=m, x=x, target=t)

        def wrap_interp(m, x, t):
            return loss_fn_interp(model=m, x=x, target=t)

        def wrap_stoch(m, x, t, k):
            return loss_fn_stoch(model=m, x=x, target=t, key=k)

        assert_filter_jittable(wrap_exact, mlp, inputs, targets)
        assert_filter_jittable(wrap_interp, mlp, inputs, targets)
        assert_filter_jittable(wrap_stoch, mlp, inputs, targets, lkey)

    def test_auto_routing(self, mlp):
        key = jr.key(0)

        inputs = jr.normal(key, (5, 2, 4))
        targets = jr.normal(key, (5, 2, 4))

        # Evaluate routing conditions to avoid crashing, just verify return type
        loss_val_1 = SobolevLoss(k=1, method="auto")(mlp, x=inputs, target=targets)
        loss_val_2 = SobolevLoss(k=2, method="auto")(mlp, x=inputs, target=targets)
        assert not jnp.isnan(loss_val_1)
        assert not jnp.isnan(loss_val_2)
