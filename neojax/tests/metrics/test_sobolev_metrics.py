import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.metrics.sobolev_metrics import SobolevMetric
from neojax.models.fno import FNO
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture(scope="class")
def mlp():
    key = jr.key(0)
    return PointwiseMLP(key, layers=(2, 16, 2), activations=jax.nn.tanh)


@pytest.mark.usefixtures("mlp")
class TestSobolevMetric:
    def test_instantiation(self):
        metric = SobolevMetric(k=1, p=2.0, method="exact", diff_mode="fwd")
        assert metric.k == 1
        assert metric.p == 2.0

        with pytest.raises((ValueError, TypeCheckError)):
            SobolevMetric(method="invalid")

        with pytest.raises((ValueError, TypeCheckError)):
            SobolevMetric(diff_mode="invalid")

    def test_exact_sobolev(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4, 4))
        targets = jr.normal(key, (10, 2, 4, 4))

        metric_fn = SobolevMetric(k=1, method="exact", diff_mode="fwd")
        metric_val = metric_fn(mlp, x=inputs, target=targets)

        assert metric_val.shape == ()
        assert not jnp.isnan(metric_val)

    def test_interpolated_sobolev(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (5, 2, 4))
        targets = jr.normal(key, (5, 2, 4))

        metric_fn = SobolevMetric(k=2, method="interpolate")
        metric_val = metric_fn(mlp, x=inputs, target=targets)

        assert metric_val.shape == ()
        assert not jnp.isnan(metric_val)

    def test_stochastic_stde_sobolev(self, mlp):
        key = jr.key(0)
        xkey, ykey, lkey = jr.split(key, 3)
        inputs = jr.normal(xkey, (10, 2, 4))
        targets = jr.normal(ykey, (10, 2, 4))

        metric_fn = SobolevMetric(k=2, method="stochastic", n_random_samples=2)
        metric_val = metric_fn(mlp, x=inputs, target=targets, key=lkey)

        assert metric_val.shape == ()
        assert not jnp.isnan(metric_val)

    def test_stochastic_first_order_sobolev(self, mlp):
        key = jr.key(0)
        xkey, ykey, lkey = jr.split(key, 3)
        inputs = jr.normal(xkey, (10, 2, 4))
        targets = jr.normal(ykey, (10, 2, 4))

        metric_fn_fwd = SobolevMetric(
            k=1, method="stochastic", n_random_samples=2, diff_mode="fwd"
        )
        metric_fn_bwd = SobolevMetric(
            k=1, method="stochastic", n_random_samples=2, diff_mode="bwd"
        )
        metric_fn_auto = SobolevMetric(
            k=1, method="stochastic", n_random_samples=2, diff_mode="auto"
        )

        metric_val_fwd = metric_fn_fwd(mlp, x=inputs, target=targets, key=lkey)
        metric_val_bwd = metric_fn_bwd(mlp, x=inputs, target=targets, key=lkey)
        metric_val_auto = metric_fn_auto(mlp, x=inputs, target=targets, key=lkey)

        for mv in (metric_val_fwd, metric_val_bwd, metric_val_auto):
            assert mv.shape == ()
            assert not jnp.isnan(mv)

            # x.size == target.size here, so "auto" resolves to the "fwd" branch
            # (ties go to jvp) -- same code path and key, so this must match exactly.
        assert jnp.allclose(metric_val_fwd, metric_val_auto)

        # Ensure fwd and bwd match exact if n_random_samples >= total number of basis vectors
        exact_metric = SobolevMetric(k=1, method="exact")(mlp, x=inputs, target=targets)
        metric_fn_fwd_full = SobolevMetric(
            k=1, method="stochastic", n_random_samples=inputs[0].size, diff_mode="fwd"
        )
        # Use inputs or targets here, they are the same size
        metric_fn_bwd_full = SobolevMetric(
            k=1, method="stochastic", n_random_samples=inputs[0].size, diff_mode="bwd"
        )
        fwd_full = metric_fn_fwd_full(mlp, x=inputs, target=targets, key=lkey)
        bwd_full = metric_fn_bwd_full(mlp, x=inputs, target=targets, key=lkey)
        assert jnp.allclose(fwd_full, exact_metric, atol=1e-4)
        assert jnp.allclose(bwd_full, exact_metric, atol=1e-4)

    def test_missing_key_stochastic(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4))
        targets = jr.normal(key, (10, 2, 4))

        metric_fn = SobolevMetric(k=2, method="stochastic")
        with pytest.raises(ValueError):
            metric_fn(mlp, x=inputs, target=targets)

    def test_incomplete_inputs_raises(self, mlp):
        metric_fn = SobolevMetric(k=1, method="auto")
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4))
        targets = inputs
        with pytest.raises(ValueError):
            metric_fn(mlp, target=targets)

        with pytest.raises(ValueError):
            metric_fn(target=targets, x=inputs)

    def test_diff_modes(self):
        key = jr.key(0)
        mlp_fwd = PointwiseMLP(key, layers=(2, 8, 4), activations=jax.nn.tanh)
        inputs_fwd = jr.normal(key, (5, 2, 3))
        targets_fwd = jr.normal(key, (5, 4, 3))

        metric_fn_auto = SobolevMetric(k=1, method="exact", diff_mode="auto")

        # Trigger fwd mode since in_size (2) <= out_size (4)
        metric_fwd = metric_fn_auto(mlp_fwd, x=inputs_fwd, target=targets_fwd)
        assert not jnp.isnan(metric_fwd)

        mlp_bwd = PointwiseMLP(key, layers=(4, 8, 2), activations=jax.nn.tanh)
        inputs_bwd = jr.normal(key, (5, 4, 3))
        targets_bwd = jr.normal(key, (5, 2, 3))

        # Trigger bwd mode since in_size (4) > out_size (2)
        metric_bwd = metric_fn_auto(mlp_bwd, x=inputs_bwd, target=targets_bwd)
        assert not jnp.isnan(metric_bwd)

    def test_unsupported_primitive_in_model_raises_key_error(self):
        key = jr.key(0)
        fno = FNO(
            key,
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=(8,),
            n_layers=2,
        )
        no_ins = jr.normal(key, (1, 10))
        coords = jnp.linspace(jnp.array([0.0]), jnp.array([1.0]), 10, axis=1)
        ins = jnp.stack([no_ins, coords], axis=1)
        metric = SobolevMetric(k=1, method="interpolate")
        with pytest.raises(KeyError):
            metric(fno, x=ins, target=no_ins)

    def test_stochastic_sobolev_converges_to_exact_k1(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (1, 2, 4))
        targets = jr.normal(key, (1, 2, 4))
        metric_fn_exact = SobolevMetric(k=1, method="exact")
        metric_fn = SobolevMetric(k=1, method="stochastic", n_random_samples=8)
        exact_metric = metric_fn_exact(mlp, target=targets, x=inputs)
        stoch_metric = metric_fn(mlp, target=targets, x=inputs, key=key)
        assert jnp.allclose(exact_metric, stoch_metric, rtol=1e-3, atol=1e-3)

    def test_stochastic_sobolev_converges_to_exact_k2(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (1, 2, 4))
        targets = jr.normal(key, (1, 2, 4))
        x = inputs.squeeze(0)

        # For this simple model the exact hessian is computable.
        # Sobolev methods only ever use the pure (same-direction-repeated)
        # second partials, i.e. the diagonal of the Hessian over the
        # flattened input, not the full Hessian including mixed partials.
        def hessian(f):
            return jax.jacfwd(jax.jacrev(f))

        H = hessian(mlp)(x)
        n = x.size
        H_diag = jnp.diagonal(H.reshape(targets.shape[1:] + (n, n)), axis1=-2, axis2=-1)
        metric_fn = SobolevMetric(k=2, method="stochastic", n_random_samples=1000)
        stoch_metric = metric_fn(mlp, target=targets, x=inputs, key=key)
        fun_metric = jnp.mean(jnp.pow(jnp.abs(targets - jax.vmap(mlp)(inputs)), 2))

        # Pure second order derivative (diagonal of Hessian)
        second_order_term = jnp.mean(jnp.pow(jnp.abs(H_diag), 2))
        # First order is the Jacobian
        first_order_term = jnp.mean(jnp.pow(jnp.abs(jax.jacfwd(mlp)(x)), 2))
        exact_metric = jnp.sqrt(second_order_term + first_order_term + fun_metric)
        assert jnp.allclose(exact_metric, stoch_metric, rtol=1e-3, atol=1e-3)

    def test_interpolated_sobolev_matches_exact_k1(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (1, 2, 4))
        targets = jr.normal(key, (1, 2, 4))
        interp_metric_fn = SobolevMetric(k=1, method="interpolate")
        exact_metric_fn = SobolevMetric(k=1, method="exact")
        interp_metric = interp_metric_fn(mlp, target=targets, x=inputs)
        exact_metric = exact_metric_fn(mlp, target=targets, x=inputs)
        assert jnp.allclose(exact_metric, interp_metric)

    def test_jittability(self, mlp):
        key = jr.key(0)
        xkey, ykey, lkey = jr.split(key, 3)
        inputs = jr.normal(xkey, (5, 2, 3))
        targets = jr.normal(ykey, (5, 2, 3))

        metric_fn_exact = SobolevMetric(k=1, method="exact")
        metric_fn_interp = SobolevMetric(k=2, method="interpolate")
        metric_fn_stoch = SobolevMetric(k=2, method="stochastic", n_random_samples=2)

        # Wrap metric logic because `assert_filter_jittable` doesn't pass **kwargs natively
        def wrap_exact(m, x, t):
            return metric_fn_exact(model=m, x=x, target=t)

        def wrap_interp(m, x, t):
            return metric_fn_interp(model=m, x=x, target=t)

        def wrap_stoch(m, x, t, k):
            return metric_fn_stoch(model=m, x=x, target=t, key=k)

        assert_filter_jittable(wrap_exact, mlp, inputs, targets)
        assert_filter_jittable(wrap_interp, mlp, inputs, targets)
        assert_filter_jittable(wrap_stoch, mlp, inputs, targets, lkey)


@pytest.mark.usefixtures("mlp")
class TestSobolevMetricAutoDirectionThreshold:
    """Specific tests to check if the call method routing logic works as defined.

    These are quite exhaustive. A routing error almost always causes OOM.
    """

    def test_auto_matches_exact_below_threshold(self, mlp):
        key = jr.key(0)
        # channels=2 (fixed by `mlp`), spatial=1 -> x_i.size = target_i.size = 2
        inputs = jr.normal(key, (5, 2, 1))
        targets = jr.normal(key, (5, 2, 1))

        auto_metric = SobolevMetric(k=1, method="auto", direction_threshold=4)
        exact_metric = SobolevMetric(k=1, method="exact", direction_threshold=4)

        auto_val = auto_metric(mlp, x=inputs, target=targets)
        exact_val = exact_metric(mlp, x=inputs, target=targets)
        assert jnp.allclose(auto_val, exact_val)

    @pytest.mark.parametrize("order", [1, 2])
    def test_auto_matches_stochastic_above_threshold(self, mlp, order):
        key = jr.key(0)
        mkey, lkey = jr.split(key)
        # channels=2, spatial=4 -> x_i.size = target_i.size = 8 > threshold=4
        inputs = jr.normal(mkey, (5, 2, 4))
        targets = jr.normal(mkey, (5, 2, 4))

        auto_metric = SobolevMetric(
            k=order, method="auto", direction_threshold=4, n_random_samples=3
        )
        stochastic_metric = SobolevMetric(
            k=order, method="stochastic", direction_threshold=4, n_random_samples=3
        )

        auto_val = auto_metric(mlp, x=inputs, target=targets, key=lkey)
        stochastic_val = stochastic_metric(mlp, x=inputs, target=targets, key=lkey)
        assert jnp.allclose(auto_val, stochastic_val)

    def test_auto_matches_interpolate_below_threshold(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (5, 2, 1))
        targets = jr.normal(key, (5, 2, 1))

        auto_metric = SobolevMetric(k=2, method="auto", direction_threshold=4)
        interp_metric = SobolevMetric(k=2, method="interpolate", direction_threshold=4)

        auto_val = auto_metric(mlp, x=inputs, target=targets)
        interp_val = interp_metric(mlp, x=inputs, target=targets)
        assert jnp.allclose(auto_val, interp_val)

    def test_auto_boundary_is_inclusive_of_threshold(self, mlp):
        key = jr.key(0)
        # channels=2, spatial=2 -> x_i.size = target_i.size = 4 == threshold
        inputs = jr.normal(key, (5, 2, 2))
        targets = jr.normal(key, (5, 2, 2))

        metric = SobolevMetric(k=1, method="auto", direction_threshold=4)
        # No key passed: would raise if this were (incorrectly) routed to stochastic.
        metric_val = metric(mlp, x=inputs, target=targets)
        assert not jnp.isnan(metric_val)

    def test_auto_just_above_threshold_requires_key(self, mlp):
        key = jr.key(0)
        # channels=2, spatial=3 -> x_i.size = target_i.size = 6 > threshold=4
        inputs = jr.normal(key, (5, 2, 3))
        targets = jr.normal(key, (5, 2, 3))

        metric = SobolevMetric(k=1, method="auto", direction_threshold=4)
        with pytest.raises(ValueError):
            metric(
                mlp, x=inputs, target=targets
            )  # no key -> must fail if routed to stochastic

    def test_auto_k1_uses_min_size_but_k_ge_2_uses_input_size(self):
        key = jr.key(0)
        # in_channels=6 (large), out_channels=2 (small)
        mlp_asym = PointwiseMLP(key, layers=(6, 8, 2), activations=jax.nn.tanh)
        inputs = jr.normal(key, (5, 6, 1))  # x_i.size = 6
        targets = jr.normal(key, (5, 2, 1))  # target_i.size = 2

        threshold = 4
        # k=1: num_directions = min(6, 2) = 2 <= threshold -> "exact" (no key needed)
        k1_metric = SobolevMetric(k=1, method="auto", direction_threshold=threshold)
        k1_val = k1_metric(mlp_asym, x=inputs, target=targets)
        assert not jnp.isnan(k1_val)

        # k=2: num_directions = x.size = 6 > threshold -> "stochastic" (key required)
        k2_metric = SobolevMetric(k=2, method="auto", direction_threshold=threshold)
        with pytest.raises(ValueError):
            k2_metric(mlp_asym, x=inputs, target=targets)
