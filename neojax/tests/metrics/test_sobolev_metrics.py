import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import TypeCheckError

from neojax.metrics.sobolev_metrics import SobolevMetric
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

        with pytest.raises((ValueError, TypeCheckError)):
            SobolevMetric(random_type="invalid")

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

    def test_stochastic_sobolev(self, mlp):
        key = jr.key(0)
        mkey, xkey, ykey, lkey = jr.split(key, 4)
        inputs = jr.normal(xkey, (10, 2, 4))
        targets = jr.normal(ykey, (10, 2, 4))

        metric_fn = SobolevMetric(k=2, method="stochastic", n_random_samples=2)
        metric_val = metric_fn(mlp, x=inputs, target=targets, key=lkey)

        assert metric_val.shape == ()
        assert not jnp.isnan(metric_val)

    def test_missing_key_stochastic(self, mlp):
        key = jr.key(0)
        inputs = jr.normal(key, (10, 2, 4))
        targets = jr.normal(key, (10, 2, 4))

        metric_fn = SobolevMetric(k=2, method="stochastic")
        with pytest.raises(ValueError):
            metric_fn(mlp, x=inputs, target=targets)

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

    def test_jittability(self, mlp):
        key = jr.key(0)
        mkey, xkey, ykey, lkey = jr.split(key, 4)
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

    def test_auto_matches_stochastic_above_threshold(self, mlp):
        key = jr.key(0)
        mkey, lkey = jr.split(key)
        # channels=2, spatial=4 -> x_i.size = target_i.size = 8 > threshold=4
        inputs = jr.normal(mkey, (5, 2, 4))
        targets = jr.normal(mkey, (5, 2, 4))

        auto_metric = SobolevMetric(
            k=1, method="auto", direction_threshold=4, n_random_samples=3
        )
        stochastic_metric = SobolevMetric(
            k=1, method="stochastic", direction_threshold=4, n_random_samples=3
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
