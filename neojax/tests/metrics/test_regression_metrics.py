import jax
import jax.numpy as jnp

from neojax.metrics import MSEMetric, R2Metric, RMSEMetric
from neojax.tests.conftest import assert_jittable


class TestRegressionMetrics:
    def test_mse_values(self):
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 4.0, 6.0]]])
        metric = MSEMetric()

        expected = jnp.mean(jnp.square(target - pred))
        actual = metric(pred=pred, target=target)
        assert jnp.allclose(actual, expected)
        assert actual.shape == ()

    def test_rmse_values(self):
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 4.0, 6.0]]])
        metric = RMSEMetric()

        expected = jnp.sqrt(jnp.mean(jnp.square(target - pred)) + 1e-7)
        actual = metric(pred=pred, target=target)
        assert jnp.allclose(actual, expected)
        assert actual.shape == ()

    def test_r2_values(self):
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 4.0, 6.0]]])
        metric = R2Metric()

        # Target mean = 4.0
        # ss_res = (2-1)^2 + (4-2)^2 + (6-3)^2 = 1 + 4 + 9 = 14
        # ss_tot = (2-4)^2 + (4-4)^2 + (6-4)^2 = 4 + 0 + 4 = 8
        # R2 = 1 - 14/8 = -0.75
        expected = -0.75
        actual = metric(pred=pred, target=target)
        assert jnp.allclose(actual, expected)
        assert actual.shape == ()

    def test_jit_and_grad(self):
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        for metric in [MSEMetric(), RMSEMetric(), R2Metric()]:
            assert_jittable(
                lambda p, t, metric=metric: metric(pred=p, target=t), pred, target
            )

            grad_fn = jax.grad(lambda p, t, metric=metric: metric(pred=p, target=t))
            grads = grad_fn(pred, target)
            assert grads.shape == pred.shape
            assert jnp.all(jnp.isfinite(grads))
