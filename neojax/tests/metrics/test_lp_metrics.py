import jax
import jax.numpy as jnp
import pytest

from neojax.metrics.lp_metrics import LpMetric, RelativeLpMetric
from neojax.tests.conftest import assert_jittable


class TestLpMetric:
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_values(self, p):
        # Test with positive differences
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 4.0, 6.0]]])
        weight = 1.0
        metric_fn = LpMetric(p=p, weight=weight)

        expected_diff = target - pred
        expected_metric = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)

        actual_metric = metric_fn(pred=pred, target=target)
        assert jnp.allclose(actual_metric, expected_metric)

        # Test with mixed differences
        pred = jnp.array([[[3.0, 1.0]]])
        target = jnp.array([[[2.0, 2.0]]])
        # target - pred = [-1.0, 1.0]
        actual_metric = metric_fn(pred=pred, target=target)

        expected_diff = target - pred
        expected_metric = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)
        assert jnp.allclose(actual_metric, expected_metric)

    def test_linf_values(self):
        # Test Linf support (p='inf' and p=float('inf'))
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 5.0, 6.0]]])  # diff is [1.0, 3.0, 3.0]

        metric_inf = LpMetric(p="inf")
        actual_inf = metric_inf(pred=pred, target=target)
        assert jnp.allclose(actual_inf, 3.0)

        metric_inf_float = LpMetric(p=float("inf"))
        actual_inf_float = metric_inf_float(pred=pred, target=target)
        assert jnp.allclose(actual_inf_float, 3.0)

    @pytest.mark.parametrize("shape", [(1, 10), (2, 5, 5), (1, 3, 4, 2)])
    def test_shapes(self, shape):
        p = 2.0
        metric_fn = LpMetric(p=p)
        pred = jax.random.normal(jax.random.PRNGKey(0), shape)
        target = jax.random.normal(jax.random.PRNGKey(1), shape)

        actual_metric = metric_fn(pred=pred, target=target)
        assert actual_metric.shape == ()

    def test_jit(self):
        p = 2.0
        metric_fn = LpMetric(p=p)
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        assert_jittable(lambda p, t: metric_fn(pred=p, target=t), pred, target)

    def test_batching(self):
        p = 2.0
        metric_fn = LpMetric(p=p)
        preds = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        targets = jnp.array([[[2.0, 4.0]], [[6.0, 8.0]]])

        batch_metric = metric_fn(pred=preds, target=targets)

        assert batch_metric.shape == ()

        metric_0 = metric_fn(pred=preds[0:1], target=targets[0:1])
        metric_1 = metric_fn(pred=preds[1:2], target=targets[1:2])
        expected = jnp.mean(jnp.array([metric_0, metric_1]))
        assert jnp.allclose(batch_metric, expected)

    def test_grad(self):
        p = 2.0
        metric_fn = LpMetric(p=p)
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        grad_fn = jax.grad(lambda p, t: metric_fn(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))


class TestRelativeLpMetric:
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_values(self, p):
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 4.0, 6.0]]])
        metric_fn = RelativeLpMetric(p=p)

        expected_diff = target - pred
        expected_lp_norm = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)
        expected_target_norm = jnp.pow(jnp.mean(jnp.pow(jnp.abs(target), p)), 1 / p)
        expected_metric = expected_lp_norm / expected_target_norm

        actual_metric = metric_fn(pred=pred, target=target)
        assert jnp.allclose(actual_metric, expected_metric)

    def test_linf_values(self):
        pred = jnp.array([[[1.0, 2.0, 3.0]]])
        target = jnp.array([[[2.0, 5.0, 6.0]]])  # diff is [1.0, 3.0, 3.0]

        rel_metric_inf = RelativeLpMetric(p="inf")
        actual_rel_inf = rel_metric_inf(pred=pred, target=target)
        assert jnp.allclose(actual_rel_inf, 0.5)

    def test_weight(self):
        p = 2.0
        weight = 3.2
        metric_fn = RelativeLpMetric(p=p, weight=weight)
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        base_metric_fn = RelativeLpMetric(p=p, weight=1.0)
        base_metric = base_metric_fn(pred=pred, target=target)

        actual_metric = metric_fn(pred=pred, target=target)
        assert jnp.allclose(actual_metric, weight * base_metric)

    @pytest.mark.parametrize("shape", [(1, 10), (2, 5, 5)])
    def test_shapes(self, shape):
        p = 2.0
        metric_fn = RelativeLpMetric(p=p)
        pred = jax.random.normal(jax.random.PRNGKey(0), shape)
        target = jax.random.normal(jax.random.PRNGKey(1), shape)

        actual_metric = metric_fn(pred=pred, target=target)
        assert actual_metric.shape == ()

    def test_jit(self):
        p = 2.0
        metric_fn = RelativeLpMetric(p=p)
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        assert_jittable(lambda p, t: metric_fn(pred=p, target=t), pred, target)

    def test_batching(self):
        p = 2.0
        metric_fn = RelativeLpMetric(p=p)
        preds = jnp.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        targets = jnp.array([[[2.0, 4.0]], [[6.0, 8.0]]])

        batch_metric = metric_fn(pred=preds, target=targets)

        assert batch_metric.shape == ()

        metric_0 = metric_fn(pred=preds[0:1], target=targets[0:1])
        metric_1 = metric_fn(pred=preds[1:2], target=targets[1:2])
        expected = jnp.mean(jnp.array([metric_0, metric_1]))
        assert jnp.allclose(batch_metric, expected)

    def test_grad(self):
        p = 2.0
        metric_fn = RelativeLpMetric(p=p)
        pred = jnp.array([[[1.0, 2.0]]])
        target = jnp.array([[[2.0, 4.0]]])

        grad_fn = jax.grad(lambda p, t: metric_fn(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))
