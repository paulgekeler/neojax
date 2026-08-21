import jax
import jax.numpy as jnp

from neojax.metrics import ComposedMetric, LpMetric, RelativeLpMetric
from neojax.tests.conftest import assert_jittable


class TestComposedMetric:
    def test_values(self):
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        metric1 = LpMetric(p=2)
        metric2 = RelativeLpMetric(p=2)
        weight = 0.5
        composed = ComposedMetric(metric1, metric2, weight=weight)

        val1 = metric1(pred=pred, target=target)
        val2 = metric2(pred=pred, target=target)
        expected = weight * (val1 + val2)

        actual = composed(pred=pred, target=target)
        assert jnp.allclose(actual, expected)

    def test_multiple_losses(self):
        pred = jnp.ones((1, 5))
        target = jnp.ones((1, 5)) * 2.0

        metrics = [LpMetric(p=1), LpMetric(p=2), RelativeLpMetric(p=2)]
        composed = ComposedMetric(*metrics)

        expected = sum(m(pred=pred, target=target) for m in metrics)
        actual = composed(pred=pred, target=target)

        assert jnp.allclose(actual, expected)

    def test_shapes(self):
        shape = (2, 4, 4)
        pred = jnp.ones(shape)
        target = jnp.ones(shape) * 2.0

        composed = ComposedMetric(LpMetric(p=2), RelativeLpMetric(p=1))
        actual = composed(pred=pred, target=target)

        assert actual.shape == ()

    def test_jit(self):
        pred = jnp.array([[1.0, 1.0]])
        target = jnp.array([[2.0, 2.0]])
        composed = ComposedMetric(LpMetric(p=2))

        assert_jittable(lambda p, t: composed(pred=p, target=t), pred, target)

    def test_grad_and_batching(self):
        # test grad
        composed = ComposedMetric(LpMetric(p=2))
        pred = jnp.array([[1.0, 2.0]])
        target = jnp.array([[0.0, 0.0]])

        grad_fn = jax.grad(lambda p, t: composed(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))

        # test batching automatically handled
        preds = jnp.ones((3, 2))
        targets = jnp.ones((3, 2)) * 2.0
        batch_results = composed(pred=preds, target=targets)
        assert batch_results.shape == ()
