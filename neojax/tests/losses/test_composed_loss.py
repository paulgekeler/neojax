import jax
import jax.numpy as jnp

from neojax.losses import ComposedLoss, LpLoss, RelativeLpLoss
from neojax.tests.conftest import assert_jittable


class TestComposedLoss:
    def test_values(self):
        # Setup data
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        # Setup losses
        loss1 = LpLoss(p=2)  # MSE
        loss2 = RelativeLpLoss(p=2)  # Rel MSE
        weight = 0.5
        composed = ComposedLoss(loss1, loss2, weight=weight)

        # Expected calculation
        val1 = loss1(pred=pred, target=target)
        val2 = loss2(pred=pred, target=target)
        expected = weight * (val1 + val2)

        actual = composed(pred=pred, target=target)
        assert jnp.allclose(actual, expected)

    def test_multiple_losses(self):
        pred = jnp.ones((1, 5))
        target = jnp.ones((1, 5)) * 2.0

        losses = [LpLoss(p=1), LpLoss(p=2), RelativeLpLoss(p=2)]
        composed = ComposedLoss(*losses)

        expected = sum(l(pred=pred, target=target) for l in losses)
        actual = composed(pred=pred, target=target)

        assert jnp.allclose(actual, expected)

    def test_shapes(self):
        shape = (2, 4, 4)
        pred = jnp.ones(shape)
        target = jnp.ones(shape) * 2.0

        composed = ComposedLoss(LpLoss(p=2), RelativeLpLoss(p=1))
        actual = composed(pred=pred, target=target)

        assert actual.shape == ()

    def test_jit(self):
        pred = jnp.array([[1.0, 1.0]])
        target = jnp.array([[2.0, 2.0]])
        composed = ComposedLoss(LpLoss(p=2))

        assert_jittable(lambda p, t: composed(pred=p, target=t), pred, target)

    def test_grad_and_batching(self):
        # test grad
        composed = ComposedLoss(LpLoss(p=2))
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
