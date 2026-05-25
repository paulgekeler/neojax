import jax
import jax.numpy as jnp
import pytest

from neojax.losses.lp_losses import LpLoss, RelativeLpLoss
from neojax.tests.conftest import assert_jittable


class TestLpLoss:
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_values(self, p):
        # Test with positive differences
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([2.0, 4.0, 6.0])
        weight = 1.0
        loss_fn = LpLoss(p=p, weight=weight)

        expected_diff = target - pred
        expected_loss = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)

        actual_loss = loss_fn(pred=pred, target=target)
        assert jnp.allclose(actual_loss, expected_loss)

        # Test with mixed differences
        pred = jnp.array([3.0, 1.0])
        target = jnp.array([2.0, 2.0])
        # target - pred = [-1.0, 1.0]
        actual_loss = loss_fn(pred=pred, target=target)

        expected_diff = target - pred
        expected_loss = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)
        assert jnp.allclose(actual_loss, expected_loss)

    @pytest.mark.parametrize("shape", [(1, 10), (2, 5, 5), (1, 3, 4, 2)])
    def test_shapes(self, shape):
        p = 2.0
        loss_fn = LpLoss(p=p)
        pred = jax.random.normal(jax.random.PRNGKey(0), shape)
        target = jax.random.normal(jax.random.PRNGKey(1), shape)

        actual_loss = loss_fn(pred=pred, target=target)
        assert actual_loss.shape == ()

    def test_jit(self):
        p = 2.0
        loss_fn = LpLoss(p=p)
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([2.0, 4.0])

        assert_jittable(lambda p, t: loss_fn(pred=p, target=t), pred, target)

    def test_vmap(self):
        p = 2.0
        loss_fn = LpLoss(p=p)
        preds = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[2.0, 4.0], [6.0, 8.0]])

        vmapped_loss = jax.vmap(lambda p, t: loss_fn(pred=p, target=t), in_axes=(0, 0))
        actual_losses = vmapped_loss(preds, targets)

        assert actual_losses.shape == (2,)
        assert jnp.allclose(actual_losses[0], loss_fn(pred=preds[0], target=targets[0]))
        assert jnp.allclose(actual_losses[1], loss_fn(pred=preds[1], target=targets[1]))

    def test_grad(self):
        p = 2.0
        loss_fn = LpLoss(p=p)
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([2.0, 4.0])

        grad_fn = jax.grad(lambda p, t: loss_fn(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))


class TestRelativeLpLoss:
    @pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
    def test_values(self, p):
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([2.0, 4.0, 6.0])
        loss_fn = RelativeLpLoss(p=p)

        expected_diff = target - pred
        expected_lp_norm = jnp.pow(jnp.mean(jnp.pow(jnp.abs(expected_diff), p)), 1 / p)
        expected_target_norm = jnp.pow(jnp.mean(jnp.pow(jnp.abs(target), p)), 1 / p)
        expected_loss = expected_lp_norm / expected_target_norm

        actual_loss = loss_fn(pred=pred, target=target)
        assert jnp.allclose(actual_loss, expected_loss)

    def test_weight(self):
        p = 2.0
        weight = 3.2
        loss_fn = RelativeLpLoss(p=p, weight=weight)
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([2.0, 4.0])

        base_loss_fn = RelativeLpLoss(p=p, weight=1.0)
        base_loss = base_loss_fn(pred=pred, target=target)

        actual_loss = loss_fn(pred=pred, target=target)
        assert jnp.allclose(actual_loss, weight * base_loss)

    @pytest.mark.parametrize("shape", [(1, 10), (2, 5, 5)])
    def test_shapes(self, shape):
        p = 2.0
        loss_fn = RelativeLpLoss(p=p)
        pred = jax.random.normal(jax.random.PRNGKey(0), shape)
        target = jax.random.normal(jax.random.PRNGKey(1), shape)

        actual_loss = loss_fn(pred=pred, target=target)
        assert actual_loss.shape == ()

    def test_jit(self):
        p = 2.0
        loss_fn = RelativeLpLoss(p=p)
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([2.0, 4.0])

        assert_jittable(lambda p, t: loss_fn(pred=p, target=t), pred, target)

    def test_vmap(self):
        p = 2.0
        loss_fn = RelativeLpLoss(p=p)
        preds = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[2.0, 4.0], [6.0, 8.0]])

        vmapped_loss = jax.vmap(lambda p, t: loss_fn(pred=p, target=t), in_axes=(0, 0))
        actual_losses = vmapped_loss(preds, targets)

        assert actual_losses.shape == (2,)
        assert jnp.allclose(actual_losses[0], loss_fn(pred=preds[0], target=targets[0]))
        assert jnp.allclose(actual_losses[1], loss_fn(pred=preds[1], target=targets[1]))

    def test_grad(self):
        p = 2.0
        loss_fn = RelativeLpLoss(p=p)
        pred = jnp.array([1.0, 2.0])
        target = jnp.array([2.0, 4.0])

        grad_fn = jax.grad(lambda p, t: loss_fn(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))
