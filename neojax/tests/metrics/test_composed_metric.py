import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from neojax.metrics import ComposedMetric, LpMetric, RelativeLpMetric
from neojax.metrics.utils import is_learnable_metric_weight
from neojax.tests.conftest import assert_jittable


@pytest.fixture
def sum_composition():
    return lambda *vals: sum(vals)


class UncertaintyWeighting(eqx.Module):
    """Minimal Kendall & Gal-style stateful composition_fn for testing."""

    log_vars: jax.Array

    def __call__(self, *raw_vals):
        total = jnp.array(0.0)
        for v, s in zip(raw_vals, self.log_vars, strict=True):
            total = total + jnp.exp(-s) * v + s
        return total


class TestComposedMetric:
    def test_values(self, sum_composition):
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        metric1 = LpMetric(p=2)
        metric2 = RelativeLpMetric(p=2)
        weight = 0.5
        composed = ComposedMetric(
            metric1, metric2, weight=weight, composition_fn=sum_composition
        )

        val1 = metric1(pred=pred, target=target)
        val2 = metric2(pred=pred, target=target)
        expected = weight * (val1 + val2)

        actual = composed(pred=pred, target=target)
        assert jnp.allclose(actual, expected)

    def test_multiple_losses(self, sum_composition):
        pred = jnp.ones((1, 5))
        target = jnp.ones((1, 5)) * 2.0

        metrics = [LpMetric(p=1), LpMetric(p=2), RelativeLpMetric(p=2)]
        composed = ComposedMetric(*metrics, composition_fn=sum_composition)

        expected = sum(m(pred=pred, target=target) for m in metrics)
        actual = composed(pred=pred, target=target)

        assert jnp.allclose(actual, expected)

    def test_shapes(self, sum_composition):
        shape = (2, 4, 4)
        pred = jnp.ones(shape)
        target = jnp.ones(shape) * 2.0

        composed = ComposedMetric(
            LpMetric(p=2), RelativeLpMetric(p=1), composition_fn=sum_composition
        )
        actual = composed(pred=pred, target=target)

        assert actual.shape == ()

    def test_jit(self, sum_composition):
        pred = jnp.array([[1.0, 1.0]])
        target = jnp.array([[2.0, 2.0]])
        composed = ComposedMetric(LpMetric(p=2), composition_fn=sum_composition)

        assert_jittable(lambda p, t: composed(pred=p, target=t), pred, target)

    def test_grad_and_batching(self, sum_composition):
        # test grad
        composed = ComposedMetric(LpMetric(p=2), composition_fn=sum_composition)
        pred = jnp.array([[1.0, 2.0]])
        target = jnp.array([[0.0, 0.0]])

        grad_fn = jax.grad(lambda p, t: composed(pred=p, target=t))
        grads = grad_fn(pred, target)
        assert grads.shape == pred.shape
        assert jnp.all(jnp.isfinite(grads))

        # test batching
        preds = jnp.ones((3, 2))
        targets = jnp.ones((3, 2)) * 2.0
        batch_results = composed(pred=preds, target=targets)
        assert batch_results.shape == ()

    def test_composition_fn_is_required(self):
        with pytest.raises(TypeError):
            ComposedMetric(LpMetric(p=2))

    def test_composition_fn_receives_raw_not_weighted_values(self):
        # Each metric's own (non-learnable) weight should have no bearing on
        # what composition_fn sees, it must always be the raw value.
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        metric1 = LpMetric(p=2, weight=3.7)
        metric2 = RelativeLpMetric(p=2, weight=0.2)
        seen = {}

        def capturing_composition(*vals):
            seen["vals"] = vals
            return sum(vals)

        with pytest.warns(UserWarning):
            composed = ComposedMetric(
                metric1, metric2, composition_fn=capturing_composition
            )
        composed(pred=pred, target=target)

        raw1 = metric1(pred=pred, target=target) / metric1.weight
        raw2 = metric2(pred=pred, target=target) / metric2.weight
        assert jnp.allclose(seen["vals"][0], raw1)
        assert jnp.allclose(seen["vals"][1], raw2)

    def test_learnable_child_weight_warns_and_is_inert(self, sum_composition):
        learnable_child = LpMetric(p=2, weight=0.5, learnable_weight=True)
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        with pytest.warns(UserWarning, match="learnable_weight' has no effect"):
            composed = ComposedMetric(learnable_child, composition_fn=sum_composition)

        # The composed child's own weight can never receive a gradient,
        # since composition_fn only ever sees its raw (unweighted) value.
        # The algebraic gradient is zero (weight cancels out of
        # metric(...) / metric.weight identically).
        grad = eqx.filter_grad(lambda c: c(pred=pred, target=target))(composed)
        assert jnp.allclose(grad.metrics[0].raw_weight, 0.0, atol=1e-6)

    def test_fixed_nonunit_child_weight_warns_and_is_inert(self, sum_composition):
        # A fixed (non-learnable) weight is just as inert once composed as
        # a learnable one
        # So it must also warn in the learnable_weight=False case.
        fixed_child = LpMetric(p=2, weight=0.5)
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        with pytest.warns(UserWarning, match="This also applies to fixed weights."):
            composed = ComposedMetric(fixed_child, composition_fn=sum_composition)

        # weight=1.0 (the default) is the true no-op case and must not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ComposedMetric(LpMetric(p=2), composition_fn=sum_composition)

        actual = composed(pred=pred, target=target)
        expected = fixed_child(pred=pred, target=target) / fixed_child.weight
        assert jnp.allclose(actual, expected)

    def test_stateful_composition_fn_receives_real_gradients(self):
        # composition_fn is a regular (non-static) field so a
        # stateful composition_fn (e.g. Kendall & Gal uncertainty weighting)
        # can hold learnable parameters that actually get updated.
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])
        uw = UncertaintyWeighting(log_vars=jnp.zeros(2))
        composed = ComposedMetric(
            LpMetric(p=2), RelativeLpMetric(p=2), composition_fn=uw
        )

        grad = eqx.filter_grad(lambda c: c(pred=pred, target=target))(composed)
        assert not jnp.allclose(grad.composition_fn.log_vars, 0.0)

        # Also confirm it survives a jit boundary without the "static field
        # holds a JAX array" warning equinox raises for the broken version.
        jitted = eqx.filter_jit(lambda c, p, t: c(pred=p, target=t))
        assert jnp.allclose(
            jitted(composed, pred, target), composed(pred=pred, target=target)
        )

    def test_return_components(self, sum_composition):
        pred = jnp.array([[1.0, 2.0, 3.0]])
        target = jnp.array([[2.0, 4.0, 6.0]])

        metric1 = LpMetric(p=2, weight=3.7)
        metric2 = RelativeLpMetric(p=2, weight=0.2)
        with pytest.warns(UserWarning):
            composed = ComposedMetric(metric1, metric2, composition_fn=sum_composition)

        plain_result = composed(pred=pred, target=target)
        result, components = composed(pred=pred, target=target, return_components=True)

        # Same value as a plain call, plus each metric's own raw value.
        assert jnp.allclose(result, plain_result)
        raw1 = metric1(pred=pred, target=target) / metric1.weight
        raw2 = metric2(pred=pred, target=target) / metric2.weight
        assert jnp.allclose(components[0], raw1)
        assert jnp.allclose(components[1], raw2)

    def test_is_learnable_metric_weight_marks_composition_fn_state_trainable(self):
        uw = UncertaintyWeighting(log_vars=jnp.zeros(2))
        # learnable_weight left at its default (False): composition_fn's own
        # state should still be marked trainable, since its learnability is
        # independent of the outer ComposedMetric.weight's own flag.
        composed = ComposedMetric(
            LpMetric(p=2), RelativeLpMetric(p=2), composition_fn=uw
        )

        mask = is_learnable_metric_weight(composed)
        assert jnp.all(mask.composition_fn.log_vars)
        assert not mask.raw_weight
