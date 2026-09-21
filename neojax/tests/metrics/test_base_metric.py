import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from neojax.metrics.lp_metrics import LpMetric


class TestBaseMetricWeightReparameterization:
    """LpMetric is used as a concrete stand-in for BaseMetric (abstract)."""

    def test_fixed_weight_is_unchanged(self):
        metric = LpMetric(p=2.0, weight=0.9, learnable_weight=False)
        assert jnp.allclose(metric.weight, 0.9)
        assert jnp.allclose(metric.raw_weight, 0.9)

    def test_learnable_weight_round_trips_at_construction(self):
        for w in (0.1, 0.9, 1.0, 5.0):
            metric = LpMetric(p=2.0, weight=w, learnable_weight=True)
            assert jnp.allclose(metric.weight, w, atol=1e-5)
            # The stored leaf is the pre-softplus value, not the weight itself.
            assert not jnp.allclose(metric.raw_weight, w, atol=1e-3)

    def test_learnable_weight_stays_positive_after_large_negative_update(self):
        # Simulate many steps of gradient descent having driven raw_weight far negative
        metric = LpMetric(p=2.0, weight=0.1, learnable_weight=True)
        adversarial = eqx.tree_at(lambda m: m.raw_weight, metric, jnp.array(-50.0))
        assert adversarial.weight > 0
        assert jnp.isfinite(adversarial.weight)

    def test_learnable_weight_requires_positive_initial_value(self):
        with pytest.raises(ValueError, match="strictly positive"):
            LpMetric(p=2.0, weight=0.0, learnable_weight=True)

        with pytest.raises(ValueError, match="strictly positive"):
            LpMetric(p=2.0, weight=-0.5, learnable_weight=True)

    def test_non_learnable_weight_allows_non_positive_value(self):
        # No gradient ever touches this fixed weight, stays the same
        metric = LpMetric(p=2.0, weight=-0.5, learnable_weight=False)
        assert jnp.allclose(metric.weight, -0.5)

    def test_raw_weight_is_the_only_array_leaf(self):
        metric = LpMetric(p=2.0, weight=0.9, learnable_weight=True)
        leaves = jax.tree_util.tree_leaves(metric)
        assert len(leaves) == 1
        assert jnp.allclose(leaves[0], metric.raw_weight)
