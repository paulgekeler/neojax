import equinox as eqx
import jax
import jax.numpy as jnp

from neojax.metrics.composed_metric import ComposedMetric
from neojax.metrics.lp_metrics import LpMetric, RelativeLpMetric
from neojax.metrics.utils import is_learnable_metric_weight
from neojax.models.fno import FNO


class TestUtils:
    def test_is_learnable_metric_weight_fno(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(8,),
        )

        mask = is_learnable_metric_weight(fno)

        # Verify that all leaves in the mask are False (since FNO is a model, not a metric)
        all_false = all(not leaf for leaf in jax.tree_util.tree_leaves(mask))
        assert all_false

    def test_is_learnable_metric_weight_with_metric(self):
        # lp_metrics.LpMetric inherits from BaseMetric
        metric_fixed = LpMetric(p=2.0, weight=1.5, learnable_weight=False)
        metric_learnable = LpMetric(p=2.0, weight=1.5, learnable_weight=True)

        mask_fixed = is_learnable_metric_weight(metric_fixed)
        # BaseMetric has 'weight' (array) and 'learnable_weight' (static bool)
        # LpMetric also has 'p' (static float)
        assert not mask_fixed.weight

        mask_learnable = is_learnable_metric_weight(metric_learnable)
        assert mask_learnable.weight

    def test_is_learnable_metric_weight_nested_composed_metric(self):
        # Each nested metric's own learnable_weight must be respected
        # independently of the enclosing ComposedMetric's flag.
        learnable_child = RelativeLpMetric(p=2.0, weight=0.9, learnable_weight=True)
        fixed_child = LpMetric(p=2.0, weight=0.1, learnable_weight=False)

        composed = ComposedMetric(learnable_child, fixed_child, learnable_weight=True)

        mask = is_learnable_metric_weight(composed)
        assert mask.weight
        assert mask.metrics[0].weight
        assert not mask.metrics[1].weight

    def test_is_learnable_metric_weight_composed(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(8,),
        )

        class ModelWithMetric(eqx.Module):
            model: FNO
            metric_fn: LpMetric

        model_fixed = ModelWithMetric(
            model=fno,
            metric_fn=LpMetric(p=2.0, weight=1.0, learnable_weight=False),
        )

        mask_fixed = is_learnable_metric_weight(model_fixed)
        assert not mask_fixed.model.lifting.weights[0]
        assert not mask_fixed.metric_fn.weight

        model_learnable = ModelWithMetric(
            model=fno,
            metric_fn=LpMetric(p=2.0, weight=1.0, learnable_weight=True),
        )

        mask_learnable = is_learnable_metric_weight(model_learnable)
        assert not mask_learnable.model.lifting.weights[0]
        assert mask_learnable.metric_fn.weight

    def test_is_learnable_metric_weight_gradient_update(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(4,),
        )
        metric_fn = LpMetric(p=2.0, weight=1.0, learnable_weight=True)

        class ModelWithMetric(eqx.Module):
            model: FNO
            metric_fn: LpMetric

        model_with_metric = ModelWithMetric(model=fno, metric_fn=metric_fn)

        # Combine filter specs into ModelWithMetric structure
        filter_spec = ModelWithMetric(
            model=jax.tree_util.tree_map(eqx.is_inexact_array, model_with_metric.model),
            metric_fn=is_learnable_metric_weight(model_with_metric.metric_fn),
        )

        diff, static = eqx.partition(model_with_metric, filter_spec)

        x = jnp.ones((2, 1, 8))
        y = jnp.zeros((2, 1, 8))

        @jax.value_and_grad
        def grad_metric_fn(diff_args, static_args, x_val, y_val):
            combined = eqx.combine(diff_args, static_args)
            pred = jax.vmap(combined.model)(x_val)
            return combined.metric_fn(pred=pred, target=y_val)

        metric_val, grads = grad_metric_fn(diff, static, x, y)
        assert grads.metric_fn.weight is not None
        assert grads.model.lifting.weights[0] is not None

        # Perform simple one-step gradient update
        updates = jax.tree_util.tree_map(lambda g: -g if g is not None else None, grads)
        new_diff = eqx.apply_updates(diff, updates)

        # Recombine to get the updated model_with_metric
        new_model_with_metric = eqx.combine(new_diff, static)

        assert not jnp.allclose(
            new_model_with_metric.metric_fn.weight, model_with_metric.metric_fn.weight
        )
        assert not jnp.allclose(
            new_model_with_metric.model.lifting.weights[0],
            model_with_metric.model.lifting.weights[0],
        )
