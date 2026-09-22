import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

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
        # BaseMetric has 'raw_weight' (array) and 'learnable_weight' (static bool)
        # LpMetric also has 'p' (static float)
        assert not mask_fixed.raw_weight

        mask_learnable = is_learnable_metric_weight(metric_learnable)
        assert mask_learnable.raw_weight

    def test_is_learnable_metric_weight_nested_composed_metric(self):
        # Each nested metric's own learnable_weight must be respected
        # independently of the enclosing ComposedMetric's flag.
        learnable_child = RelativeLpMetric(p=2.0, weight=0.9, learnable_weight=True)
        fixed_child = LpMetric(p=2.0, weight=0.1, learnable_weight=False)

        class StatefulComposedMetric(eqx.Module):
            weight_learnable: Float[Array, ""]
            own_weight_learnable: Float[Array, ""]
            # Mark as static to avoid recompilation (shouldn't change but this is clearer)
            weight_fixed: float = eqx.field(static=True)

            def __init__(self, weight_learnable, own_weight_learnable, weight_fixed):
                self.weight_learnable = jnp.asarray(weight_learnable, dtype=jnp.float32)
                self.own_weight_learnable = jnp.asarray(
                    own_weight_learnable, dtype=jnp.float32
                )
                self.weight_fixed = weight_fixed

            def __call__(
                self, *raw_values: tuple[Float[Array, ""]]
            ) -> Float[Array, ""]:
                return self.own_weight_learnable * (
                    self.weight_learnable * raw_values[0]
                    + self.weight_fixed * raw_values[1]
                )

        scm = StatefulComposedMetric(
            weight_learnable=0.9, own_weight_learnable=1.0, weight_fixed=0.1
        )
        with pytest.warns(UserWarning):
            composed = ComposedMetric(
                learnable_child, fixed_child, learnable_weight=True, composition_fn=scm
            )

        # The passed learnable_weight flags should be ignored
        mask = is_learnable_metric_weight(composed)
        assert mask.composition_fn.own_weight_learnable
        assert mask.composition_fn.weight_learnable
        # weight_fixed is not a bool in masked pytree but static float
        assert (
            isinstance(mask.composition_fn.weight_fixed, float)
            and mask.composition_fn.weight_fixed == 0.1
        )
        # raw_weights are still learnable here that is bool=True if learnable_weight=True but will be ignored
        # Should be fixed in the future to be False -> currently somewhat ambiguous despite warning
        assert not mask.metrics[1].raw_weight
        assert mask.metrics[0].raw_weight
        assert mask.raw_weight

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
        assert not mask_fixed.metric_fn.raw_weight

        model_learnable = ModelWithMetric(
            model=fno,
            metric_fn=LpMetric(p=2.0, weight=1.0, learnable_weight=True),
        )

        mask_learnable = is_learnable_metric_weight(model_learnable)
        assert not mask_learnable.model.lifting.weights[0]
        assert mask_learnable.metric_fn.raw_weight

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
