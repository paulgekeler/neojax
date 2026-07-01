import equinox as eqx
import jax
import jax.numpy as jnp

from neojax.losses.lp_losses import LpLoss
from neojax.losses.utils import is_learnable_loss_weight
from neojax.models.fno import FNO


class TestUtils:
    def test_is_learnable_loss_weight_fno(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(8,),
        )

        mask = is_learnable_loss_weight(fno)

        # Verify that all leaves in the mask are False (since FNO is a model, not a loss)
        all_false = all(not leaf for leaf in jax.tree_util.tree_leaves(mask))
        assert all_false

    def test_is_learnable_loss_weight_with_loss(self):
        # lp_losses.LpLoss inherits from BaseLoss
        loss_fixed = LpLoss(p=2.0, weight=1.5, learnable_weight=False)
        loss_learnable = LpLoss(p=2.0, weight=1.5, learnable_weight=True)

        mask_fixed = is_learnable_loss_weight(loss_fixed)
        # BaseLoss has 'weight' (array) and 'learnable_weight' (static bool)
        # LpLoss also has 'p' (static float)
        assert not mask_fixed.weight

        mask_learnable = is_learnable_loss_weight(loss_learnable)
        assert mask_learnable.weight

    def test_is_learnable_loss_weight_composed(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(8,),
        )

        class ModelWithLoss(eqx.Module):
            model: FNO
            loss_fn: LpLoss

        model_fixed = ModelWithLoss(
            model=fno,
            loss_fn=LpLoss(p=2.0, weight=1.0, learnable_weight=False),
        )

        mask_fixed = is_learnable_loss_weight(model_fixed)
        assert not mask_fixed.model.lifting.weights[0]
        assert not mask_fixed.loss_fn.weight

        model_learnable = ModelWithLoss(
            model=fno,
            loss_fn=LpLoss(p=2.0, weight=1.0, learnable_weight=True),
        )

        mask_learnable = is_learnable_loss_weight(model_learnable)
        assert not mask_learnable.model.lifting.weights[0]
        assert mask_learnable.loss_fn.weight

    def test_is_learnable_loss_weight_gradient_update(self):
        fno = FNO(
            key=jax.random.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=1,
            modes=(4,),
        )
        loss_fn = LpLoss(p=2.0, weight=1.0, learnable_weight=True)

        class ModelWithLoss(eqx.Module):
            model: FNO
            loss_fn: LpLoss

        model_with_loss = ModelWithLoss(model=fno, loss_fn=loss_fn)

        # Combine filter specs into ModelWithLoss structure
        filter_spec = ModelWithLoss(
            model=jax.tree_util.tree_map(eqx.is_inexact_array, model_with_loss.model),
            loss_fn=is_learnable_loss_weight(model_with_loss.loss_fn)
        )

        diff, static = eqx.partition(model_with_loss, filter_spec)

        # Fake inputs/targets
        x = jnp.ones((2, 1, 8))
        y = jnp.zeros((2, 1, 8))

        @jax.value_and_grad
        def grad_loss_fn(diff_args, static_args, x_val, y_val):
            combined = eqx.combine(diff_args, static_args)
            pred = jax.vmap(combined.model)(x_val)
            return combined.loss_fn(pred=pred, target=y_val)

        loss_val, grads = grad_loss_fn(diff, static, x, y)
        assert grads.loss_fn.weight is not None
        assert grads.model.lifting.weights[0] is not None

        # Perform simple one-step gradient update
        updates = jax.tree_util.tree_map(lambda g: -g if g is not None else None, grads)
        new_diff = eqx.apply_updates(diff, updates)

        # Recombine to get the updated model_with_loss
        new_model_with_loss = eqx.combine(new_diff, static)

        # Verify weight is updated
        assert not jnp.allclose(new_model_with_loss.loss_fn.weight, model_with_loss.loss_fn.weight)
        assert not jnp.allclose(new_model_with_loss.model.lifting.weights[0], model_with_loss.model.lifting.weights[0])

