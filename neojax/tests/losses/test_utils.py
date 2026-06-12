import equinox as eqx
import jax

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

        # Basic verification that inexact arrays (parameters) are True
        assert mask.lifting.weights[0]
        # Verify that static fields (like bools in sub-modules) are preserved or False
        # In FNOBlock, preactivation is a static bool field
        assert not mask.fno_blocks.fno_layers[0].preactivation

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
        assert mask_fixed.model.lifting.weights[0]
        assert not mask_fixed.loss_fn.weight

        model_learnable = ModelWithLoss(
            model=fno,
            loss_fn=LpLoss(p=2.0, weight=1.0, learnable_weight=True),
        )

        mask_learnable = is_learnable_loss_weight(model_learnable)
        assert mask_learnable.model.lifting.weights[0]
        assert mask_learnable.loss_fn.weight
