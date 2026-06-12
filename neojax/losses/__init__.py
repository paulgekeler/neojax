"""Loss functions for neural operators."""

from neojax.losses.base_loss import BaseLoss
from neojax.losses.composed_loss import ComposedLoss
from neojax.losses.lp_losses import LpLoss, RelativeLpLoss
from neojax.losses.sobolev_losses import SobolevLoss

__all__ = [
    "BaseLoss",
    "ComposedLoss",
    "LpLoss",
    "RelativeLpLoss",
    "SobolevLoss",
]
