"""Models for neural operators."""

from neojax.models.baseno import BaseNO
from neojax.models.deeponet import DeepONet
from neojax.models.fno import FNO
from neojax.models.tfno import TFNO

__all__ = [
    "DeepONet",
    "FNO",
    "TFNO",
    "BaseNO",
]
