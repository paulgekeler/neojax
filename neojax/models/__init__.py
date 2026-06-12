"""Models for neural operators."""

from neojax.models.deeponet import DeepONet
from neojax.models.fno import FNO
from neojax.models.tfno import TFNO

# from neojax.models.otno import OTNO
# from neojax.models.sfno import SFNO
# from neojax.models.uno import UNO

__all__ = [
    "DeepONet",
    "FNO",
    "TFNO",
    # "OTNO",
    # "SFNO",
    # "UNO",
]
