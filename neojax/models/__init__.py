"""Models for neural operators."""

from neojax.models.baseno import BaseNO as BaseNO
from neojax.models.deeponet import DeepONet as DeepONet
from neojax.models.deeponet import MLPDeepONet as MLPDeepONet
from neojax.models.fno import FNO as FNO
from neojax.models.geo_fno import GeoFNO as GeoFNO
from neojax.models.tfno import TFNO as TFNO
from neojax.models.uno import UNO as UNO

__all__ = [
    "DeepONet",
    "MLPDeepONet",
    "FNO",
    "GeoFNO",
    "TFNO",
    "UNO",
    "BaseNO",
]
