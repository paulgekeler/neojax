"""Utility modules for Neojax."""

from neojax.utils.torch_converters import (
    load_torch_weights_into_fno,
    load_torch_weights_into_geo_fno,
)

__all__ = [
    "load_torch_weights_into_fno",
    "load_torch_weights_into_geo_fno",
]
