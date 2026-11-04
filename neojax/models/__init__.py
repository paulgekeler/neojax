# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for neural operators."""

from neojax.models.baseno import BaseNO as BaseNO
from neojax.models.cno import CNO as CNO
from neojax.models.deeponet import DeepONet as DeepONet
from neojax.models.deeponet import MLPDeepONet as MLPDeepONet
from neojax.models.fno import FNO as FNO
from neojax.models.gaot import GAOT as GAOT
from neojax.models.geo_fno import GeoFNO as GeoFNO
from neojax.models.gino import GiNO as GiNO
from neojax.models.mgno import MgNO as MgNO
from neojax.models.rno import RNO as RNO
from neojax.models.tfno import TFNO as TFNO
from neojax.models.transolver import Transolver as Transolver
from neojax.models.uno import UNO as UNO

__all__ = [
    "DeepONet",
    "MLPDeepONet",
    "FNO",
    "GeoFNO",
    "TFNO",
    "UNO",
    "BaseNO",
    "RNO",
    "MgNO",
    "GiNO",
    "Transolver",
    "CNO",
    "GAOT",
]
