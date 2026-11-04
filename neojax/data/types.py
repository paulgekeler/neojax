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

"""Shared array-type alias for host/device-agnostic data containers."""

from typing import TYPE_CHECKING

import jax
import numpy as np
from jaxtyping import Inexact

if TYPE_CHECKING:
    pass


JaxNpArray = jax.Array | np.ndarray
"""Base type for jaxtyping shape/dtype annotations that accept either a numpy or jax array.

A sub type of jaxtyping.ArrayLike excluding python built-in types
and numpy scalar-like types (np.bool, np.number).

`DataBundle`, `BundleDataset`, and `RawDataset` sit on the boundary between host-resident
data (plain numpy, as returned by the dataset loading utilities to avoid eagerly placing
a whole dataset on an accelerator) and jax-converted data (once a batch has flowed through
normalization/schemas). Using bare `jaxtyping.Array` (which resolves to `jax.Array`)
or `jaxtyping.ArrayLike` doesn't work. This alias keeps the annotations
meaningful for both.
"""

BundleOrArray = "DataBundle" | Inexact[JaxNpArray, "..."]
"""Base type for Schema.tranform inputs that accept either a DataBundle or inexact JaxNpArray.

This type is shared among most Schemas as input type.

The string literal and TYPE_CHECKING import is needed to avoid circular imports.
"""
