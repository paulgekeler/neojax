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

"""Spectral tensors."""

from neojax.tensor.base_tensor import BaseTensor as BaseTensor
from neojax.tensor.cp_tensor import CPTensor as CPTensor
from neojax.tensor.dense_tensor import DenseTensor as DenseTensor
from neojax.tensor.tt_tensor import TTTensor as TTTensor
from neojax.tensor.tucker_tensor import TuckerTensor as TuckerTensor

__all__ = ["BaseTensor", "CPTensor", "TuckerTensor", "TTTensor", "DenseTensor"]
