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

"""Implementation of the base dataset class."""

from abc import abstractmethod
from typing import Any

import equinox as eqx


class BaseDataset(eqx.Module):
    """Abstract base class for neojax datasets.

    Datasets are designed to be compatible with grain's MapDataset,
    meaning they must implement `__len__` and `__getitem__`.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Returns the sample at the given index.

        Args:
            idx: Index of the sample.

        Returns:
            The data sample (e.g. DataBundle or dictionary of arrays).
        """
        ...
