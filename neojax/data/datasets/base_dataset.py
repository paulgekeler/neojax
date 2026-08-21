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
