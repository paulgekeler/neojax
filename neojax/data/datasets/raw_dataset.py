"""Implementation of the RawDataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from neojax.data.datasets.base_dataset import BaseDataset


class RawDataset(BaseDataset):
    """Dataset that yields standard dictionaries of arrays for each sample.

    Useful for when you want to bypass the DataBundle abstraction and
    use grain with raw numpy/jax arrays directly.

    Args:
        data_dict: Dictionary mapping string keys to arrays.
            It is assumed that the first dimension of all array values
            corresponds to the number of samples (except possibly coordinates, i.e., `coords`).

    ??? info "Internal Attributes"
        * **data_dict** (`dict[str, Any]`): Dictionary mapping string keys to Array-likes.
        * **_num_samples** (`int`): Number of samples.
    """

    data_dict: dict[str, Array]
    _num_samples: int = eqx.field(static=True)

    def __init__(
        self,
        **data_dict: Any,
    ) -> None:
        self.data_dict = data_dict
        # Assumes the 'fields' key exists or takes the first array to determine length.
        if "fields" in data_dict:
            self._num_samples = data_dict["fields"].shape[0]
        else:
            self._num_samples = next(iter(data_dict.values())).shape[0]

    def __len__(self) -> int:
        """Returns number of samples in dataset."""
        return self._num_samples

    def __getitem__(self, idx: int | slice | jnp.ndarray | list) -> dict[str, Any]:
        """Gets item(s) at index from dataset.

        Args:
            idx: Index or batch of indices of samples to fetch.

        Returns:
            Dict of sample(s) at index. If a batch is fetched, all arrays
            will have a leading batch dimension.
        """
        sample = {}
        # First, process fields to determine if this is a batched request and get batch size
        # We assume the array with shape[0] == _num_samples contains the sample dimension.
        # If we index it, it either drops the dimension (unbatched) or keeps it (batched).
        is_batched = False
        batch_size = None

        # Determine batch status by taking the first batched array we can find
        for v in self.data_dict.values():
            if (
                getattr(v, "shape", None) is not None
                and v.shape[0] == self._num_samples
            ):
                sliced_v = v[idx]
                is_batched = sliced_v.ndim == v.ndim
                if is_batched:
                    batch_size = sliced_v.shape[0]
                break

        for k, v in self.data_dict.items():
            if (
                getattr(v, "shape", None) is not None
                and v.shape[0] == self._num_samples
            ):
                sample[k] = v[idx]
            else:
                if is_batched:
                    sample[k] = jnp.broadcast_to(
                        v, (batch_size, *getattr(v, "shape", ()))
                    )
                else:
                    sample[k] = v

        return sample

    @classmethod
    def from_pdebench(
        cls,
        file_path: str | Path,
        field_mapping: dict[str, str | Sequence[str]] | None = None,
        concat_axes: Sequence[int] | None = None,
    ) -> "RawDataset":
        """Loads a dataset from a PDEBench format file or directory into a raw dictionary.

        Args:
            file_path: Path to the PDEBench dataset file or directory.
            field_mapping: Optional mapping from output keys to one or more HDF5 variable names.
            concat_axes: Optional sequence of axes to concatenate along for each field mapping.

        Returns:
            An instantiated RawDataset.

        Raises:
            ValueError: If dataset is not found or concatenation fails.
        """
        from neojax.data.datasets.utils import load_pdebench_data, map_dataset_fields

        raw_data = load_pdebench_data(file_path)

        if field_mapping is None:
            return cls(**raw_data)

        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)
        return cls(**mapped_data)

    @classmethod
    def from_pdegym(
        cls,
        file_path: str | Path,
        field_mapping: dict[str, str | Sequence[str]] | None = None,
        concat_axes: Sequence[int] | None = None,
    ) -> "RawDataset":
        """Loads a dataset from a PDEGym format file or directory into a raw dictionary.

        Args:
            file_path: Path to the PDEGym dataset file.
            field_mapping: Optional mapping from output keys to one or more NetCDF variable names.
            concat_axes: Optional sequence of axes to concatenate along for each field mapping.

        Returns:
            An instantiated RawDataset.

        Raises:
            ValueError: If dataset is not found or concatenation fails.
        """
        from neojax.data.datasets.utils import load_pdegym_data, map_dataset_fields

        raw_data, _ = load_pdegym_data(file_path)

        if field_mapping is None:
            return cls(**raw_data)

        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)
        return cls(**mapped_data)
