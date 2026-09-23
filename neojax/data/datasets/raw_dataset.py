"""Implementation of the RawDataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from neojax.data.datasets.base_dataset import BaseDataset
from neojax.data.datasets.utils import broadcast_like
from neojax.data.types import JaxNpArray


class RawDataset(BaseDataset):
    """Dataset that yields standard dictionaries of arrays for each sample.

    Useful for when you want to bypass the DataBundle abstraction and
    use grain with raw numpy/jax arrays directly.

    Datasets built from `from_pdegym`/`from_pdebench` hold their arrays as plain
    numpy on the host (see `neojax.data.datasets.utils.load_pdegym_data`), so
    constructing or indexing a `RawDataset` doesn't eagerly place a large
    dataset on an accelerator. Arrays passed in directly may be numpy or jax.

    Args:
        data_dict: Dictionary mapping string keys to arrays.
            It is assumed that the first dimension of all array values
            corresponds to the number of samples (except possibly coordinates, i.e., `coords`).

    ??? info "Internal Attributes"
        * **data_dict** (`dict[str, JaxNpArray]`): Dictionary mapping string keys to arrays.
        * **_num_samples** (`int`): Number of samples.
    """

    data_dict: dict[str, JaxNpArray]
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

    def __getitem__(
        self, idx: int | slice | JaxNpArray | list
    ) -> dict[str, JaxNpArray]:
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
                    sample[k] = broadcast_like(
                        v, (batch_size, *getattr(v, "shape", ()))
                    )
                else:
                    sample[k] = v

        return sample

    def get_batch(
        self,
        idx: int | slice | JaxNpArray | list,
        *,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ) -> dict[str, JaxNpArray]:
        """Fetches a batch like `self[idx]`, optionally committing it to a device.

        Datasets hold their arrays on the host (see `load_pdegym_data`); JAX
        already implicitly (and efficiently) places a host-resident batch on
        the default device the moment it reaches a `jax.jit`-compiled step, so
        `device=None` (the default) needs no extra call here. Pass a `jax.Device`
        to pin a batch to one specific accelerator, or a `jax.sharding.Sharding`
        for full custom multi-device placement — this method doesn't need to
        know or care which, since `jax.device_put` accepts either uniformly.

        Args:
            idx: Index or batch of indices of samples to fetch.
            device: Optional device or sharding to commit the batch to.
                Defaults to `None`, leaving placement to JAX's normal implicit
                behavior.

        Returns:
            Dict of sample(s) at index, optionally committed to `device`.
        """
        batch = self[idx]
        if device is None:
            return batch
        return jax.device_put(batch, device)

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
        from neojax.data.datasets.utils import (
            enforce_consistent_batch_size,
            load_pdebench_data,
            map_dataset_fields,
        )

        raw_data = load_pdebench_data(file_path)

        if field_mapping is None:
            return cls(**raw_data)

        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)
        mapped_data = enforce_consistent_batch_size(mapped_data)
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
        from neojax.data.datasets.utils import (
            enforce_consistent_batch_size,
            load_pdegym_data,
            map_dataset_fields,
        )

        raw_data, _ = load_pdegym_data(file_path)

        if field_mapping is None:
            return cls(**raw_data)

        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)
        mapped_data = enforce_consistent_batch_size(mapped_data)
        return cls(**mapped_data)
