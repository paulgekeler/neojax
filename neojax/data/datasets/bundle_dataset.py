"""Implementation of the BundleDataset."""

from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
from jaxtyping import Float, Int, Real

from neojax.data.array_types import JaxNpArray
from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.datasets.base_dataset import BaseDataset
from neojax.data.datasets.utils import broadcast_like


class BundleDataset(BaseDataset):
    """Unified dataset that yields DataBundles for each sample.

    Datasets built from `from_pdegym`/`from_pdebench` hold their arrays as plain
    numpy on the host (see `neojax.data.datasets.utils.load_pdegym_data`), so
    constructing or indexing a `BundleDataset` doesn't eagerly place a large
    dataset on an accelerator. Arrays passed in directly may be numpy or jax.

    Args:
        coords: Array of grid or mesh coordinates.
        fields: Array of physical fields (e.g., [num_samples, time, channels, *spatial]).
        parameters: Optional parameters array (e.g., [num_samples, ...]).
        bc_masks: Optional boundary condition masks.
        bc_values: Optional boundary condition values.
        edge_indices: Optional edge connectivity map for meshes.

    ??? info "Internal Attributes"
        * **coords** (`Real[JaxNpArray, "d *spatial"]`):
        * **fields** (`Float[JaxNpArray, "b t c *spatial"]`):
        * **parameters** (`Float[JaxNpArray, "b ..."] | None`):
        * **bc_masks** (`Int[JaxNpArray, "c_bc *spatial"] | None`):
        * **bc_values** (`Float[JaxNpArray, "b t c_bc *spatial"] | None`):
        * **edge_indices** (`Int[JaxNpArray, "2 e"] | None`):
        * **_num_samples** (`int`):
    """

    coords: Real[JaxNpArray, "d *spatial"]
    fields: Float[JaxNpArray, "b t c *spatial"]
    parameters: Float[JaxNpArray, "b ..."] | None
    bc_masks: Int[JaxNpArray, "c_bc *spatial"] | None
    bc_values: Float[JaxNpArray, "b t c_bc *spatial"] | None
    edge_indices: Int[JaxNpArray, "2 e"] | None
    _num_samples: int = eqx.field(static=True)

    def __init__(
        self,
        coords: Real[JaxNpArray, "d *spatial"],
        fields: Float[JaxNpArray, "b t c *spatial"],
        parameters: Float[JaxNpArray, "b ..."] | None = None,
        bc_masks: Int[JaxNpArray, "c_bc *spatial"] | None = None,
        bc_values: Float[JaxNpArray, "b t c_bc *spatial"] | None = None,
        edge_indices: Int[JaxNpArray, "2 e"] | None = None,
    ) -> None:
        self.coords = coords
        self.fields = fields
        self.parameters = parameters
        self.bc_masks = bc_masks
        self.bc_values = bc_values
        self.edge_indices = edge_indices
        self._num_samples = fields.shape[0]

    def __len__(self) -> int:
        """Returns number of samples in dataset."""
        return self._num_samples

    def __getitem__(self, idx: int | slice | JaxNpArray | list) -> DataBundle:
        """Gets item(s) at index from dataset.

        Args:
            idx: Index or batch of indices of samples to fetch.

        Returns:
            DataBundle of sample(s) at index. If a batch is fetched, all fields
            will have a leading batch dimension.
        """
        fields = self.fields[idx]

        # Check if idx represents a batch by looking at the dimension of the resulting fields.
        # fields has shape [b, t, c, *spatial] vs unbatched [t, c, *spatial].
        is_batched = fields.ndim == self.fields.ndim

        coords = self.coords
        bc_masks = self.bc_masks
        edge_indices = self.edge_indices

        if is_batched:
            batch_size = fields.shape[0]
            coords = broadcast_like(coords, (batch_size, *coords.shape))
            if bc_masks is not None:
                bc_masks = broadcast_like(bc_masks, (batch_size, *bc_masks.shape))
            if edge_indices is not None:
                edge_indices = broadcast_like(
                    edge_indices, (batch_size, *edge_indices.shape)
                )

        return DataBundle(
            coords=coords,
            fields=fields,
            parameters=self.parameters[idx] if self.parameters is not None else None,
            bc_masks=bc_masks,
            bc_values=self.bc_values[idx] if self.bc_values is not None else None,
            edge_indices=edge_indices,
        )

    def get_batch(
        self,
        idx: int | slice | JaxNpArray | list,
        *,
        device: jax.Device | jax.sharding.Sharding | None = None,
    ) -> DataBundle:
        """Fetches a batch like `self[idx]`, optionally committing it to a device.

        Datasets hold their arrays on the host (see `load_pdegym_data`); JAX
        already implicitly (and efficiently) places a host-resident batch on
        the default device the moment it reaches a `jax.jit`-compiled step, so
        `device=None` (the default) needs no extra call here. Pass a `jax.Device`
        to pin a batch to one specific accelerator, or a `jax.sharding.Sharding`
        for full custom multi-device placement.

        Args:
            idx: Index or batch of indices of samples to fetch.
            device: Optional device or sharding to commit the batch to.
                Defaults to `None`, leaving placement to JAX's normal implicit
                behavior.

        Returns:
            DataBundle of sample(s) at index, optionally committed to `device`.
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
    ) -> "BundleDataset":
        """Loads a dataset from a PDEBench format file or directory.

        Args:
            file_path: Path to the PDEBench dataset file or directory.
            field_mapping: Dictionary mapping DataBundle keys to HDF5 variables. Required.
            concat_axes: Optional sequence of axes to concatenate along for each mapping.

        Returns:
            An instantiated BundleDataset.

        Raises:
            ValueError: If dataset is not found, field_mapping is missing, or keys are invalid.
        """
        from neojax.data.datasets.utils import (
            enforce_consistent_batch_size,
            load_pdebench_data,
            map_dataset_fields,
        )

        if field_mapping is None:
            raise ValueError("field_mapping must be provided for BundleDataset.")

        valid_keys = {
            "coords",
            "fields",
            "parameters",
            "bc_masks",
            "bc_values",
            "edge_indices",
        }
        for k in field_mapping:
            if k not in valid_keys:
                raise ValueError(
                    f"Invalid DataBundle key: '{k}'. Valid keys are: {valid_keys}"
                )

        raw_data = load_pdebench_data(file_path)
        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)
        mapped_data = enforce_consistent_batch_size(mapped_data)
        return cls(**mapped_data)

    @classmethod
    def from_pdegym(
        cls,
        file_path: str | Path,
        field_mapping: dict[str, str | Sequence[str]] | None = None,
        concat_axes: Sequence[int] | None = None,
        has_time_dim: bool | None = None,
        has_channel_dim: bool | None = None,
    ) -> "BundleDataset":
        """Loads a dataset from a PDEGym format file or directory.

        PDEGym datasets vary in structure across problems: some have no
        explicit time or channel axis on their field variable, and none of
        them ship an explicit coordinate variable (the domain is implicitly
        an evenly spaced unit hypercube). This method normalizes both:

        - If `field_mapping` has no `'coords'` entry, coordinates are
          generated automatically with `create_default_coords`, as an evenly
          spaced grid over the unit hypercube `[0, 1]^d` matching the
          spatial shape of `fields`. A warning is raised when this happens.
        - The variable mapped to `'fields'` is expanded with singleton time
          and/or channel axes if it doesn't already have them, so the result
          always matches `DataBundle`'s `[b, t, c, *spatial]` layout.

        Args:
            file_path: Path to the PDEGym dataset file or directory.
            field_mapping: Dictionary mapping DataBundle keys to NetCDF/HDF5
                variables. Required. The `'coords'` key may be omitted.
            concat_axes: Optional sequence of axes to concatenate along for each mapping.
            has_time_dim: Whether the variable mapped to `'fields'` already has
                a time axis. For `.nc` sources this is inferred from the
                variable's `'time'` dimension name and this argument is
                ignored; it is only used for `.h5`/`.hdf5` sources (which
                carry no dimension names), where it defaults to `True` (no
                axis inserted) if not given.
            has_channel_dim: Whether the variable mapped to `'fields'` already
                has a channel axis. Same inference/default rules as
                `has_time_dim`.

        Returns:
            An instantiated BundleDataset.

        Raises:
            ValueError: If dataset is not found, field_mapping is missing, or keys are invalid.
        """
        import warnings

        from neojax.data.datasets.utils import (
            create_default_coords,
            enforce_consistent_batch_size,
            ensure_time_and_channel_axes,
            load_pdegym_data,
            map_dataset_fields,
        )

        if field_mapping is None:
            raise ValueError("field_mapping must be provided for BundleDataset.")

        valid_keys = {
            "coords",
            "fields",
            "parameters",
            "bc_masks",
            "bc_values",
            "edge_indices",
        }
        for k in field_mapping:
            if k not in valid_keys:
                raise ValueError(
                    f"Invalid DataBundle key: '{k}'. Valid keys are: {valid_keys}"
                )

        raw_data, var_dims = load_pdegym_data(file_path)
        mapped_data = map_dataset_fields(raw_data, field_mapping, concat_axes)

        if "fields" in mapped_data:
            fields_vars = field_mapping["fields"]
            fields_var_name = (
                fields_vars if isinstance(fields_vars, str) else fields_vars[0]
            )
            dims = var_dims.get(fields_var_name)
            if dims is not None:
                resolved_has_time_dim = "time" in dims
                resolved_has_channel_dim = "channel" in dims
            else:
                resolved_has_time_dim = (
                    has_time_dim if has_time_dim is not None else True
                )
                resolved_has_channel_dim = (
                    has_channel_dim if has_channel_dim is not None else True
                )

            mapped_data["fields"] = ensure_time_and_channel_axes(
                mapped_data["fields"], resolved_has_time_dim, resolved_has_channel_dim
            )

            if mapped_data["fields"].ndim <= 3:
                raise ValueError(
                    f"'{fields_var_name}' resolved to a fields array with no "
                    f"remaining spatial dimensions (shape "
                    f"{mapped_data['fields'].shape} after inserting time/channel "
                    "axes). This dataset's source variable has no dimension "
                    "names to infer from; pass 'has_time_dim' and "
                    "'has_channel_dim' explicitly to describe its actual shape."
                )

            if "coords" not in mapped_data:
                spatial_shape = mapped_data["fields"].shape[3:]
                warnings.warn(
                    "No 'coords' entry in field_mapping; generating default "
                    f"coordinates as an evenly spaced grid over the unit "
                    f"hypercube [0, 1]^{len(spatial_shape)} with spatial "
                    f"shape {spatial_shape}. Pass an explicit 'coords' "
                    "mapping to use the dataset's own physical domain.",
                    stacklevel=2,
                )
                mapped_data["coords"] = create_default_coords(
                    spatial_shape, [(0.0, 1.0)] * len(spatial_shape)
                )

        mapped_data = enforce_consistent_batch_size(mapped_data)
        return cls(**mapped_data)
