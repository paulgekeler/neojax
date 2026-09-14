"""Utility functions for dataset loading."""

import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Inexact, Real

from neojax.data.array_types import JaxNpArray


def broadcast_like(array: JaxNpArray, shape: Sequence[int]) -> JaxNpArray:
    """Broadcasts `array` to `shape`, staying in whichever array library it's already in.

    Datasets are host-resident numpy arrays by default (see `load_pdegym_data`),
    but callers may construct one from jax arrays directly. Using `np.broadcast_to`
    unconditionally silently pulls a jax array back to the host. This picks
    the matching broadcast implementation instead, so a dataset's array library
    choice is preserved rather than overridden.

    Args:
        array: The array to broadcast. May be a numpy or jax array.
        shape: Target shape.

    Returns:
        `array` broadcast to `shape`, in the same array library as the input.
    """
    xp = jnp if isinstance(array, jax.Array) else np
    return xp.broadcast_to(array, shape)


def _natural_sort_key(path: Path) -> list[int | str]:
    """Sort key for file names that orders embedded integers numerically (e.g. '_2' before '_10').

    PDEGym files need to be sorted by numerical order.

    Args:
        path: The path to sort.

    Returns:
        Sorted integers or strings.
    """
    return [
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)
    ]


def load_pdegym_data(
    file_path: str | Path, output_file: str | Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Loads PDEGym data from a single .nc/.h5 file or a directory of files.

    This function is largely adapted from the data assembly logic in the
    camlab-ethz/pdegym repository.

    Args:
        file_path: Path to a .nc/.h5 file or a directory containing such files.
        output_file: Optional path to save the aggregated dataset to a new file
            with respective suffix and filetype.

    Returns:
        A tuple of:

        - Dictionary mapping variable names to their concatenated numpy arrays.
          Arrays stay on the host (not moved to a JAX device) so that loading
          a large dataset doesn't eagerly consume accelerator memory. They are
          only placed on-device once a batch is actually drawn.
        - Dictionary mapping variable names to their NetCDF dimension names
          (e.g. `('sample', 'time', 'channel', 'x', 'y')`). Only populated for
          variables read from `.nc` files. HDF5 datasets carry no dimension
          names, so variables read from `.h5`/`.hdf5` files are absent from
          this dictionary.

    Raises:
        ValueError: If the `file_path` is neither a file nor a directory,
            or if no `.nc`/`.h5` files are found in the given directory.
    """
    import h5py
    import netCDF4 as nc

    path = Path(file_path)

    if path.is_file() and path.suffix == ".nc":
        nc_files = [path]
        h5_files = []
    elif path.is_file() and path.suffix in (".h5", ".hdf5"):
        nc_files = []
        h5_files = [path]
    elif path.is_dir():
        nc_files = sorted(
            (path / f for f in os.listdir(path) if f.endswith(".nc")),
            key=_natural_sort_key,
        )
        h5_files = sorted(
            (
                path / f
                for f in os.listdir(path)
                if f.endswith((".h5", ".hdf5")) and not f.endswith(".part")
            ),
            key=_natural_sort_key,
        )
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory.")

    if not nc_files and not h5_files:
        raise ValueError(f"No .nc or .h5/.hdf5 files found in {path}")

    np_data_dict = {}
    var_dims = {}
    var_datatypes = {}
    var_attrs = {}

    if nc_files:
        var_chunks: dict[str, list[np.ndarray]] = {}

        for nc_file in nc_files:
            with nc.Dataset(nc_file, "r") as ds:
                for var_name, var in ds.variables.items():
                    if var_name not in var_chunks:
                        var_chunks[var_name] = []
                        var_dims[var_name] = var.dimensions
                        var_datatypes[var_name] = var.datatype
                        var_attrs[var_name] = {
                            k: var.getncattr(k) for k in var.ncattrs()
                        }
                    var_chunks[var_name].append(var[:])

        for var_name, chunks in var_chunks.items():
            dims = var_dims[var_name]
            if len(chunks) == 1:
                concat_arr = chunks[0]
            else:
                if "sample" in dims or (dims and dims[0] == "sample"):
                    concat_arr = np.concatenate(chunks, axis=0)
                else:
                    concat_arr = chunks[0]

            np_data_dict[var_name] = concat_arr

    elif h5_files:
        import re

        has_sample_groups = False
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                if any(
                    isinstance(f[k], h5py.Group) and k.startswith("Sample_")
                    for k in f.keys()
                ):
                    has_sample_groups = True
                    break

        if has_sample_groups:
            var_samples: dict[str, list[np.ndarray]] = {}
            for h5_file in h5_files:
                with h5py.File(h5_file, "r") as f:
                    sample_keys = [
                        k
                        for k in f.keys()
                        if isinstance(f[k], h5py.Group) and k.startswith("Sample_")
                    ]

                    def _get_sample_idx(k: str) -> int | str:
                        match = re.search(r"\d+", k)
                        return int(match.group()) if match else k

                    sample_keys.sort(key=_get_sample_idx)

                    for sk in sample_keys:
                        grp = f[sk]
                        for vk in grp.keys():
                            if isinstance(grp[vk], h5py.Dataset):
                                if vk not in var_samples:
                                    var_samples[vk] = []
                                var_samples[vk].append(grp[vk][()])

            for vk, arr_list in var_samples.items():
                if len(arr_list) > 1 and all(
                    np.array_equal(arr_list[0], arr) for arr in arr_list[1:]
                ):
                    concat_arr = arr_list[0]
                else:
                    concat_arr = np.stack(arr_list, axis=0)
                np_data_dict[vk] = concat_arr
        else:
            var_chunks_h5: dict[str, list[np.ndarray]] = {}
            for h5_file in h5_files:
                with h5py.File(h5_file, "r") as f:
                    for vk in f.keys():
                        if isinstance(f[vk], h5py.Dataset):
                            if vk not in var_chunks_h5:
                                var_chunks_h5[vk] = []
                            var_chunks_h5[vk].append(f[vk][:])

            for vk, chunks in var_chunks_h5.items():
                if len(chunks) == 1:
                    concat_arr = chunks[0]
                else:
                    if all(np.array_equal(chunks[0], c) for c in chunks[1:]):
                        concat_arr = chunks[0]
                    else:
                        concat_arr = np.concatenate(chunks, axis=0)

                np_data_dict[vk] = concat_arr

    if output_file is not None:
        out_path = Path(output_file)
        if out_path.suffix in (".h5", ".hdf5"):
            with h5py.File(out_path, "w") as out_h5:
                for var_name, arr in np_data_dict.items():
                    out_h5.create_dataset(var_name, data=arr)
        else:
            with nc.Dataset(out_path, "w") as out_nc:
                for var_name, arr in np_data_dict.items():
                    dims = var_dims.get(var_name)
                    if dims is None or len(dims) != arr.ndim:
                        dims = tuple(f"dim_{var_name}_{i}" for i in range(arr.ndim))
                    for dname, dsize in zip(dims, arr.shape, strict=True):
                        if dname not in out_nc.dimensions:
                            out_nc.createDimension(dname, dsize)
                    dtype = var_datatypes.get(var_name, arr.dtype)
                    out_var = out_nc.createVariable(var_name, dtype, dims)
                    if var_name in var_attrs:
                        out_var.setncatts(var_attrs[var_name])
                    out_var[:] = arr
        print(f"Saved aggregated data to {output_file}")

    return np_data_dict, var_dims


def load_pdebench_data(
    file_path: str | Path, output_file: str | Path | None = None
) -> dict[str, np.ndarray]:
    """Loads PDEBench data from a single .hdf5 file or directory of .hdf5 files.

    PDEBench files have a trailing channel dimension, shape (batch, time, x1, ..., xd, channel)
    where x1, ..., xd are the spatial dimensions.

    Args:
        file_path: Path to a .hdf5 file or a directory containing .hdf5 files.
        output_file: Optional path to save the aggregated dataset to a new .hdf5 file.

    Returns:
        Dictionary mapping variable names to their concatenated numpy arrays.
        Arrays stay on the host (not moved to a JAX device) so that loading a
        large dataset doesn't eagerly consume accelerator memory; they are
        only placed on-device once a batch is actually drawn.

    Raises:
        ValueError: If no files are found or the path is invalid.
    """
    import h5py

    path = Path(file_path)

    if path.is_file():
        hdf5_files = [path]
    elif path.is_dir():
        hdf5_files = [
            path / f for f in os.listdir(path) if f.endswith((".h5", ".hdf5"))
        ]
        hdf5_files.sort()
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory.")

    if not hdf5_files:
        raise ValueError(f"No .h5/.hdf5 files found in {path}")

    # Read the first file to determine the variables and dimensions
    data_dict = {}
    with h5py.File(hdf5_files[0], "r") as first_h5:
        for var_name in first_h5.keys():
            if isinstance(first_h5[var_name], h5py.Dataset):
                data_dict[var_name] = [first_h5[var_name][:]]

    # Read the rest of the files
    for hdf5_file in hdf5_files[1:]:
        with h5py.File(hdf5_file, "r") as ds:
            for var_name in data_dict.keys():
                if var_name in ds and isinstance(ds[var_name], h5py.Dataset):
                    data_dict[var_name].append(ds[var_name][:])

    np_data_dict = {}

    for var_name, var_list in data_dict.items():
        if len(var_list) > 1:
            # Check if array is identical across files (usually coords)
            if np.array_equal(var_list[0], var_list[1]):
                concat_arr = var_list[0]
            else:
                concat_arr = np.concatenate(var_list, axis=0)
        else:
            concat_arr = var_list[0]

        # PDEBench: move trailing channel dimension to axis 2 for main field tensors.
        # Heuristic: if ndim >= 3, it represents [batch, time, ..., channel]
        if concat_arr.ndim >= 3:
            concat_arr = np.moveaxis(concat_arr, -1, 2)

        np_data_dict[var_name] = concat_arr

    if output_file is not None:
        with h5py.File(output_file, "w") as out_h5:
            for var_name, arr in np_data_dict.items():
                out_h5.create_dataset(var_name, data=arr)
        print(f"Saved aggregated data to {output_file}")

    return np_data_dict


def map_dataset_fields(
    raw_data: dict[str, np.ndarray],
    field_mapping: dict[str, str | Sequence[str]],
    concat_axes: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Applies field mapping and concatenation to a raw dataset dictionary.

    Args:
        raw_data: Dictionary mapping variable names to their arrays.
        field_mapping: Dictionary mapping output DataBundle keys to input variable names.
        concat_axes: Optional sequence of axes to concatenate along for each mapping.

    Returns:
        Mapped and concatenated dictionary.

    Raises:
        ValueError: If `field_mapping` and `concat_axes` have different lengths,
            if a variable in `field_mapping` is not found in `raw_data`, or if
            arrays for the same output key have incompatible shapes for concatenation.
    """
    if concat_axes is not None and len(field_mapping) != len(concat_axes):
        raise ValueError("Length of field_mapping and concat_axes must be equal.")

    data_dict = {}
    for i, (out_key, variables) in enumerate(field_mapping.items()):
        if isinstance(variables, str):
            variables = [variables]

        arrays = []
        for var in variables:
            if var not in raw_data:
                raise ValueError(
                    f"Variable '{var}' not found in dataset. Available: {list(raw_data.keys())}"
                )
            arrays.append(raw_data[var])

        if len(arrays) == 1:
            data_dict[out_key] = arrays[0]
        else:
            axis = concat_axes[i] if concat_axes is not None else 2

            base_shape = list(arrays[0].shape)
            try:
                base_shape.pop(axis)
            except IndexError as ie:
                raise ValueError(
                    f"Axis {axis} is out of bounds for array of shape {arrays[0].shape}"
                ) from ie

            for arr in arrays[1:]:
                arr_shape = list(arr.shape)
                try:
                    arr_shape.pop(axis)
                except IndexError as ie:
                    raise ValueError(
                        f"Axis {axis} is out of bounds for array of shape {arr.shape}"
                    ) from ie

                if arr_shape != base_shape:
                    raise ValueError(
                        f"Variables for '{out_key}' cannot be concatenated along axis {axis}. "
                        f"Shapes: {arrays[0].shape} and {arr.shape}"
                    )

            data_dict[out_key] = np.concatenate(arrays, axis=axis)

    return data_dict


def enforce_consistent_batch_size(
    mapped_data: dict[str, np.ndarray],
    batch_keys: Sequence[str] = ("fields", "parameters", "bc_values"),
) -> dict[str, np.ndarray]:
    """Truncates batch-carrying arrays to match `fields`' sample count.

    A dataset assembled from independently-chunked source files can end up with
    mismatched per-attribute sample counts without any error being raised, e.g.
    downloading every `c_*.nc` (parameters) file for `wave_layer` but only some
    of the `solution_*.nc` (fields) files: `parameters` would then have more
    samples than `fields`, and nothing catches it before it reaches training.

    `fields` (always required) is treated as authoritative for the batch size.
    Any other key in `batch_keys` present in `mapped_data` with *more* samples
    is truncated down to match `fields`, with a warning naming what happened.
    A key with *fewer* samples than `fields` raises instead of truncating
    `fields` itself, since there's no direction to safely truncate `fields`
    without silently discarding field data the caller presumably wanted.

    `coords`, `bc_masks`, and `edge_indices` are not batch-carrying by
    `DataBundle`'s convention (no leading `#b` axis) and are never touched here.

    Args:
        mapped_data: Dict of arrays as returned by `map_dataset_fields`.
        batch_keys: Which keys, if present, carry a per-sample batch axis.
            Defaults to `("fields", "parameters", "bc_values")`.

    Returns:
        `mapped_data` with any oversized batch-carrying arrays truncated to
        `fields`' sample count. Returned unchanged if `fields` is absent.

    Raises:
        ValueError: If a batch-carrying key has fewer samples than `fields`.
    """
    if "fields" not in mapped_data:
        return mapped_data

    num_samples = mapped_data["fields"].shape[0]
    result = dict(mapped_data)
    for key in batch_keys:
        if key == "fields" or key not in result:
            continue

        arr = result[key]
        n = arr.shape[0]
        if n > num_samples:
            warnings.warn(
                f"'{key}' has {n} samples but 'fields' has {num_samples}; "
                f"truncating '{key}' to the first {num_samples} samples to "
                "match. This usually means the dataset was only partially "
                "downloaded (e.g. every parameter file but not every field "
                "file) -- download the dataset completely, or pass matching "
                "subsets, to avoid this.",
                stacklevel=2,
            )
            result[key] = arr[:num_samples]
        elif n < num_samples:
            raise ValueError(
                f"'{key}' has only {n} samples but 'fields' has {num_samples}. "
                f"'fields' cannot be safely truncated to match without "
                "discarding field data. Ensure the dataset was downloaded "
                f"completely, or pass a '{key}' subset that matches."
            )

    return result


def create_default_coords(
    spatial_shape: Sequence[int], domain_shape: Sequence[Sequence[int | float]]
) -> Real[np.ndarray, "d *spatial"]:
    """Creates default evenly spaced coordinates over a (hyper-)rectangular domain.

    Args:
        spatial_shape: Number of discretization points per spatial dimension.
        domain_shape: Physical domain size per dimension, as a sequence of
            (start, stop) pairs, one per spatial dimension.

    Returns:
        Coordinate array of shape (d, *spatial_shape), where d is the number
        of spatial dimensions.

    Raises:
        ValueError: If `spatial_shape` and `domain_shape` have different lengths.
    """
    if len(spatial_shape) != len(domain_shape):
        raise ValueError(
            "spatial_shape and domain_shape must have the same length, got "
            f"{len(spatial_shape)} and {len(domain_shape)}."
        )

    axes = [
        np.linspace(start, stop, num)
        for (start, stop), num in zip(domain_shape, spatial_shape, strict=True)
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack(grids, axis=0)


def ensure_time_and_channel_axes(
    fields: Inexact[np.ndarray, "b *vary"], has_time_dim: bool, has_channel_dim: bool
) -> Inexact[np.ndarray, "b t c *spatial"]:
    """Inserts singleton time and/or channel axes so `fields` matches `[b, t, c, *spatial]`.

    PDEGym variables are not always shaped with both a time and a channel
    axis. Steady-state datasets have no time axis, and datasets with a single
    physical field have no explicit channel axis. `DataBundle`/`BundleDataset`
    require both axes to be present (as size 1 if there is only one time step
    or channel). This inserts whichever is missing.

    Args:
        fields: Array shaped `[b, (t), (c), *spatial]`, i.e. `b` followed by
            `t` and `c` only if they are already present.
        has_time_dim: Whether `fields` already has a time axis at position 1.
        has_channel_dim: Whether `fields` already has a channel axis right
            after the time axis (or at position 1 if there is no time axis).

    Returns:
        Array shaped `[b, t, c, *spatial]`.
    """
    if not has_time_dim:
        fields = np.expand_dims(fields, axis=1)
    if not has_channel_dim:
        fields = np.expand_dims(fields, axis=2)
    return fields
