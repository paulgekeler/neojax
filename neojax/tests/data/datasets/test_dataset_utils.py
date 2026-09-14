import pathlib
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest

from neojax.data.datasets.utils import (
    broadcast_like,
    create_default_coords,
    enforce_consistent_batch_size,
    ensure_time_and_channel_axes,
    load_pdebench_data,
    load_pdegym_data,
    map_dataset_fields,
)


@pytest.fixture
def dummy_pdegym_file() -> pathlib.Path:
    import netCDF4 as nc

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "wave.nc"
    with nc.Dataset(file_path, "w") as f:
        f.createDimension("sample", 4)
        f.createDimension("time", 3)
        f.createDimension("x", 8)
        f.createDimension("y", 8)
        var = f.createVariable("solution", "f4", ("sample", "time", "x", "y"))
        var[:] = np.random.randn(4, 3, 8, 8)
    return file_path


@pytest.fixture
def dummy_pdebench_file() -> pathlib.Path:
    import h5py

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "bench.hdf5"
    with h5py.File(file_path, "w") as f:
        f.create_dataset("tensor", data=np.random.randn(4, 3, 8, 8, 1))
    return file_path


class TestLoadingStaysHost:
    """`load_*_data` should never eagerly place a whole dataset on a JAX device."""

    def test_load_pdegym_data_returns_numpy(self, dummy_pdegym_file: pathlib.Path):
        raw_data, var_dims = load_pdegym_data(dummy_pdegym_file)
        assert isinstance(raw_data["solution"], np.ndarray)
        assert "solution" in var_dims

    def test_load_pdebench_data_returns_numpy(self, dummy_pdebench_file: pathlib.Path):
        raw_data = load_pdebench_data(dummy_pdebench_file)
        assert isinstance(raw_data["tensor"], np.ndarray)


class TestMapDatasetFieldsStaysHost:
    def test_single_variable_stays_numpy(self):
        raw = {"solution": np.zeros((4, 3, 8, 8))}
        mapped = map_dataset_fields(raw, {"fields": "solution"})
        assert isinstance(mapped["fields"], np.ndarray)

    def test_concatenated_variables_stay_numpy(self):
        raw = {"a": np.zeros((4, 1, 8, 8)), "b": np.zeros((4, 1, 8, 8))}
        mapped = map_dataset_fields(raw, {"fields": ["a", "b"]}, concat_axes=[1])
        assert isinstance(mapped["fields"], np.ndarray)
        assert mapped["fields"].shape == (4, 2, 8, 8)


class TestCoordAndAxisHelpersStayHost:
    def test_create_default_coords_returns_numpy(self):
        coords = create_default_coords((8, 8), [(0.0, 1.0), (0.0, 1.0)])
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (2, 8, 8)

    def test_ensure_time_and_channel_axes_returns_numpy(self):
        fields = np.zeros((4, 8, 8))
        out = ensure_time_and_channel_axes(
            fields, has_time_dim=False, has_channel_dim=False
        )
        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 1, 1, 8, 8)


class TestEnforceConsistentBatchSize:
    """`fields` is authoritative for sample count; other batch keys must match."""

    def test_matching_sizes_are_unchanged(self):
        mapped = {
            "fields": np.zeros((4, 1, 2, 8, 8)),
            "parameters": np.zeros((4, 8, 8)),
        }
        result = enforce_consistent_batch_size(mapped)
        assert result["fields"].shape[0] == 4
        assert result["parameters"].shape[0] == 4
        np.testing.assert_array_equal(result["parameters"], mapped["parameters"])

    def test_oversized_parameters_are_truncated_with_warning(self):
        # e.g. every parameter file downloaded, but only some field files.
        mapped = {
            "fields": np.zeros((4, 1, 2, 8, 8)),
            "parameters": np.arange(10 * 8 * 8).reshape(10, 8, 8),
        }
        with pytest.warns(UserWarning, match="parameters.*10 samples.*fields.*4"):
            result = enforce_consistent_batch_size(mapped)
        assert result["parameters"].shape[0] == 4
        np.testing.assert_array_equal(result["parameters"], mapped["parameters"][:4])

    def test_oversized_bc_values_are_truncated_with_warning(self):
        mapped = {
            "fields": np.zeros((4, 1, 2, 8, 8)),
            "bc_values": np.zeros((10, 1, 1, 8, 8)),
        }
        with pytest.warns(UserWarning, match="bc_values"):
            result = enforce_consistent_batch_size(mapped)
        assert result["bc_values"].shape[0] == 4

    def test_undersized_batch_key_raises(self):
        mapped = {
            "fields": np.zeros((4, 1, 2, 8, 8)),
            "parameters": np.zeros((2, 8, 8)),
        }
        with pytest.raises(ValueError, match="parameters.*2 samples.*fields.*4"):
            enforce_consistent_batch_size(mapped)

    def test_missing_fields_key_is_a_noop(self):
        mapped = {"parameters": np.zeros((10, 8, 8))}
        result = enforce_consistent_batch_size(mapped)
        assert result is mapped

    def test_non_batch_keys_are_never_touched(self):
        # coords/bc_masks/edge_indices carry no leading batch axis by
        # DataBundle's convention and must be left alone regardless of shape.
        mapped = {
            "fields": np.zeros((4, 1, 2, 8, 8)),
            "coords": np.zeros((2, 8, 8)),
            "bc_masks": np.zeros((1, 8, 8)),
            "edge_indices": np.zeros((2, 100)),
        }
        result = enforce_consistent_batch_size(mapped)
        assert result["coords"].shape == mapped["coords"].shape
        assert result["bc_masks"].shape == mapped["bc_masks"].shape
        assert result["edge_indices"].shape == mapped["edge_indices"].shape


class TestBroadcastLike:
    def test_preserves_numpy(self):
        arr = np.zeros((2, 4))
        out = broadcast_like(arr, (3, 2, 4))
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 2, 4)

    def test_preserves_jax(self):
        arr = jnp.zeros((2, 4))
        out = broadcast_like(arr, (3, 2, 4))
        assert isinstance(out, jnp.ndarray)
        assert out.shape == (3, 2, 4)
