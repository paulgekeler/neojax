import pathlib
import tempfile

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neojax.data.datasets.raw_dataset import RawDataset


@pytest.fixture
def dummy_pdebench_dir() -> pathlib.Path:
    temp_dir = pathlib.Path(tempfile.mkdtemp())

    for i in range(2):
        file_path = temp_dir / f"dummy_bench_{i}.hdf5"
        with h5py.File(file_path, "w") as f:
            # data shape [batch, time, x, y, channel]
            f.create_dataset("tensor", data=np.random.randn(2, 5, 16, 16, 1))
            f.create_dataset("velocity", data=np.random.randn(2, 5, 16, 16, 1))
            # coords: [x, y] - must be identical across files to avoid being concatenated
            f.create_dataset("x-coordinate", data=np.ones((2, 16, 16)))

    return temp_dir


@pytest.fixture
def dummy_pdegym_dir() -> pathlib.Path:
    import netCDF4 as nc

    temp_dir = pathlib.Path(tempfile.mkdtemp())

    for i in range(2):
        file_path = temp_dir / f"navier_stokes_{i}.nc"
        with nc.Dataset(file_path, "w") as f:
            f.createDimension("sample", 2)
            f.createDimension("time", 5)
            f.createDimension("channel", 1)
            f.createDimension("x", 16)
            f.createDimension("y", 16)

            var = f.createVariable(
                "navier", "f4", ("sample", "time", "channel", "x", "y")
            )
            var[:] = np.random.randn(2, 5, 1, 16, 16)

            var2 = f.createVariable(
                "velocity", "f4", ("sample", "time", "channel", "x", "y")
            )
            var2[:] = np.random.randn(2, 5, 1, 16, 16)

            coords = f.createVariable("x_coords", "f4", ("x", "y"))
            coords[:] = np.random.randn(16, 16)

    return temp_dir


@pytest.fixture
def dummy_pdegym_separated_vars_dir() -> pathlib.Path:
    import netCDF4 as nc

    temp_dir = pathlib.Path(tempfile.mkdtemp())

    # File 0: navier variable (2 samples)
    with nc.Dataset(temp_dir / "navier_0.nc", "w") as f:
        f.createDimension("sample", 2)
        f.createDimension("time", 5)
        f.createDimension("channel", 1)
        f.createDimension("x", 16)
        f.createDimension("y", 16)
        var = f.createVariable("navier", "f4", ("sample", "time", "channel", "x", "y"))
        var[:] = np.random.randn(2, 5, 1, 16, 16)

    # File 1: velocity variable (2 samples)
    with nc.Dataset(temp_dir / "velocity_0.nc", "w") as f:
        f.createDimension("sample", 2)
        f.createDimension("time", 5)
        f.createDimension("channel", 1)
        f.createDimension("x", 16)
        f.createDimension("y", 16)
        var = f.createVariable(
            "velocity", "f4", ("sample", "time", "channel", "x", "y")
        )
        var[:] = np.random.randn(2, 5, 1, 16, 16)

    # File 2: x_coords variable
    with nc.Dataset(temp_dir / "coords_0.nc", "w") as f:
        f.createDimension("x", 16)
        f.createDimension("y", 16)
        coords = f.createVariable("x_coords", "f4", ("x", "y"))
        coords[:] = np.random.randn(16, 16)

    return temp_dir


@pytest.fixture
def dummy_pdegym_h5_dir() -> pathlib.Path:
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "helmholtz.h5"
    coords_data = np.ones((16, 16))

    with h5py.File(file_path, "w") as f:
        for i in range(4):
            grp = f.create_group(f"Sample_{i}")
            grp.create_dataset("a", data=np.random.randn(16, 16))
            grp.create_dataset("u", data=np.random.randn(16, 16))
            grp.create_dataset("bc", data=np.float64(i))
            grp.create_dataset("coords", data=coords_data)

    return temp_dir


class TestRawDataset:
    def test_numpy_backed_dataset_stays_numpy_on_getitem(self):
        """Constructing/indexing with numpy shouldn't eagerly place data on a device."""
        fields = np.zeros((5, 1, 2, 4), dtype=np.float32)
        coords = np.zeros((2, 4), dtype=np.float32)
        ds = RawDataset(fields=fields, coords=coords)

        batch = ds[[0, 1]]
        assert isinstance(batch["fields"], np.ndarray)
        assert isinstance(batch["coords"], np.ndarray)

    def test_get_batch_default_leaves_batch_on_host(self):
        fields = np.zeros((5, 1, 2, 4), dtype=np.float32)
        coords = np.zeros((2, 4), dtype=np.float32)
        ds = RawDataset(fields=fields, coords=coords)

        batch = ds.get_batch([0, 1])
        assert isinstance(batch["fields"], np.ndarray)
        assert isinstance(batch["coords"], np.ndarray)

    def test_get_batch_with_device_commits_to_jax(self):
        fields = np.zeros((5, 1, 2, 4), dtype=np.float32)
        coords = np.zeros((2, 4), dtype=np.float32)
        ds = RawDataset(fields=fields, coords=coords)

        batch = ds.get_batch([0, 1], device=jax.devices()[0])
        assert isinstance(batch["fields"], jax.Array)
        assert isinstance(batch["coords"], jax.Array)
        np.testing.assert_array_equal(np.asarray(batch["fields"]), fields[[0, 1]])

    @pytest.mark.parametrize("idx", [5, slice(3, None), slice(None, None, 2)])
    def test_getitem(self, idx: int | slice):
        fields = jnp.repeat(
            jnp.arange(0, 20, 1)[None, None, :], repeats=jnp.array([6]), axis=0
        )
        coords = jnp.arange(0, 20, 1)
        ds = RawDataset(fields=fields, coords=coords)

        assert isinstance(ds[idx], dict)
        if isinstance(idx, int):
            exp_tree = {"fields": jnp.arange(0, 20, 1)[None, :], "coords": coords}
            assert eqx.tree_equal(ds[idx], exp_tree)
        else:
            sliced_fields = fields[idx]
            b = sliced_fields.shape[0]
            exp_tree = {
                "fields": sliced_fields,
                "coords": jnp.broadcast_to(coords, (b, *coords.shape)),
            }
            assert eqx.tree_equal(ds[idx], exp_tree)

    def test_len(self):
        fields = jnp.repeat(
            jnp.arange(0, 20, 1)[None, None, :], repeats=jnp.array([6]), axis=0
        )
        coords = jnp.arange(0, 20, 1)
        ds = RawDataset(fields=fields, coords=coords)
        ds2 = RawDataset(foo=fields)
        assert len(ds) == fields.shape[0]
        assert len(ds2) == fields.shape[0]

    def test_grain_integration(self):
        import grain

        fields = jnp.zeros((10, 2, 3, 4, 4))
        coords = jnp.zeros((2, 4, 4))
        ds = RawDataset(fields=fields, coords=coords)

        source = grain.MapDataset.source(ds)
        batched_ds = source.shuffle(seed=42).batch(batch_size=2)

        it = batched_ds.__iter__()
        batch = next(it)

        # Should be standard dict here
        assert isinstance(batch, dict)
        assert batch["fields"].shape == (2, 2, 3, 4, 4)
        assert batch["coords"].shape == (2, 2, 4, 4)

    def test_from_pdebench(self, dummy_pdebench_dir: pathlib.Path):
        field_mapping = {"fields": ["tensor", "velocity"], "coords": "x-coordinate"}
        concat_axes = [2, 0]

        ds = RawDataset.from_pdebench(
            dummy_pdebench_dir, field_mapping=field_mapping, concat_axes=concat_axes
        )

        assert "fields" in ds.data_dict
        # Channel dimension moved to index 2, then concatenated
        assert ds.data_dict["fields"].shape == (4, 5, 2, 16, 16)
        # Loaded datasets stay host-resident numpy, not eagerly placed on a device.
        assert isinstance(ds.data_dict["fields"], np.ndarray)

        assert "coords" in ds.data_dict
        assert ds.data_dict["coords"].shape == (2, 16, 16)

        ds_raw = RawDataset.from_pdebench(dummy_pdebench_dir)
        assert "tensor" in ds_raw.data_dict
        assert ds_raw.data_dict["tensor"].shape == (4, 5, 1, 16, 16)

    def test_from_pdegym(
        self,
        dummy_pdegym_dir: pathlib.Path,
        dummy_pdegym_separated_vars_dir: pathlib.Path,
        dummy_pdegym_h5_dir: pathlib.Path,
    ):
        # Standard NetCDF directory with all variables in each file
        field_mapping = {"fields": ["navier", "velocity"], "coords": "x_coords"}
        concat_axes = [2, 0]

        ds = RawDataset.from_pdegym(
            dummy_pdegym_dir, field_mapping=field_mapping, concat_axes=concat_axes
        )

        assert "fields" in ds.data_dict
        assert ds.data_dict["fields"].shape == (4, 5, 2, 16, 16)
        assert "coords" in ds.data_dict
        assert ds.data_dict["coords"].shape == (16, 16)

        ds_raw = RawDataset.from_pdegym(dummy_pdegym_dir)
        assert "navier" in ds_raw.data_dict
        assert "velocity" in ds_raw.data_dict
        assert "x_coords" in ds_raw.data_dict
        assert ds_raw.data_dict["navier"].shape == (4, 5, 1, 16, 16)

        # NetCDF directory with variables separated into different files
        ds_sep = RawDataset.from_pdegym(
            dummy_pdegym_separated_vars_dir,
            field_mapping=field_mapping,
            concat_axes=concat_axes,
        )
        assert "fields" in ds_sep.data_dict
        assert ds_sep.data_dict["fields"].shape == (2, 5, 2, 16, 16)
        assert "coords" in ds_sep.data_dict
        assert ds_sep.data_dict["coords"].shape == (16, 16)

        # HDF5 directory with Sample_i groups
        field_mapping_h5 = {"fields": ["a", "u"], "coords": "coords", "bc": "bc"}
        concat_axes_h5 = [2, 0, 0]
        ds_h5 = RawDataset.from_pdegym(
            dummy_pdegym_h5_dir,
            field_mapping=field_mapping_h5,
            concat_axes=concat_axes_h5,
        )
        assert "fields" in ds_h5.data_dict
        assert ds_h5.data_dict["fields"].shape == (4, 16, 32)
        assert "coords" in ds_h5.data_dict
        assert ds_h5.data_dict["coords"].shape == (16, 16)
        assert "bc" in ds_h5.data_dict
        assert ds_h5.data_dict["bc"].shape == (4,)


@pytest.mark.slow
def test_real_pdegym_dataset_raw():
    from neojax.data.download.huggingface_downloader import HuggingFaceDownloader

    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = HuggingFaceDownloader(repo_id="camlab-ethz/SE-AF")
        downloader.download(temp_dir)

        ds = RawDataset.from_pdegym(temp_dir)
        assert len(ds) > 0
        assert len(ds.data_dict) > 0


@pytest.mark.slow
def test_real_pdebench_dataset_raw():
    from neojax.data.download.dataverse_downloader import DataverseDownloader

    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = DataverseDownloader(
            doi="10.18419/darus-2986", filenames=["1D_diff-sorp_NA_NA.h5"]
        )
        downloader.download(temp_dir)

        ds = RawDataset.from_pdebench(temp_dir)
        assert len(ds) > 0
        assert len(ds.data_dict) > 0
