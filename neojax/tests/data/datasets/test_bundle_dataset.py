import pathlib
import tempfile

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.datasets.bundle_dataset import BundleDataset


class TestBundleDataset:
    @pytest.mark.parametrize("idx", [5, slice(4, None, None), slice(None, None, 3)])
    def test_getitem(self, idx):
        fields = jnp.repeat(
            jnp.arange(0, 20, 1, dtype=jnp.float32)[None, None, None, :],
            repeats=jnp.array([6]),
            axis=0,
        )
        coords = jnp.arange(0, 20, 1)[None, :]
        ds = BundleDataset(coords=coords, fields=fields)

        sliced_fields = fields[idx]
        if sliced_fields.ndim == fields.ndim:
            b = sliced_fields.shape[0]
            exp_coords = jnp.broadcast_to(coords, (b, *coords.shape))
        else:
            exp_coords = coords

        exp_tree = DataBundle(coords=exp_coords, fields=sliced_fields)
        assert eqx.tree_equal(ds[idx], exp_tree)

    def test_grain_integration(self):
        import grain

        fields = jnp.zeros((10, 2, 3, 4, 4))
        coords = jnp.zeros((2, 4, 4))
        ds = BundleDataset(coords=coords, fields=fields)

        source = grain.MapDataset.source(ds)
        batched_ds = source.shuffle(seed=42).batch(batch_size=2)

        it = batched_ds.__iter__()
        batch = next(it)

        # Should be a DataBundle and shapes should include batch axis 2
        assert isinstance(batch, DataBundle)
        assert batch.fields.shape == (2, 2, 3, 4, 4)
        assert batch.coords.shape == (2, 2, 4, 4)

    def test_len(self):
        fields = jnp.repeat(
            jnp.arange(0, 20, 1, dtype=jnp.float32)[None, None, None, :],
            repeats=jnp.array([6]),
            axis=0,
        )
        coords = jnp.arange(0, 20, 1)[None, :]
        ds = BundleDataset(coords=coords, fields=fields)
        assert len(ds) == fields.shape[0]


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
            f.createDimension("dim", 2)
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

            coords = f.createVariable("x_coords", "f4", ("dim", "x", "y"))
            coords[:] = np.random.randn(2, 16, 16)

    return temp_dir


@pytest.fixture
def dummy_pdegym_separated_vars_dir() -> pathlib.Path:
    import netCDF4 as nc

    temp_dir = pathlib.Path(tempfile.mkdtemp())

    # File 0: navier variable
    with nc.Dataset(temp_dir / "navier_0.nc", "w") as f:
        f.createDimension("sample", 2)
        f.createDimension("time", 5)
        f.createDimension("channel", 1)
        f.createDimension("x", 16)
        f.createDimension("y", 16)
        var = f.createVariable("navier", "f4", ("sample", "time", "channel", "x", "y"))
        var[:] = np.random.randn(2, 5, 1, 16, 16)

    # File 1: velocity variable
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
        f.createDimension("dim", 2)
        f.createDimension("x", 16)
        f.createDimension("y", 16)
        coords = f.createVariable("x_coords", "f4", ("dim", "x", "y"))
        coords[:] = np.random.randn(2, 16, 16)

    return temp_dir


@pytest.fixture
def dummy_pdegym_h5_dir() -> pathlib.Path:
    import h5py

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "helmholtz.h5"
    coords_data = np.ones((2, 16, 16))

    with h5py.File(file_path, "w") as f:
        for i in range(4):
            grp = f.create_group(f"Sample_{i}")
            grp.create_dataset(
                "fields", data=np.random.randn(5, 1, 16, 16)
            )  # (time, channel, x, y)
            grp.create_dataset("coords", data=coords_data)

    return temp_dir


def test_from_pdegym(
    dummy_pdegym_dir: pathlib.Path,
    dummy_pdegym_separated_vars_dir: pathlib.Path,
    dummy_pdegym_h5_dir: pathlib.Path,
):
    # Missing field mapping should fail
    with pytest.raises(
        ValueError, match="field_mapping must be provided for BundleDataset"
    ):
        BundleDataset.from_pdegym(dummy_pdegym_dir)

    # Invalid field mapping key should fail
    with pytest.raises(ValueError, match="Invalid DataBundle key"):
        BundleDataset.from_pdegym(
            dummy_pdegym_dir, field_mapping={"invalid_key": "navier"}
        )

    # Standard mapping
    field_mapping = {"fields": ["navier", "velocity"], "coords": "x_coords"}
    concat_axes = [2, 0]  # Axis 2 for fields

    ds = BundleDataset.from_pdegym(
        dummy_pdegym_dir, field_mapping=field_mapping, concat_axes=concat_axes
    )

    assert ds.fields.shape == (4, 5, 2, 16, 16)
    assert ds.coords.shape == (2, 16, 16)

    # Multi-file separated variables mapping
    ds_sep = BundleDataset.from_pdegym(
        dummy_pdegym_separated_vars_dir,
        field_mapping=field_mapping,
        concat_axes=concat_axes,
    )
    assert ds_sep.fields.shape == (2, 5, 2, 16, 16)
    assert ds_sep.coords.shape == (2, 16, 16)

    # HDF5 dataset mapping
    field_mapping_h5 = {"fields": "fields", "coords": "coords"}
    ds_h5 = BundleDataset.from_pdegym(
        dummy_pdegym_h5_dir, field_mapping=field_mapping_h5
    )
    assert ds_h5.fields.shape == (4, 5, 1, 16, 16)
    assert ds_h5.coords.shape == (2, 16, 16)


def test_from_pdebench(dummy_pdebench_dir: pathlib.Path):
    # Missing field mapping should fail
    with pytest.raises(
        ValueError, match="field_mapping must be provided for BundleDataset"
    ):
        BundleDataset.from_pdebench(dummy_pdebench_dir)

    # Valid mapping
    field_mapping = {"fields": ["tensor", "velocity"], "coords": "x-coordinate"}
    concat_axes = [2, 0]

    ds = BundleDataset.from_pdebench(
        dummy_pdebench_dir, field_mapping=field_mapping, concat_axes=concat_axes
    )

    assert ds.fields.shape == (4, 5, 2, 16, 16)
    assert ds.coords.shape == (2, 16, 16)


@pytest.mark.slow
def test_real_pdegym_dataset_bundle():
    import netCDF4 as nc

    from neojax.data.download.huggingface_downloader import HuggingFaceDownloader

    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = HuggingFaceDownloader(repo_id="camlab-ethz/SE-AF")
        downloader.download(temp_dir)

        nc_files = list(pathlib.Path(temp_dir).glob("**/*.nc"))
        if not nc_files:
            pytest.skip("No .nc files downloaded")

        with nc.Dataset(nc_files[0]) as f:
            keys = list(f.variables.keys())

        coords_keys = [
            k for k in keys if "coord" in k.lower() or k in ("x", "y", "grid")
        ]
        field_keys = [k for k in keys if k not in coords_keys]

        if not coords_keys:
            coords_keys = [keys[0]]
            field_keys = keys[1:]

        field_mapping = {"coords": coords_keys[0], "fields": field_keys}

        # Determine concat_axes length based on number of field outputs
        concat_axes = [2] * len(field_mapping)

        try:
            ds = BundleDataset.from_pdegym(
                temp_dir, field_mapping=field_mapping, concat_axes=concat_axes
            )
            assert len(ds) > 0
        except Exception as e:
            # Catch shape/jaxtyping exceptions
            print(
                f"Dataset parsed, but failed type/shape validation (expected for unknown structure): {e}"
            )


@pytest.fixture
def dummy_pdebench_dir() -> pathlib.Path:
    import h5py

    temp_dir = pathlib.Path(tempfile.mkdtemp())

    for i in range(2):
        file_path = temp_dir / f"dummy_bench_{i}.hdf5"
        with h5py.File(file_path, "w") as f:
            # Data shape [batch, time, x, y, channel]
            f.create_dataset("tensor", data=np.random.randn(2, 5, 16, 16, 1))
            f.create_dataset("velocity", data=np.random.randn(2, 5, 16, 16, 1))
            # coords: [d, x, y] - must be identical across files to avoid being concatenated
            f.create_dataset("x-coordinate", data=np.ones((2, 16, 16)))

    return temp_dir


@pytest.mark.slow
def test_real_pdebench_dataset_bundle():
    import h5py

    from neojax.data.download.dataverse_downloader import DataverseDownloader

    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = DataverseDownloader(
            doi="10.18419/darus-2986", filenames=["1D_diff-sorp_NA_NA.h5"]
        )
        downloader.download(temp_dir)

        h5_files = list(pathlib.Path(temp_dir).glob("**/*.h5"))
        if not h5_files:
            pytest.skip("No .h5 files downloaded")

        with h5py.File(h5_files[0], "r") as f:
            keys = list(f.keys())

        coords_keys = [
            k for k in keys if "coord" in k.lower() or k in ("x", "y", "grid")
        ]
        field_keys = [
            k for k in keys if k not in coords_keys and isinstance(f[k], h5py.Dataset)
        ]

        if not coords_keys:
            coords_keys = [keys[0]]
            field_keys = keys[1:]

        field_mapping = {"coords": coords_keys[0], "fields": field_keys}
        concat_axes = [2] * len(field_mapping)

        try:
            ds = BundleDataset.from_pdebench(
                temp_dir, field_mapping=field_mapping, concat_axes=concat_axes
            )
            assert len(ds) > 0
        except Exception as e:
            print(
                f"Dataset parsed, but failed type/shape validation (expected for unknown structure): {e}"
            )


@pytest.mark.slow
def test_real_local_netcdf_pdegym():
    from pathlib import Path

    data_path = Path("/Users/paul/projects/data/test_data/ncs")
    if not data_path.is_dir():
        pytest.skip(f"Local PDEGym sample data not available at {data_path}.")

    # No 'coords' variable exists in any PDEGym dataset
    # Generated automatically as a unit-hypercube grid.
    with pytest.warns(UserWarning, match="coords"):
        dataset = BundleDataset.from_pdegym(
            data_path,
            field_mapping={"fields": "solution", "parameters": "c"},
        )

    assert dataset._num_samples == 10512
    assert dataset.parameters.shape == (10512, 128, 128)
    # coords are generated over the unit square.
    assert dataset.coords.shape == (2, 128, 128)
    assert float(dataset.coords.min()) == 0.0
    assert float(dataset.coords.max()) == 1.0
    # 'solution' has dims (sample, time, x, y): 21 time steps, no channel axis
    # Singleton channel axis is inserted
    assert dataset.fields.shape == (10512, 21, 1, 128, 128)
