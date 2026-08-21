import pathlib
import tempfile

import h5py
import numpy as np
import pytest
import yaml

from neojax.benchmark.config import BenchmarkConfig
from neojax.benchmark.runner import BenchmarkRunner


@pytest.fixture
def dummy_pdebench_file() -> pathlib.Path:
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "dummy.hdf5"

    with h5py.File(file_path, "w") as f:
        # shape: [batch, time, x, y, channel]
        # e.g., 4 samples, 5 timesteps, 16x16 grid, 1 channel
        f.create_dataset("tensor", data=np.random.randn(4, 5, 16, 16, 1))
        # coords: [dim, x, y]
        f.create_dataset("coords", data=np.random.randn(2, 16, 16))

    return file_path


@pytest.fixture
def dummy_benchmark_config(
    dummy_pdebench_file: pathlib.Path, tmp_path: pathlib.Path
) -> pathlib.Path:
    config_dict = {
        "benchmark_name": "integration_test",
        "global_settings": {"output_dir": str(tmp_path), "batch_size": 2, "seed": 42},
        "tasks": [
            {
                "name": "dummy_task",
                "datasets": [
                    {
                        "name": "dummy_dataset",
                        "type": "BundleDataset",
                        "source": "pdebench",
                        "path": str(dummy_pdebench_file),
                        "field_mapping": {"fields": "tensor", "coords": "coords"},
                        "train_split": [0.0, 0.5],
                        "test_split": [0.5, 1.0],
                    }
                ],
                "evaluator": {
                    "type": "TimeDependentEvaluator",
                    "metrics": [{"type": "RelativeLpMetric", "kwargs": {"p": 2}}],
                    "kwargs": {"history_steps": 1},
                },
                "models": [
                    {
                        "name": "fno_model",
                        "framework": "neojax",
                        "architecture": "FNO",
                        "hyperparameters": {
                            "in_channels": 3,  # 1 channel + 2 coords
                            "out_channels": 1,
                            "hidden_channels": 4,
                            "modes": [2, 2],
                            "n_layers": 1,
                        },
                        "pipeline": {
                            "normalizers": {
                                "fields": {
                                    "type": "UnitGaussianNormalizer",
                                    "compute_on_fly": True,
                                }
                            },
                            "in_schema": [
                                {
                                    "type": "FlattenTimeSchema",
                                    "kwargs": {"time_axis": 0, "channel_axis": 1},
                                },
                                {
                                    "type": "ConcatenateCoordsSchema",
                                    "kwargs": {"channel_axis": 0},
                                },
                            ],
                            "out_schema": {
                                "type": "BundleReconstructSchema",
                                "kwargs": {
                                    "unflatten_time_steps": 1,
                                    "channel_axis": 0,
                                },
                            },
                        },
                    },
                    {
                        "name": "deeponet_model",
                        "framework": "neojax",
                        "architecture": "MLPDeepONet",
                        "vmap_in_axes": [None, 0],
                        "hyperparameters": {
                            "m_sensors": 256,  # 1 channel * 1 timestep * 16x16
                            "d_dim": 2,
                            "p_latent": 8,
                            "branch_hidden_dims": [16],
                            "trunk_hidden_dims": [16],
                        },
                        "pipeline": {
                            "normalizers": {
                                "fields": {
                                    "type": "UnitGaussianNormalizer",
                                    "compute_on_fly": True,
                                }
                            },
                            "in_schema": {"type": "FlattenToPointsSchema"},
                            "out_schema": {
                                "type": "ReshapePointsToGridSchema",
                                "kwargs": {"time_steps": 1, "channels": 1},
                            },
                        },
                    },
                ],
            }
        ],
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    return config_path


@pytest.mark.slow
def test_benchmark_runner(dummy_benchmark_config: pathlib.Path) -> None:
    # Load config
    with open(dummy_benchmark_config) as f:
        config_dict = yaml.safe_load(f)

    config = BenchmarkConfig(**config_dict)

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Assertions
    assert "tasks" in results
    assert "dummy_task" in results["tasks"]
    task_res = results["tasks"]["dummy_task"]
    assert "dummy_dataset" in task_res
    ds_res = task_res["dummy_dataset"]

    assert "fno_model" in ds_res
    assert "deeponet_model" in ds_res

    # Verify metrics exist
    assert "RelativeLpMetric_mean" in ds_res["fno_model"]
    assert "RelativeLpMetric_std" in ds_res["fno_model"]
    assert "RelativeLpMetric_mean" in ds_res["deeponet_model"]
    assert "RelativeLpMetric_std" in ds_res["deeponet_model"]

    # Verify YAML file was saved
    output_dir = pathlib.Path(config_dict["global_settings"]["output_dir"])
    output_file = output_dir / f"{config_dict['benchmark_name']}_results.yaml"
    assert output_file.exists()

    with open(output_file) as f:
        saved_results = yaml.safe_load(f)

    assert saved_results["benchmark_name"] == results["benchmark_name"]
    assert "dummy_task" in saved_results["tasks"]
