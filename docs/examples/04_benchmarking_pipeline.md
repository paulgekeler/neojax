### Example 4: The Benchmarking Pipeline

[:material-download: Download Notebook](04_benchmarking_pipeline.ipynb){ .md-button }

In this tutorial we explore the **neojax** benchmarking pipeline. Comparing neural operator architectures requires evaluating them under identical training/testing pipelines, metrics, and dataset splits.

`neojax` provides `BenchmarkRunner`, a config-driven benchmarking orchestrator. In this example, we compare a **neojax** model (`GeoFNO`) against an **external Flax model** (`RIGNO` [^1]) on unstructured PDE datasets using custom builders and config schemas.

First we install the needed packages.

```python
!pip3 install "neojax-operators[benchmark,data]"
```

```python
# Clone RIGNO repo into 'rigno_repo'
!git clone https://github.com/camlab-ethz/rigno.git ./rigno_repo
```

```python
!pip install -r "./rigno_repo/requirements.txt"
```

```python
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4 as nc
import numpy as np
from jax import Array

from neojax.benchmark.config import BenchmarkConfig
from neojax.benchmark.runner import BenchmarkRunner
from neojax.models.geo_fno import GeoFNO

# Patch jnp.clip for JAX version compatibility before importing rigno
orig_clip = jnp.clip

def patched_clip(a, min=None, max=None, **kwargs):
    if "a_min" in kwargs:
        min = kwargs.pop("a_min")
    if "a_max" in kwargs:
        max = kwargs.pop("a_max")
    return orig_clip(a, min, max, **kwargs)
jnp.clip = patched_clip

# Include the cloned rigno_repo in sys.path
sys.path.append(str(Path.cwd() / "rigno_repo"))

from rigno.models.operator import Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphBuilder
```

#### 1. Define Custom Model Builders
To support custom or external models in the benchmark, you write builder functions that instantiate the architectures and initialize their parameters.

```python
def build_geo_fno_custom(config: Any, key: jax.Array, dataset: Any) -> "GeoFNO":
    """Builder to instantiate GeoFNO directly."""
    return GeoFNO(
        key=key,
        in_channels=config.hyperparameters["in_channels"],
        out_channels=config.hyperparameters["out_channels"],
        hidden_channels=config.hyperparameters["hidden_channels"],
        n_layers=config.hyperparameters["n_layers"],
        modes=config.hyperparameters["modes"],
        grid_resolution=config.hyperparameters["grid_resolution"],
        use_coord_projection=config.hyperparameters.get("use_coord_projection", True),
    )

def build_rigno_custom(config: Any, key: jax.Array, dataset: Any) -> Callable[[Array], Array]:
    """Builder to instantiate Flax RIGNO as a pure JAX callable."""
    coords = dataset.coords
    x_in = coords.T  # (N, dim)

    # Delaunay triangulation requires 2D coordinates
    if x_in.shape[1] == 1:
        coords_2d = np.stack([x_in[:, 0], np.sin(x_in[:, 0] * 2 * np.pi) * 0.1], axis=1)
    else:
        coords_2d = np.array(x_in)

    domain = np.array([
        [coords_2d[:, 0].min(), coords_2d[:, 1].min()],
        [coords_2d[:, 0].max(), coords_2d[:, 1].max()]
    ])

    graph_builder = RegionInteractionGraphBuilder(
        periodic=False,
        rmesh_levels=1,
        subsample_factor=2.0,
        overlap_factor_p2r=1.2,
        overlap_factor_r2p=1.2,
        node_coordinate_freqs=1,
    )
    metadata = graph_builder.build_metadata(
        x_inp=coords_2d, x_out=coords_2d, domain=domain
    )
    graphs = graph_builder.build_graphs(metadata)

    flax_model = RIGNO(
        num_outputs=config.hyperparameters["out_channels"],
        processor_steps=config.hyperparameters.get("processor_steps", 2),
        node_latent_size=config.hyperparameters.get("node_latent_size", 16),
        edge_latent_size=config.hyperparameters.get("edge_latent_size", 16),
    )

    u_init = jnp.zeros((1, 1, coords_2d.shape[0], config.hyperparameters["in_channels"]))
    x_init = jnp.array(coords_2d)[None, None, ...]
    inputs_init = Inputs(u=u_init, c=None, x_inp=x_init, x_out=x_init, t=0.0, tau=1.0)

    variables = flax_model.init(key, inputs_init, graphs=graphs)
    params = variables["params"]

    return lambda inputs: flax_model.apply({"params": params}, inputs, graphs=graphs)
```

#### 2. Create Mock PDEGym Datasets
We create mock dataset NetCDF files to represent PDEGym unstructured meshes.

```python
def generate_mock_pdegym_nc(file_path: Path) -> None:
    with nc.Dataset(file_path, "w") as f:
        f.createDimension("sample", 4)
        f.createDimension("time", 1)
        f.createDimension("nodes", 256)
        f.createDimension("spatial_dim", 2)
        f.createDimension("channel", 1)

        np.random.seed(42)
        coords = np.random.rand(2, 256)
        c_var = f.createVariable("coords", "f4", ("spatial_dim", "nodes"))
        c_var[:] = coords

        s_var = f.createVariable("solution", "f4", ("sample", "time", "channel", "nodes"))
        s_var[:] = np.random.randn(4, 1, 1, 256)
```

#### 3. Define the Benchmarking Configurations
We formulate a `BenchmarkConfig` dictionary detailing evaluator classes, normalizers, input/output schemas, and model structures.

```python
# Setup temporary files
tmpdir = tempfile.TemporaryDirectory()
tmp_path = Path(tmpdir.name)
ace_path = tmp_path / "ace_solution.nc"
wave_layer_path = tmp_path / "wave_layer_solution.nc"

generate_mock_pdegym_nc(ace_path)
generate_mock_pdegym_nc(wave_layer_path)

config_dict = {
    "benchmark_name": "pdegym_rigno_comparison",
    "global_settings": {
        "output_dir": str(tmp_path),
        "batch_size": 2,
        "seed": 42
    },
    "tasks": [
        {
            "name": "ace_task",
            "datasets": [
                {
                    "name": "ace_dataset",
                    "type": "BundleDataset",
                    "source": "pdegym",
                    "path": str(ace_path),
                    "field_mapping": {
                        "fields": "solution",
                        "coords": "coords"
                    },
                }
            ],
            "evaluator": {
                "type": "SteadyStateEvaluator",
                "metrics": [
                    {"type": "RelativeLpMetric", "kwargs": {"p": 2.0}},
                    "MSEMetric"
                ]
            },
            "models": [
                {
                    "name": "geo_fno",
                    "framework": "custom",
                    "architecture": "GeoFNO",
                    "hyperparameters": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "hidden_channels": 8,
                        "n_layers": 3,
                        "modes": [4, 4],
                        "grid_resolution": [8, 8],
                    },
                    "pipeline": {
                        "in_schema": {
                            "type": "MeshInputSchema",
                            "kwargs": {"time_axis": 0, "channel_axis": 1}
                        },
                        "out_schema": {
                            "type": "BundleReconstructSchema",
                            "kwargs": {
                                "unflatten_time_steps": 1,
                                "channel_axis": 0
                            }
                        }
                    }
                },
                {
                    "name": "rigno",
                    "framework": "custom",
                    "architecture": "RIGNO",
                    "hyperparameters": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "processor_steps": 2,
                        "node_latent_size": 8,
                        "edge_latent_size": 8,
                    },
                    "pipeline": {
                        "in_schema": {
                            "type": "GraphTupleInputSchema"
                        },
                        "out_schema": {
                            "type": "GraphTupleOutputSchema"
                        }
                    }
                }
            ]
        }
    ]
}

config = BenchmarkConfig(**config_dict)
print("Config constructed.")
```
**Output**
```bash
Config constructed.
```

#### 4. Initialize and Run the Orchestrator
We supply the custom model builders and execute the benchmark.

```python
custom_model_builders = {
    "GeoFNO": build_geo_fno_custom,
    "RIGNO": build_rigno_custom,
}

runner = BenchmarkRunner(config, custom_model_builders=custom_model_builders)
results = runner.run()

# Clean up temp dir
tmpdir.cleanup()
```

**Output**
```bash
Starting task: ace_task
  -> Loading dataset: ace_dataset
    -> Evaluating model: geo_fno
    -> Evaluating model: rigno
Benchmarking complete. Results saved to /tmp/tmpsf2yk2ym/pdegym_rigno_comparison_results.yaml
```

#### 5. Analyze the Outputs
Results are structured task-by-task, dataset-by-dataset, and metric-by-metric.

```python
print("\n================ BENCHMARK RESULTS ==================")
for task_name, task_res in results["tasks"].items():
    print(f"\nTask: {task_name}")
    for dataset_name, ds_res in task_res.items():
        print(f"  Dataset: {dataset_name}")
        for model_name, model_stats in ds_res.items():
            print(f"    Model: {model_name}")
            for metric_name, val in model_stats.items():
                print(f"      {metric_name}: {val:.6f}")
print("=====================================================")
```

**Output**
```bash
================ BENCHMARK RESULTS ==================

Task: ace_task
  Dataset: ace_dataset
    Model: geo_fno
      RelativeLpMetric_mean: 1.086907
      RelativeLpMetric_std: 0.018957
      MSEMetric_mean: 1.180793
      MSEMetric_std: 0.049722
    Model: rigno
      RelativeLpMetric_mean: 1.025317
      RelativeLpMetric_std: 0.004395
      MSEMetric_mean: 1.051488
      MSEMetric_std: 0.053286
=====================================================
```

[^1]: [RIGNO](https://arxiv.org/abs/2501.19205) publication