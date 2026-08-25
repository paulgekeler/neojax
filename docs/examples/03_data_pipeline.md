### Example 3: Neural Operator Data Pipeline

[:material-download: Download Notebook](03_data_pipeline.ipynb){ .md-button }

In this tutorial we explore the **neojax** data pipeline: **Datasets**, **Schemas**, and the **BundleProcessor**. Together, these decouple a dataset's raw storage format from the input format a specific model expects, so a dataset and a model can be swapped independently without rewriting the code that connects them. Normalization (also handled by `BundleProcessor`) is one part of that transformation, not the reason the pipeline exists.

The pipeline is agnostic to model architecture and can be used to pre-process data for arbitrary models, including non-neojax ones.

First we install and import the needed packages.

```python
!pip3 install "neojax-operators[data]"
```

```python
import pathlib
import tempfile
import h5py
import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.normalizers import UnitGaussianNormalizer
from neojax.data.processor import BundleProcessor
from neojax.data.schemas import (
    ComposedSchema,
    FlattenTimeSchema,
    ConcatenateCoordsSchema,
    BundleReconstructSchema,
)
```

#### 1. Setup a Dummy PDEBench Dataset
First, we create a dummy HDF5 file formatted like PDEBench datasets. This represents standard simulations with a time dimension and a 2D spatial grid.

```python
# Create temporary dummy PDEBench file
temp_dir = pathlib.Path(tempfile.mkdtemp())
file_path = temp_dir / "pdebench_dummy.hdf5"

with h5py.File(file_path, "w") as f:
    # Shape: [batch=4, time=5, x=16, y=16, channel=1]
    f.create_dataset("tensor", data=np.random.randn(4, 5, 16, 16, 1))
    # Coordinates: [dim=2, x=16, y=16]
    f.create_dataset("coords", data=np.random.randn(2, 16, 16))

print(f"Dummy dataset created at: {file_path}")
```

**Output:**
```text
Dummy dataset created at: [temp_path]/pdebench_dummy.hdf5
```

#### 2. Load the Dataset into a `BundleDataset`
Now we use `BundleDataset.from_pdebench` to load our variables. `BundleDataset` returns typed `DataBundle` objects, which serve as the unified payload structure in **neojax**.

```python
field_mapping = {"fields": "tensor", "coords": "coords"}
dataset = BundleDataset.from_pdebench(file_path, field_mapping=field_mapping)

print(f"Dataset size: {len(dataset)} samples")
sample_bundle = dataset[0]
print(f"Sample fields shape: {sample_bundle.fields.shape}") # [time, channel, x, y]
print(f"Sample coords shape: {sample_bundle.coords.shape}") # [dim, x, y]
```

**Output:**
```text
Dataset size: 4 samples
Sample fields shape: (5, 1, 16, 16)
Sample coords shape: (2, 16, 16)
```

#### 3. Processor & Normalization
`BundleProcessor` combines normalization and schema-based reshaping (added in the next step) into a single `transform`/`inverse_transform` call. We start by fitting a normalizer on the bundle fields.

```python
# Fetch all samples to compute statistics
full_bundle = dataset[:]

# Setup a BundleProcessor with a Gaussian normalizer
processor = BundleProcessor(normalizers={"fields": UnitGaussianNormalizer()})
processor = processor.compute_stats(full_bundle)

print("Processor fitted successfully.")
```

**Output:**
```text
Processor fitted successfully.
```

#### 4. Defining Transformation Schemas
Schemas define how `DataBundle` payloads are transformed into raw arrays expected by models (and vice versa).

- For inputs: FNOs typically expect the time dimension flattened into the channels, followed by the spatial coordinate maps concatenated along the channel axis.
- For outputs: We want to unflatten predictions back to the original physical shape and wrap them in a `DataBundle`.

```python
# Input schema: Flatten time, then concatenate coordinate maps
in_schema = ComposedSchema([
    FlattenTimeSchema(time_axis=1, channel_axis=2),
    ConcatenateCoordsSchema(channel_axis=1),
])

# Output schema: Reconstruct a bundle with 1 predicted timestep
out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=1)
```

#### 5. Executing the Pipeline
Let's slice a batch of samples, project them using our schemas and normalizers, run them through a dummy model, and invert the output.

```python
batch_bundle = dataset[0:4]

# Pre-process proper meshgrid coordinates shape [4, 2, 16, 16]
x, y = jnp.meshgrid(jnp.linspace(0, 1, 16), jnp.linspace(0, 1, 16), indexing="ij")
coords = jnp.stack([x, y], axis=0)
batched_coords = jnp.broadcast_to(coords, (4, *coords.shape))
batch_bundle = eqx.tree_at(lambda b: b.coords, batch_bundle, batched_coords)

# Forward transform
model_inputs = processor.transform(batch_bundle, schema=in_schema)
print(f"Model inputs shape: {model_inputs.shape}") # [batch, channels, x, y]

# Dummy Model Step (predicting 1 future timestep)
model_outputs = jnp.zeros((4, 1, 16, 16))

# Inverse transform (reconstruct and denormalize)
pred_bundle = processor.inverse_transform(model_outputs, schema=out_schema, reference_bundle=batch_bundle)
print(f"Reconstructed fields shape: {pred_bundle.fields.shape}") # [batch, time, channel, x, y]
```

**Output:**
```text
Model inputs shape: (4, 7, 16, 16)
Reconstructed fields shape: (4, 1, 1, 16, 16)
```

#### 6. JIT Compilation of the Pipeline
The data pipeline is fully traceable and compatible with JAX JIT compilation. Here we compile a forward function.

```python
def model_and_pipeline(bundle: DataBundle, proc: BundleProcessor) -> DataBundle:
    inputs = proc.transform(bundle, schema=in_schema)
    # Dummy forward pass
    outputs = jnp.zeros((inputs.shape[0], 1, 16, 16))
    return proc.inverse_transform(outputs, schema=out_schema, reference_bundle=bundle)

# Compile the function
jitted_fn = jax.jit(model_and_pipeline)

# Warm up and execute
out_bundle = jitted_fn(batch_bundle, processor)
print(f"JIT-compiled pipeline output fields shape: {out_bundle.fields.shape}")
```

**Output:**
```text
JIT-compiled pipeline output fields shape: (4, 1, 1, 16, 16)
```
