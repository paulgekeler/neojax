# Datasets API

`neojax` provides dataset classes for loading physical simulation and PDE data, in two formats:

1. **`BundleDataset`**: Returns typed `DataBundle` objects, packaging fields, coordinates, boundary conditions, and parameters into a single structured PyTree. Use this to work with `Schemas` and `BundleProcessor` for normalization and reshaping.
2. **`RawDataset`**: Returns plain Python dictionaries of raw JAX arrays (e.g. `{"fields": ..., "coords": ...}`). Use this to write your own pre-processing logic instead of going through `DataBundle` and `Schemas`.

### Diagram: Dataset Workflows

```mermaid
graph TD
    A[PDEBench / PDEGym HDF5] -->|Load| B[BundleDataset]
    A -->|Load| C[RawDataset]
    
    subgraph datasets [neojax Datasets]
        B
        C
    end
    
    B -->|Returns DataBundle| D[BundleProcessor + Schemas]
    C -->|Returns Dict of Arrays| E[Manual User Pre-processing]
    
    D --> F[Neural Operator / Custom Model]
    E --> F
```

---

::: neojax.data.datasets.base_dataset.BaseDataset

::: neojax.data.datasets.bundle_dataset.BundleDataset

::: neojax.data.datasets.raw_dataset.RawDataset
