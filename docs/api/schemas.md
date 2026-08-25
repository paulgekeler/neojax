# Schemas API

Schemas define deterministic structural transformations for physical data. Neural operators expect data in specific shapes (e.g. spatial coordinates appended to the channel dimension, time flattened into channels), and schemas are the adapters between a dataset's `DataBundle` format and whatever shape a given model architecture expects. `ComposedSchema` chains several schemas together into a single transformation.

### Diagram: Schema Pipeline

```mermaid
flowchart LR
    A[DataBundle] --> B[FlattenTimeSchema]
    B -->|Time flattened to channels| C[ConcatenateCoordsSchema]
    C -->|Coordinates appended| D[JAX Array]
    D --> E[Neural Operator]
```

!!! info
    All schemas accept both `DataBundle` objects and raw JAX arrays, so they can also be used to manipulate raw arrays from `RawDataset` without going through `DataBundle`.

---

::: neojax.data.schemas.IdentitySchema

::: neojax.data.schemas.FlattenTimeSchema

::: neojax.data.schemas.ConcatenateCoordsSchema
    options:
        members:
            - transform

::: neojax.data.schemas.BundleReconstructSchema
    options:
        members:
            - transform

::: neojax.data.schemas.ComposedSchema

::: neojax.data.schemas.BaseSchema