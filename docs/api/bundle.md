# Data Bundle API

The `DataBundle` class is designed to wrap arbitrary PDE data and expose a unified interface for `Schemas` and the `BundleProcessor`.

In conjunction with the structural transformations of `Schemas` and the `BundleProcessor`, `DataBundle` enables plug-and-play setups of different JAX Neural Operator architectures (even non-neojax or non-equinox models) and different PDE datasets.

The documentation on [Datasets](datasets.md) and [Schemas](schemas.md) provides a schematic overview of how these components interact in common workflows. The [Data Pipeline Example](../examples/03_data_pipeline.md) demonstrates what a setup might look like for a standard training workflow.

!!! info
    Using `DataBundle` as a wrapper for your data/model inputs is entirely optional. All neojax models are designed to work with raw JAX arrays.

::: neojax.data.bundles.data_bundle.DataBundle