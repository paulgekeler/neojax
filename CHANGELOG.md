# Changelog

All notable changes to **neojax** will be documented in this file.

## [Released]

## [0.2.1] - 2026-09-21

No additions. Simple version bump for pypi compatibility.

## [0.2.0] - 2026-09-21

### Added

#### Schemas Module
- `BaseSchema` interface, plus `IdentitySchema`, `FlattenTimeSchema`, `ConcatenateCoordsSchema`, `FlattenToPointsSchema`, `ReshapePointsToGridSchema`, `BundleReconstructSchema`, and `MeshInputSchema` for transforming `DataBundle`s into model-specific input formats and back.
- `ComposedSchema` for chaining schemas together.
- `GraphTupleInputSchema`/`GraphTupleOutputSchema` for graph-based operators (requires the optional `rigno` package).

#### BundleProcessor
- `BundleProcessor` combines normalization and schema transforms into a single `transform`/`inverse_transform` pipeline, with `compute_stats` for fitting normalizers on a dataset.

#### Datasets Module
- `BaseDataset`, `BundleDataset`, and `RawDataset`, with `from_pdebench`/`from_pdegym` loaders.
- `neojax.data.download`: registry-based dataset downloading (Hugging Face, Zenodo, DaRUS Dataverse, plain HTTP) with resumable downloads, retries, and checksum verification. 37 datasets registered across PDEgym, PDEBench, and verification sources.

#### Benchmark Module
- `BenchmarkConfig` (Pydantic-based configuration), `BenchmarkRunner`, `SteadyStateEvaluator`, `TimeDependentEvaluator`, and `save_results` for standardized, reproducible neural operator benchmarking, including against non-neojax models.

#### Metrics Module
- `BaseMetric`, `ComposedMetric`, `LpMetric`/`RelativeLpMetric`, `SobolevMetric`, `MSEMetric`/`RMSEMetric`/`R2Metric`, and physics-consistency metrics `BoundaryConsistencyMetric`, `ConservationMetric`, `ResidualMetric`.
- `is_learnable_metric_weight` for training with learnable metric/loss weights.

#### Training Module
- `Trainer` and `TrainState` for functional JAX training loops with checkpointing.
- Diagnostics to uncover gradient anomalies and training failures.

#### Tensor Module
- `BaseTensor` interface, plus `DenseTensor`, `CPTensor`, `TuckerTensor`, and `TTTensor` factorized spectral weight representations, with `separable` support.

#### Models
- UNO, GeoFNO
- Abstract `BaseNO` class, with common functionality: `size()`, `astype()`, `profile_compile()`

#### Other
- `neojax.data.generation.generate_burgers_1d` for generating 1D Burgers' equation training data.
- `neojax.utils.torch_converters`: `load_torch_weights_into_fno`, `load_torch_weights_into_geo_fno` for importing pretrained PyTorch `neuraloperator` weights.
- `Resampler`, `GeoMapNd`, `GeoSpectralConvNd`, and `make_skip_connection` building blocks.

### Changed
- FNO/TFNO: added channel-MLP dropout, configurable FFT normalization, Hermitian symmetry enforcement, and resolution scaling.
- `SpectralConvNd`/`FNOBlocks` refactored to support all tensor factorizations (Dense, CP, Tucker, TT) through a single implementation.

### Removed
- `neojax.losses` (`BaseLoss`, `ComposedLoss`, `LpLoss`, `RelativeLpLoss`, `SobolevLoss`) — superseded by `neojax.metrics`, which covers the same functionality plus regression and physics-consistency metrics.
- `neojax.nn.tfno_blocks`, `neojax.nn.tucker_spectral_conv` — merged into `neojax.nn.fno_blocks`/`neojax.nn.spectral_conv`.
