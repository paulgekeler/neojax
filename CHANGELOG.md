# Changelog

All notable changes to **neojax** will be documented in this file.

## [Unreleased] -2026-11-06

### Added
- Support for Python 3.14 and a weekly github workflow `upstream_deps.yml` that checks compatibility with new dependency versions
- Local patch for outstanding jet primitive rules: `neojax.utils._jet_rules` now defines the needed jet rules for higher order derivative Sobolev metrics and neojax models. Rules are registered automatically upon import of `neojax.metrics.sobolev_metrics`.
- All metrics now support PyTree inputs for models (needed in e.g. GiNO, GAOT and GeoFNO):
  - `BaseMetric` now defines `vmap_model` and `batch_axes` static helper methods. They vmap the model over different PyTree inputs (all metrics use them except `SobolevMetric`).


### Changes
- All `Tensor`s now store complex weights as stacked (`real, imaginary`) pairs to avoid the need of conjugating gradients when training spectral-based models.
  - `BaseTensor` now has `to_complex/real` static methods for conversion of weights in computations
- `SpectralConvNd` now correctly projects real inputs when `enforce_hermitian_symmetry=True` using static method `_hermitian_part`. For 2D and higher, the previous logic removed real signal.
- Minor fixes to example notebooks 1 and 5.
- All layers using FFTs or interpolation now upcast inputs to single precision and downcast the outputs again.
- Future versions of neojax now have lower bounds on the versions of its core dependencies. Also `jaxlib` is no longer required (`jax` already mandates it).
- All future versions of neojax are published under Apache-2.0 license instead of MIT.
  - This makes copying code blocks from other libraries with the same license easier (e.g. segment-ops from `jraph`).
  - All python files now include the required license header and a `pre-commit` hook checks and adds it.
- As part of the jet rules patch, `RNO` is the only model that still raises a `KeyError`.
- Sobolev metric changes:
  - Tests now include verification that the jet patch works and `RNO` still raises.
  - Now takes an optional `wrt` argument specifying the pytree leaf w.r.t. which to differentiate.
  - The respective sobolev method takes any `Callable` as model now: `_bind_wrt` method returns a callable with the closed over additional kwargs of model (`_select_wrt` finds `wrt` in PyTree).

## [Released]

## [0.5.0] -2026-11-03

### Added
- `GAOT` Geometry Aware Operator Transformer and its components.
  - `GAOT` components: `MAGNOBlock`, `NeighborAttention`, `GeometricEmbedding`, `edge_drop`, `MultiHeadAttention` (grouped-query, optional rotary embeddings), `TransformerBlock`, `SwiGLU`, `UViT`, `patchify`/`unpatchify`, `RMSNorm`, `RotaryEmbedding`.
  - `neighbor_truncation`: fraction of queries whose padded neighbor list is full (and may thus have been truncated).
  - `SinusoidalEmbedding` has a new `"linear"` embedding type (the node position encoding of GAOT).
- `conftest.py` utility `assert_reduced_precision_close_to_full`, which compares the outputs of a callable when run in reduced/lower and full precision

### Changes
- `IntegralTransform` takes an optional `attention` (`NeighborAttention`) that reweighs the neighbors (Attentional GNO). `GNOBlock` exposes it with `attention_type` and `attention_dim`, and accepts `"linear"` as `pos_embedding_type`. The defaults are unchanged.
- Updated roadmap in `README`.
- Parallel testing session support with `nox`.

## [0.4.0] -2026-10-27

### Added
- `apply_layer` function in `neojax.models.utils` (also added the respective submodule): Needed to seemlessly pass inputs through layers which may or may not be stateful. If stateful (`model.is_stateful()`), passes the state to the respective layer, otherwise returns state as is.
- `CNO` Neural Operator (classic variant of the reference with alias-free filtered activations) for n-dim inputs.
- `CNO` components: `CNOConvBlock`, `CNOResidualBlock`, `CNOLiftProjectBlock`, `FilteredLReLU`, `KaiserFIRFilter` (matches `scipy.signal.firwin` behaviour).
- `FilteredLReLU` is the pure JAX version of StyleGAN3's CUDA version. Assembles a static filter with `KaiserFIRFilter` that is then compiled into the XLA graph.
- `scipy` as a dev dependency to check `KaiserFIRFilter` matches `scipy.signal.firwin`.
- `Transolver` Neural Operator (Transformer with `PhysicsAttention`)
  - `Transolver` components: `PhysicsAttention`, `ReferenceGridEmbeddingNd`, `ChannelLayerNorm`, `TransolverBlock`
- Restructured documentation to not show the exposed components of the submodules in the navigation bar
- Geometry Informed Neural Operator `GiNO`
  - `GiNO` components: `IntegralTransform`, `GNOBlock`, mollifier functions (`bump`, `quartic`, ...), neighbor search (`radius_neighbor_search`), `SinusoidalEmbedding` nerf- and transformer-style, 
- Multigrid Neural Operator `MgNO`
  - `MgNO` components: `MgConv`, `MgRestrict`, `MgIter`
- Minor changes to the documentation
- Recurrent Neural Operator `RNO`
- Add array broadcasting over PyTrees with `neojax.utils.tree_add_broadcast_to`

### Changed
- `PointwiseMLP` now accepts `use_bias` to disable the bias
- Fixed `PointwiseMLP` broadcasting issue with einsum: if input channels and first layer weight shape don't match, `__call__` raises now
- Removed `jnp.repeat` on grid channels in `GridEmbeddingNd`. It now correctly appends one grid channel per spatial dimension.
- Spectral-based models now size their first lifting `PointwiseMLP` layer according to the new `n_extra_channels` property of `GridEmbeddingNd`
- Fix ci workflow failure caused by large arrays used in `test_resample.py`
- Fix doc workflow by correctly declaring all third library hidden objects in hippogriffe `extra_public_objects` in `mkdocs.yaml`

## [0.3.0] -2026-09-22

### Added
- Added `nox` as dev tool to run test suite against all supported Python versions.
  - Add `noxfile.py` to repository root.
- Add Python version matrix to `ci.yaml` workflow.
- Update `CONTRIBUTING.md` with use of `nox`.

### Removed
- Dropped support for Python `3.10`.

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
