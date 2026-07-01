# Benchmarking Neural Operators

The `neojax.benchmark` module provides utilities for running standardized and reproducible evaluations of neural operators.

---

## Why a Unified PDE Benchmark is Vital

In neural operator and scientific machine learning (SciML) research, a major bottleneck is the **lack of standardized evaluation protocols**. Unlike computer vision (with ImageNet) or natural language processing (with GLUE), SciML lacks a universally adopted testing suite. As a result:

1. **Non-Standardized Normalization**: Different papers apply custom, ad-hoc normalizations (e.g., standardizing globally vs. per-channel, min-max scaling, or non-dimensionalization). A minor change in normalization can change accuracy metrics by several percentage points, masking whether a new model architecture is actually better.
2. **Custom Simulation Setup**: Researchers often generate their own datasets using differing solvers, grid resolutions, domains, boundary values, or time step sizes. This makes it almost impossible to compare a model from one paper directly with a model from another.
3. **Rollout Discrepancies**: In time-dependent problems, some papers evaluate single-step error (predicting $t+1$ from ground truth at $t$), while others evaluate autoregressive rollout error over long horizons. 
4. **Inconsistent Metrics**: Metric calculations vary, such as using differing definitions of relative $L^2$ or $L^p$ losses, or ignoring spectral accuracy and physical invariant conservation.

### How `Neojax` Addresses This

The `neojax.benchmark` module addresses these challenges by building towards a unified framework:

* **Unified Data Wrappers (`OperatorInput` / `OperatorOutput`)**: Standardizing the interface for coordinates, time, discretized function fields, physical parameters, and boundary values across all datasets (see `neojax.data.wrappers`).
* **Separation of Concerns via Schemas (`BaseSchema`)**: Decoupling the structured operator inputs from model-specific raw array formats, allowing any Jax model (whether Equinox, Flax, or custom) to be benchmarked. As long as your model defines a `Schema` attribute that inherits from `BaseSchema`, you can use the benchmarking suite with your own models.
* **Robust, Standardized Evaluations**: Separating stationary (steady-state) and time-dependent evaluations (single-step vs. long-term autoregressive rollouts) with unified metrics.
* **Reproducible Data Registry**: Automatically fetching public, peer-reviewed datasets with strict checksum checks, eliminating simulation inconsistencies. See `neojax.data.download` for more details on downloading datasets.

---

## Registered Benchmarking Datasets

The registry contains **37 standardized PDE datasets** (including 20 **PDEgym** datasets from Hugging Face, 2 verification datasets from **Zenodo**, and 15 complete **PDEBench** datasets from Stuttgart's DaRUS repository).

See the [download](download.md) page for details.

---


