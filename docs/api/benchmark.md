# Benchmarking Neural Operators

The `neojax.benchmark` module provides utilities for running standardized and reproducible evaluations of neural operators.

---

## Motivation

Comparing neural operator results across papers is difficult in practice, for a few recurring reasons:

1. **Non-standardized normalization**: Papers apply different normalization schemes (global vs. per-channel, min-max, non-dimensionalization), and normalization choice alone can shift accuracy metrics by several percentage points.
2. **Custom simulation setups**: Datasets are often generated with different solvers, grid resolutions, domains, boundary values, or time step sizes, which makes direct comparison across papers unreliable.
3. **Rollout discrepancies**: Time-dependent problems are evaluated inconsistently — some papers report single-step error (predicting $t+1$ from ground truth at $t$), others report autoregressive rollout error over long horizons.
4. **Inconsistent metrics**: Definitions of relative $L^2$/$L^p$ error vary across papers, and spectral accuracy or physical-invariant conservation are often not reported at all.

`neojax.benchmark` addresses this by standardizing the pieces that make results comparable:

* **`DataBundle`** gives every dataset the same interface for coordinates, time, discretized fields, physical parameters, and boundary values (see `neojax.data.bundles.data_bundle`).
* **`BaseSchema`** decouples the dataset representation from model-specific input formats, so any JAX model (Equinox, Flax, or custom) can be benchmarked as long as a schema exists that maps `DataBundle` to its expected input.
* Steady-state and time-dependent problems are evaluated through separate evaluators (see [Evaluators](evaluators.md)) with a consistent metric interface, so single-step and rollout results aren't conflated.
* The dataset registry (see [Data Downloading](download.md)) fetches public datasets directly, removing per-paper simulation setup as a variable.

For an example of a benchmarking setup, see [here](../examples/04_benchmarking_pipeline.md).

---

## Registered Benchmarking Datasets

The registry contains **37 standardized PDE datasets** (including 20 **PDEgym** datasets from Hugging Face, 2 verification datasets from **Zenodo**, and 15 complete **PDEBench** datasets from Stuttgart's DaRUS repository).

You can choose any subset to benchmark with and add your own datasets on top if desired.

See the [download](download.md) page for details.

---

## Benchmark Runner

The high-level orchestrator that takes the benchmark configuration and executes the full benchmark run.

::: neojax.benchmark.runner.BenchmarkRunner

---

## Saving Results

We provide a utility `save_results` to serialize results to JSON or binary NumPy format.

::: neojax.benchmark.evaluators.base_evaluator.save_results
