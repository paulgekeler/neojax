"""The core Benchmark Runner orchestration module."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "pyyaml is required for the benchmarking module. "
        "Install it with `pip install pyyaml`."
    ) from e

from neojax.benchmark.config import BenchmarkConfig
from neojax.benchmark.factories import build_component, build_model, build_schema
from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.datasets.raw_dataset import RawDataset
from neojax.data.processor import BundleProcessor


class BenchmarkRunner:
    """Orchestrates the entire benchmarking suite across datasets, pipelines, and models.

    Args:
        config: The parsed `BenchmarkConfig` object.
        custom_model_builders: Optional mapping of model names to custom callables
            defining how given non-neojax models should be built. Default is `None`.
            Each callable should return a "pure" Jax callable, meaning e.g. Flax models
            should be wrapped in a lambda around `flax_model.apply()`.

    ??? info "Attributes"
        * **config** (`BenchmarkConfig`): The benchmark configuration.
        * **custom_model_builders** (`dict[str, Callable] | None`): Optional mapping of model
            names to custom callables defining how given non-neojax models should be built.
            Default is `None`.

    Examples:
        ```python
        from jaxtyping import PRNGKeyArray
        from neojax.models.geo_fno import GeoFNO
        from neojax.benchmark.config import BenchmarkConfig
        from neojax.benchmark.runner import BenchmarkRunner

        config_dict = {...}
        benchmark_config = BenchmarkConfig(**config_dict)

        def build_custom_geo_fno(config: Any, key: PRNGKeyArray, dataset: Any) -> Callable:
            return GeoFNO(
                key=key,
                in_channels=config.hyperparameters["in_channels"],
                out_channels=config.hyperparameters["out_channels"],
                hidden_channels=config.hyperparameters["hidden_channels"],
                n_layers=config.hyperparameters["n_layers"],
                modes=config.hyperparameters["modes"],
                grid_resolution=config.hyperparameters["grid_resolution"],
                use_coord_projection=config.hyperparameters.get("use_coord_projection",
                True),
            )

        custom_model_builders = {
            "GeoFNO": build_custom_geo_fno
        }

        runner = BenchmarkRunner(benchmark_config, custom_model_builders)
        results = runner.run()
        ```
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        custom_model_builders: dict[str, Callable[[Any], Callable]] | None = None,
    ) -> None:
        self.config = config
        self.custom_model_builders = custom_model_builders or {}

    def run(self) -> dict[str, Any]:
        """Executes the benchmarking suite.

        Iterates through all tasks, loads datasets, fits normalizers on the fly if needed,
        builds models, and evaluates them.

        Returns:
            A dictionary containing the aggregated benchmark results.
        """
        results: dict[str, Any] = {
            "benchmark_name": self.config.benchmark_name,
            "tasks": {},
        }
        output_dir = Path(self.config.global_settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for task_cfg in self.config.tasks:
            task_results: dict[str, Any] = {}
            print(f"Starting task: {task_cfg.name}")

            for dataset_cfg in task_cfg.datasets:
                print(f"  -> Loading dataset: {dataset_cfg.name}")
                dataset = self._load_dataset(dataset_cfg)

                # Get dataset length to compute splits
                n_samples = len(dataset)
                test_start = math.floor(dataset_cfg.test_split[0] * n_samples)
                test_end = math.floor(dataset_cfg.test_split[1] * n_samples)

                dataset_results: dict[str, Any] = {}

                for model_cfg in task_cfg.models:
                    print(f"    -> Evaluating model: {model_cfg.name}")

                    # Setup Processor Pipeline
                    normalizers = {}
                    for field, norm_cfg in model_cfg.pipeline.normalizers.items():
                        normalizers[field] = build_component(
                            "neojax.data.normalizers", norm_cfg
                        )

                    processor = BundleProcessor(normalizers=normalizers)

                    # Setup Schemas
                    in_schema = build_schema(model_cfg.pipeline.in_schema)
                    out_schema = build_schema(model_cfg.pipeline.out_schema)

                    # Compute statistics on the fly if requested
                    needs_stats = any(
                        n.compute_on_fly
                        for n in model_cfg.pipeline.normalizers.values()
                    )
                    if needs_stats and dataset_cfg.train_split is not None:
                        train_start = math.floor(dataset_cfg.train_split[0] * n_samples)
                        train_end = math.floor(dataset_cfg.train_split[1] * n_samples)
                        train_batch = dataset[train_start:train_end]
                        # Convert to DataBundle if it's a dict (e.g. from RawDataset)
                        train_bundle = (
                            in_schema(train_batch)
                            if isinstance(train_batch, dict)
                            else train_batch
                        )
                        processor = processor.compute_stats(train_bundle)
                    elif needs_stats:
                        print(
                            "       WARNING: compute_on_fly is true but no train_split provided!"
                        )

                    # Setup Evaluator
                    metrics_dict = {}
                    for m_cfg in task_cfg.evaluator.metrics:
                        if isinstance(m_cfg, str):
                            from neojax.benchmark.config import ComponentConfig

                            m_cfg = ComponentConfig(type=m_cfg)
                        # Instantiate the metric
                        metric_instance = build_component("neojax.metrics", m_cfg)
                        metrics_dict[m_cfg.type] = metric_instance

                    evaluator_kwargs = task_cfg.evaluator.kwargs.copy()
                    evaluator_kwargs.update(
                        {
                            "metrics": metrics_dict,
                            "in_schema": in_schema,
                            "out_schema": out_schema,
                            "processor": processor,
                        }
                    )

                    if task_cfg.evaluator.autoregressive_steps is not None:
                        evaluator_kwargs["autoregressive_steps"] = (
                            task_cfg.evaluator.autoregressive_steps
                        )

                    from neojax.benchmark.config import ComponentConfig

                    evaluator_cfg = ComponentConfig(
                        type=task_cfg.evaluator.type, kwargs=evaluator_kwargs
                    )
                    evaluator = build_component(
                        "neojax.benchmark.evaluators", evaluator_cfg
                    )

                    # Instantiate Model
                    # Use a fixed key for initialization
                    key = jax.random.PRNGKey(42)
                    model = build_model(
                        model_cfg,
                        key=key,
                        dataset=dataset,
                        custom_model_builders=self.custom_model_builders,
                    )

                    # Execute Evaluation
                    eval_output = evaluator.evaluate_dataset(
                        model=model,
                        dataset=dataset,
                        batch_size=self.config.global_settings.batch_size,
                        eval_start=test_start,
                        eval_end=test_end,
                    )

                    # Only store summary in the top-level results dict
                    dataset_results[model_cfg.name] = eval_output["summary"]

                task_results[dataset_cfg.name] = dataset_results

            results["tasks"][task_cfg.name] = task_results

        # Save to YAML
        output_file = output_dir / f"{self.config.benchmark_name}_results.yaml"
        with open(output_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        print(f"Benchmarking complete. Results saved to {output_file}")
        return results

    def _load_dataset(self, config: Any) -> Any:
        """Helper to load a dataset from config."""
        mapping = config.field_mapping.model_dump(exclude_none=True)
        if config.type == "BundleDataset":
            if config.source == "pdebench":
                return BundleDataset.from_pdebench(config.path, field_mapping=mapping)
            elif config.source == "pdegym":
                return BundleDataset.from_pdegym(config.path, field_mapping=mapping)
        elif config.type == "RawDataset":
            if config.source == "pdebench":
                return RawDataset.from_pdebench(config.path, field_mapping=mapping)
            elif config.source == "pdegym":
                return RawDataset.from_pdegym(config.path, field_mapping=mapping)

        raise ValueError(
            f"Unsupported dataset type/source: {config.type}/{config.source}"
        )
