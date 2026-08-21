"""Pydantic configuration schemas for the Benchmark Runner."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldMappingConfig(BaseModel):
    """Configuration for mapping raw datasets to DataBundle keys.

    Attributes:
        fields: Name or list of names for fields.
        coords: Name or list of names for coordinates.
        parameters: Optional name or list of names for parameters.
        bc_masks: Optional name or list of names for boundary condition masks.
        bc_values: Optional name or list of names for boundary condition values.
        edge_indices: Optional name or list of names for edge indices.
    """

    fields: str | list[str]
    coords: str | list[str]
    parameters: str | list[str] | None = None
    bc_masks: str | list[str] | None = None
    bc_values: str | list[str] | None = None
    edge_indices: str | list[str] | None = None


class DatasetConfig(BaseModel):
    """Configuration for a dataset used in benchmarking.

    Attributes:
        name: A unique identifier for the dataset in this benchmark.
        type: Dataset class name (e.g., 'BundleDataset', 'RawDataset').
        source: Source type (e.g., 'pdebench', 'pdegym').
        path: Path to the dataset directory or file.
        field_mapping: Dictionary mapping for DataBundle keys.
        train_split: Optional tuple of [start, end] fractions (e.g., (0.0, 0.8))
            for computing normalizer statistics on the fly.
        test_split: Tuple of [start, end] fractions (e.g., (0.8, 1.0)) for evaluation.
    """

    name: str
    type: str
    source: str
    path: str | Path
    field_mapping: FieldMappingConfig
    train_split: tuple[float, float] | None = None
    test_split: tuple[float, float] = (0.0, 1.0)


class ComponentConfig(BaseModel):
    """Generic configuration for dynamic components like schemas or metrics.

    Attributes:
        type: The class name of the component to instantiate.
        kwargs: Additional keyword arguments to pass to the constructor.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class NormalizerConfig(BaseModel):
    """Configuration for normalizers within the pipeline.

    Attributes:
        type: The class name of the normalizer (e.g., 'UnitGaussianNormalizer').
        compute_on_fly: Whether to fit this normalizer on the `train_split` of the dataset.
        kwargs: Additional arguments for the normalizer constructor.
    """

    type: str
    compute_on_fly: bool = False
    kwargs: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for the data processing pipeline.

    Attributes:
        in_schema: Input schema component(s). Can be a single config or a list.
        out_schema: ComponentConfig for the output.
        normalizers: A dictionary mapping DataBundle keys to NormalizerConfigs.
    """

    in_schema: ComponentConfig | list[ComponentConfig]
    out_schema: ComponentConfig
    normalizers: dict[str, NormalizerConfig] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Configuration for a neural operator model to be benchmarked.

    Attributes:
        name: A unique identifier for the model in the benchmark.
        architecture: The class name of the model (e.g., 'FNO', 'UNet').
        framework: The framework the model is implemented in ('neojax', 'custom').
            If 'custom', ensure to pass `custom_model_builders` to `BenchmarkRunner`.
        checkpoint_path: Path to the Orbax checkpoint directory.
        hyperparameters: Dictionary of hyperparameters to pass to the model constructor.
        pipeline: The data processing pipeline required by this model.
    """

    name: str
    architecture: str
    framework: str = "neojax"
    checkpoint_path: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    pipeline: PipelineConfig
    vmap_in_axes: Any | None = None
    vmap_out_axes: Any = 0


class EvaluatorConfig(BaseModel):
    """Configuration for the evaluator used in a task.

    Attributes:
        type: The class name of the evaluator ('SteadyStateEvaluator' or 'TimeDependentEvaluator').
        metrics: A list of metric names or ComponentConfigs.
        autoregressive_steps: Number of rollout steps (for time_dependent evaluator).
    """

    type: str
    metrics: list[str | ComponentConfig]
    autoregressive_steps: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    """Configuration for a benchmark task.

    A task applies the same evaluator across a collection of datasets and models.

    Attributes:
        name: A unique identifier for the task.
        evaluator: The evaluator configuration.
        datasets: A list of dataset configurations to evaluate.
        models: A list of model configurations to benchmark.
    """

    name: str
    evaluator: EvaluatorConfig
    datasets: list[DatasetConfig]
    models: list[ModelConfig]


class GlobalSettingsConfig(BaseModel):
    """Global settings for the benchmark run.

    Attributes:
        batch_size: The default batch size for evaluation.
        output_dir: The directory where results should be saved.
        seed: Random seed for reproducibility.
    """

    batch_size: int
    output_dir: str | Path
    seed: int


class BenchmarkConfig(BaseModel):
    """Root configuration schema for a Benchmark Runner suite.

    Attributes:
        benchmark_name: The name of the overall benchmark suite.
        global_settings: Global configuration settings.
        tasks: A list of benchmarking tasks to execute.
    """

    benchmark_name: str
    global_settings: GlobalSettingsConfig = Field(default_factory=GlobalSettingsConfig)
    tasks: list[TaskConfig]
