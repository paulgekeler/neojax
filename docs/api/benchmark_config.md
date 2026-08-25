## Benchmark Configuration

The benchmark configuration (a nested dictionary) defines every model, dataset and component used for all benchmarking tasks. `neojax` then uses `pydantic` to validate all the schemas and instantiate all benchmark components.

A benchmarking configuration has to adhere to the following structure:

```python
config_dict = {
    # Benchmark config as outermost schema
    "benchmark_name": "<name>",
    # Define global settings (GlobalSettingsConfig)
    "global_settings": {
        "batch_size": <batch_size>,
        "output_dir": "<output_path>",
        "seed": <rng_seed>
    },
    # Then define each benchmark task (TaskConfig)
    "tasks": [
        {
            "name": "<task_name>",
            # Define datasets used for this task (DatasetConfig)
            "datasets": [
                {
                    "name": "<dataset_name>",
                    "type": "RawDataset",
                    "source": "<source>",
                    "path": "<dataset_path>",
                    "field_mapping": {
                        "fields": "<dataset_fieldname>",
                        ...
                    }
                    # and more optional settings
                    ...
                },
                ...
            ],
            # Define evaluator to use (EvaluatorConfig)
            "evaluator": {
                "type": "<evaluator_class>",
                # Define metrics to use in this task (ComponentConfig)
                "metrics": [
                    {
                        "type": "<metric_class>", "kwargs": {...}
                    },
                    ...
                ]
            },
            # Define models to benchmark in this task (ModelConfig)
            "models": [
                {
                    "name": "<model_name>",
                    "architecture": "<model_class>",
                    "framework": "custom",
                    "hyperparameters": {
                        ...
                    },
                    # Define a pipeline for each model (PipelineConfig)
                    "pipeline": {
                        # Define each schema for the pipeline (ComponentConfig)
                        "in_schema": {
                            "type": "<schema_class>",
                            "kwargs": {...}
                        },
                        "out_schema": {
                            "type": "<schema_class>",
                            "kwargs": {...}
                        }
                    }
                },
                ...
            ]
        },
        # And so on for all tasks to run
        ...
    ],
    ...
}
```

If you'd like to instantiate each config separately, instantiate from the innermost nested level outwards. 

### Configuring Benchmark

::: neojax.benchmark.config.BenchmarkConfig
    options:
        members: false

### Configuring Global Settings

::: neojax.benchmark.config.GlobalSettingsConfig
    options:
        members: false

### Configuring Tasks

::: neojax.benchmark.config.TaskConfig
    options:
        members: false

### Configuring Datasets

::: neojax.benchmark.config.DatasetConfig
    options:
        members: false

### Configuring Dataset Field Mappings

::: neojax.benchmark.config.FieldMappingConfig
    options:
        members: false

### Configuring Evaluators

::: neojax.benchmark.config.EvaluatorConfig
    options:
        members: false

### Configuring Models

::: neojax.benchmark.config.ModelConfig
    options:
        members: false

### Configuring Pipelines

::: neojax.benchmark.config.PipelineConfig
    options:
        members: false

### Configuring Normalizers

::: neojax.benchmark.config.NormalizerConfig
    options:
        members: false

### Configuring Metrics & Schemas

::: neojax.benchmark.config.NormalizerConfig
    options:
        members: false
