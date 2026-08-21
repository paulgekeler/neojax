"""Factories for dynamically instantiating benchmark components from configuration."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

from neojax.benchmark.config import ComponentConfig, ModelConfig, NormalizerConfig


def build_component(
    module_name: str, config: ComponentConfig | NormalizerConfig
) -> Any:
    """Dynamically instantiates a class from a given module using the configuration.

    Args:
        module_name: The fully qualified name of the module (e.g., 'neojax.metrics').
        config: The configuration object containing `type` and `kwargs`.

    Returns:
        An instance of the specified class.

    Raises:
        AttributeError: If the class is not found in the module.
    """
    module = importlib.import_module(module_name)
    cls = getattr(module, config.type)
    return cls(**config.kwargs)


def build_schema(config: ComponentConfig | list[ComponentConfig]) -> Any:
    """Builds a schema or composed schema from configuration.

    Args:
        config: A single `ComponentConfig` or a list of `ComponentConfig`.

    Returns:
        A `BaseSchema` instance.
    """
    import neojax.data.schemas as schemas

    if isinstance(config, list):
        components = [build_component("neojax.data.schemas", c) for c in config]
        return schemas.ComposedSchema(components)
    return build_component("neojax.data.schemas", config)


def build_model(
    config: ModelConfig,
    key: jax.Array,
    dataset: Any = None,
    custom_model_builders: dict[str, Callable] | None = None,
) -> Any:
    """Instantiates a model architecture and optionally loads an Orbax checkpoint.

    Args:
        config: The `ModelConfig` detailing architecture, hyperparameters, and checkpoint.
        key: A JAX PRNG key for initialization.
        dataset: The currently loaded dataset, which may be needed by custom models (e.g. for coords).
        custom_model_builders: A dictionary mapping architecture names to custom builder functions.

    Returns:
        The instantiated model (with weights restored if a checkpoint was provided).

    Raises:
        NotImplementedError: If the framework is not supported for direct instantiation.
        ValueError: If a custom framework is specified but no builder is provided.
    """
    if config.framework.lower() == "custom":
        if (
            custom_model_builders is None
            or config.architecture not in custom_model_builders
        ):
            raise ValueError(
                f"Custom model builder for '{config.architecture}' not found in custom_model_builders."
            )
        model = custom_model_builders[config.architecture](config, key, dataset)
    elif config.framework.lower() != "neojax":
        # Placeholder for other frameworks like flax/haiku
        raise NotImplementedError(
            f"Framework '{config.framework}' is not yet supported."
        )
    else:
        # Instantiate architecture
        import neojax.models as models

        cls = getattr(models, config.architecture)

        # Models often take a key argument
        model = cls(key=key, **config.hyperparameters)

    # Load Orbax checkpoint if provided
    if config.checkpoint_path is not None:
        try:
            import orbax.checkpoint as ocp
        except ImportError as e:
            raise ImportError(
                "orbax-checkpoint is required to load models in the benchmarking suite. "
                "Install it with `pip install orbax-checkpoint`."
            ) from e

        checkpoint_path = Path(config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' does not exist."
            )

        checkpointer = ocp.PyTreeCheckpointer()
        model = checkpointer.restore(checkpoint_path.resolve().as_posix(), item=model)

    if config.vmap_in_axes is not None:

        class VmappedModel(eqx.Module):
            model: eqx.Module
            in_axes: Any = eqx.field(static=True)
            out_axes: Any = eqx.field(static=True)

            def __call__(self, *args, **kwargs):
                return jax.vmap(
                    self.model, in_axes=self.in_axes, out_axes=self.out_axes
                )(*args, **kwargs)

        model = VmappedModel(
            model, in_axes=config.vmap_in_axes, out_axes=config.vmap_out_axes
        )

    return model
