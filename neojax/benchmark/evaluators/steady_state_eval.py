"""Implementation of the steady-state evaluator class."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from neojax.benchmark.evaluators.base_evaluator import BaseEvaluator
from neojax.data.bundles.data_bundle import DataBundle


@final
class SteadyStateEvaluator(BaseEvaluator):
    """Evaluator for steady-state PDE problems.

    Transforms data, runs the model, reconstructs the physical outputs, and evaluates
    all configured metrics for a single sample.

    Args:
        metrics: A dictionary mapping metric names to BaseMetric instances.
        in_schema: Input schema for the model.
        out_schema: Output schema for the model.
        processor: Optional BundleProcessor. Default is None.
    """

    @override
    def evaluate_sample(
        self, model: eqx.Module, bundle: DataBundle
    ) -> dict[str, Float[Array, ""]]:
        """Evaluates a single steady-state sample.

        Processes the input sample through normalizers and structural schemas, executes
        the neural operator, and computes the error metrics.

        Args:
            model: The neural operator model to evaluate.
            bundle: A single unbatched DataBundle.

        Returns:
            Dictionary of computed metric values.
        """
        # Transform inputs using processor and schema
        if self.processor is not None:
            inputs = self.processor.transform(bundle, schema=self.in_schema)
        else:
            inputs = self.in_schema.transform(bundle)

        # Forward pass through model
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        elif isinstance(inputs, tuple):
            outputs = model(*inputs)
        else:
            outputs = model(inputs)

        # Reconstruct prediction bundle and denormalize
        if self.processor is not None:
            pred_bundle = self.processor.inverse_transform(
                outputs, schema=self.out_schema, reference_bundle=bundle
            )
        else:
            pred_bundle = self.out_schema.transform(outputs, reference_bundle=bundle)

        # Prepare batched inputs for the metrics
        target_batched = bundle.fields[None, ...]
        pred_batched = pred_bundle.fields[None, ...]
        x_batched = jax.tree_util.tree_map(
            lambda leaf: leaf[None, ...] if isinstance(leaf, jnp.ndarray) else leaf,
            inputs,
        )
        bc_values_batched = (
            bundle.bc_values[None, ...] if bundle.bc_values is not None else None
        )

        results = {}
        for name, metric_fn in self.metrics.items():
            results[name] = metric_fn(
                model=model,
                target=target_batched,
                x=x_batched,
                pred=pred_batched,
                coords=bundle.coords,
                bc_masks=bundle.bc_masks,
                bc_values=bc_values_batched,
            )
        return results
