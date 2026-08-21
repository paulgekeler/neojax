"""Implementation of the time-dependent evaluator class."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import override

from neojax.benchmark.evaluators.base_evaluator import BaseEvaluator
from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.processor import BundleProcessor
from neojax.data.schemas.base_schema import BaseSchema
from neojax.metrics.base_metric import BaseMetric


@final
class TimeDependentEvaluator(BaseEvaluator):
    """Evaluator for time-dependent PDE problems using autoregressive rollouts.

    Iterates over temporal trajectories by feeding predicted future steps back as inputs
    via jax.lax.scan.

    Args:
        metrics: A dictionary mapping metric names to BaseMetric instances.
        in_schema: Input schema for the model.
        out_schema: Output schema for the model.
        history_steps: Number of lookback/history steps required for the model input.
        processor: Optional BundleProcessor. Default is None.
    """

    history_steps: int = eqx.field(static=True)

    def __init__(
        self,
        metrics: dict[str, BaseMetric],
        in_schema: BaseSchema,
        out_schema: BaseSchema,
        history_steps: int,
        processor: BundleProcessor | None = None,
    ) -> None:
        super().__init__(
            metrics=metrics,
            in_schema=in_schema,
            out_schema=out_schema,
            processor=processor,
        )
        self.history_steps = history_steps

    @override
    def evaluate_sample(
        self, model: eqx.Module, bundle: DataBundle
    ) -> dict[str, Float[Array, ""]]:
        """Evaluates a single time-dependent sample using autoregressive rollout.

        Args:
            model: The neural operator model to evaluate.
            bundle: A single unbatched DataBundle.

        Returns:
            Dictionary of computed metric values.
        """
        init_history = bundle.fields[: self.history_steps]

        # Execute a single dummy step to inspect the prediction steps count (P) statically
        dummy_history_bundle = DataBundle(
            coords=bundle.coords,
            fields=init_history,
            parameters=bundle.parameters,
            bc_masks=bundle.bc_masks,
            bc_values=(
                bundle.bc_values[: self.history_steps]
                if bundle.bc_values is not None
                else None
            ),
            edge_indices=bundle.edge_indices,
        )
        if self.processor is not None:
            dummy_inputs = self.processor.transform(
                dummy_history_bundle, schema=self.in_schema
            )
        else:
            dummy_inputs = self.in_schema.transform(dummy_history_bundle)
        if isinstance(dummy_inputs, dict):
            dummy_outputs = model(**dummy_inputs)
        elif isinstance(dummy_inputs, tuple):
            dummy_outputs = model(*dummy_inputs)
        else:
            dummy_outputs = model(dummy_inputs)

        if self.processor is not None:
            dummy_pred_step = self.processor.inverse_transform(
                dummy_outputs,
                schema=self.out_schema,
                reference_bundle=dummy_history_bundle,
            )
        else:
            dummy_pred_step = self.out_schema.transform(
                dummy_outputs, reference_bundle=dummy_history_bundle
            )

        P = dummy_pred_step.fields.shape[0]

        T = bundle.fields.shape[0]
        num_rollout_steps = (T - self.history_steps) // P

        def scan_fn(carry, step_idx):
            start_t = step_idx * P
            bc_val_slice = None
            if bundle.bc_values is not None:
                bc_val_slice = jax.lax.dynamic_slice(
                    bundle.bc_values,
                    (start_t, 0) + (0,) * (bundle.bc_values.ndim - 2),
                    (self.history_steps, bundle.bc_values.shape[1])
                    + bundle.bc_values.shape[2:],
                )

            history_bundle = DataBundle(
                coords=bundle.coords,
                fields=carry,
                parameters=bundle.parameters,
                bc_masks=bundle.bc_masks,
                bc_values=bc_val_slice,
                edge_indices=bundle.edge_indices,
            )

            if self.processor is not None:
                inputs = self.processor.transform(history_bundle, schema=self.in_schema)
            else:
                inputs = self.in_schema.transform(history_bundle)
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            elif isinstance(inputs, tuple):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)

            if self.processor is not None:
                pred_step_bundle = self.processor.inverse_transform(
                    outputs,
                    schema=self.out_schema,
                    reference_bundle=history_bundle,
                )
            else:
                pred_step_bundle = self.out_schema.transform(
                    outputs, reference_bundle=history_bundle
                )

            pred_field_step = pred_step_bundle.fields

            if self.history_steps == P:
                next_carry = pred_field_step
            else:
                next_carry = jnp.concatenate([carry[P:], pred_field_step], axis=0)

            return next_carry, pred_field_step

        _, pred_steps = jax.lax.scan(
            scan_fn, init_history, jnp.arange(num_rollout_steps)
        )

        # Flatten rollout steps and prediction channels dimension
        c = pred_steps.shape[2]
        spatial_shape = pred_steps.shape[3:]
        rollout_fields = jnp.reshape(
            pred_steps, (num_rollout_steps * P, c) + spatial_shape
        )

        target_fields = bundle.fields[
            self.history_steps : self.history_steps + num_rollout_steps * P
        ]

        bc_values_rollout = None
        if bundle.bc_values is not None:
            bc_values_rollout = bundle.bc_values[
                self.history_steps : self.history_steps + num_rollout_steps * P
            ]

        # Prepare batched inputs for the metrics
        target_batched = target_fields[None, ...]
        pred_batched = rollout_fields[None, ...]
        x_batched = jax.tree_util.tree_map(
            lambda leaf: leaf[None, ...] if isinstance(leaf, jnp.ndarray) else leaf,
            dummy_inputs,
        )
        bc_values_batched = (
            bc_values_rollout[None, ...] if bc_values_rollout is not None else None
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
