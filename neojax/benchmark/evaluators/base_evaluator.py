"""Implementation of the base evaluator class."""

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.datasets.base_dataset import BaseDataset
from neojax.data.processor import BundleProcessor
from neojax.data.schemas.base_schema import BaseSchema
from neojax.metrics.base_metric import BaseMetric


class BaseEvaluator(eqx.Module):
    """Abstract base evaluator for benchmarking neural operators.

    Provides a unified evaluation pipeline over dataset batches with support for progress
    checkpointing and resuming.

    Args:
        metrics: A dictionary mapping metric names to BaseMetric instances.
        in_schema: Input schema for the model.
        out_schema: Output schema for the model.
        processor: Optional BundleProcessor. Default is None.

    ??? info "Attributes"
        * **metrics** (`dict[str, BaseMetric]`): Dictionary of metrics.
        * **in_schema** (`BaseSchema`): Input schema.
        * **out_schema** (`BaseSchema`): Output schema.
        * **processor** (`BundleProcessor | None`): Data bundle processor.
    """

    metrics: dict[str, BaseMetric]
    in_schema: BaseSchema
    out_schema: BaseSchema
    processor: BundleProcessor | None = eqx.field(default=None)

    def __init__(
        self,
        metrics: dict[str, BaseMetric],
        in_schema: BaseSchema,
        out_schema: BaseSchema,
        processor: BundleProcessor | None = None,
    ) -> None:
        self.metrics = metrics
        self.in_schema = in_schema
        self.out_schema = out_schema
        self.processor = processor

    @abstractmethod
    def evaluate_sample(
        self, model: eqx.Module, bundle: DataBundle
    ) -> dict[str, Float[Array, ""]]:
        """Evaluates a single sample.

        Must be implemented by subclasses.

        Args:
            model: The neural operator model.
            bundle: A single unbatched DataBundle.

        Returns:
            Dictionary of computed metric values.
        """
        ...

    def evaluate_dataset(
        self,
        model: eqx.Module,
        dataset: BaseDataset,
        batch_size: int = 32,
        eval_start: int = 0,
        eval_end: int | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 10,
    ) -> dict[str, Any]:
        """Runs the evaluation pipeline over a subset of the dataset in batches.

        Args:
            model: The neural operator model to evaluate.
            dataset: The dataset containing samples to evaluate. Must be a BaseDataset.
            batch_size: Number of samples per evaluation batch. Default is 32.
            eval_start: Start index for evaluation. Default is 0.
            eval_end: End index for evaluation. Defaults to len(dataset).
            checkpoint_path: Optional path to save/load progress checkpoints.
                Default is None.
            checkpoint_interval: Interval (in batch steps) to save progress checkpoints.
                Default is 10.

        Returns:
            Dictionary containing 'summary' statistics and 'details' lists.
        """
        # Determine evaluation range
        num_samples = len(dataset)
        if eval_end is None:
            eval_end = num_samples

        start_idx = eval_start
        accumulated_results = {name: [] for name in self.metrics.keys()}

        # Load checkpoint if exists
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                checkpoint = self.load_checkpoint(checkpoint_path)
                start_idx = checkpoint["index"]
                accumulated_results = checkpoint["results"]

        # Helper to construct batch bundle axes mapping
        def get_bundle_in_axes(b: DataBundle) -> DataBundle:
            where = lambda x: (
                x.coords,
                x.fields,
                x.parameters,
                x.bc_masks,
                x.bc_values,
                x.edge_indices,
            )

            replace = (
                0 if b.coords is not None else None,
                0 if b.fields is not None else None,
                0 if b.parameters is not None else None,
                0 if b.bc_masks is not None else None,
                0 if b.bc_values is not None else None,
                0 if b.edge_indices is not None else None,
            )
            return eqx.tree_at(where, b, replace, is_leaf=lambda x: x is None)

        steps_since_checkpoint = 0
        for i in range(start_idx, eval_end, batch_size):
            batch_end_idx = min(i + batch_size, eval_end)

            # Slice batch dynamically using dataset
            batch_data = dataset[i:batch_end_idx]

            # If it's a dict (e.g., from RawDataset), map it to DataBundle
            if isinstance(batch_data, dict):
                batch_bundle = self.in_schema(batch_data)
            elif isinstance(batch_data, DataBundle):
                batch_bundle = batch_data
            else:
                raise TypeError(f"Unsupported batch data type: {type(batch_data)}")

            # JIT-compiled batch evaluation using vmap
            vmapped_eval = jax.vmap(
                lambda b: self.evaluate_sample(model, b),
                in_axes=(get_bundle_in_axes(batch_bundle),),
            )
            jitted_eval = jax.jit(vmapped_eval)

            batch_results = jitted_eval(batch_bundle)

            # Accumulate results
            for name, vals in batch_results.items():
                accumulated_results[name].extend(np.array(vals).tolist())

            steps_since_checkpoint += 1
            if (
                checkpoint_path is not None
                and steps_since_checkpoint >= checkpoint_interval
            ):
                self.save_checkpoint(
                    checkpoint_path, index=batch_end_idx, results=accumulated_results
                )
                steps_since_checkpoint = 0

        # Remove checkpoint file upon successful completion
        if checkpoint_path is not None and checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError:
                pass

        # Compute summary statistics
        summary = {}
        for name, vals in accumulated_results.items():
            arr = np.array(vals)
            summary[f"{name}_mean"] = float(np.mean(arr))
            summary[f"{name}_std"] = float(np.std(arr))

        return {"summary": summary, "details": accumulated_results}

    def _stack_data_bundles(self, bundles: list[DataBundle]) -> DataBundle:
        """Stacks a list of single DataBundles into a batched DataBundle."""
        first = bundles[0]
        fields = jnp.stack([b.fields for b in bundles], axis=0)
        parameters = (
            jnp.stack([b.parameters for b in bundles], axis=0)
            if first.parameters is not None
            else None
        )
        bc_values = (
            jnp.stack([b.bc_values for b in bundles], axis=0)
            if first.bc_values is not None
            else None
        )

        where = lambda b: (b.fields, b.parameters, b.bc_values)

        replace = (fields, parameters, bc_values)
        return eqx.tree_at(where, first, replace, is_leaf=lambda x: x is None)

    def save_checkpoint(
        self, filepath: Path, index: int, results: dict[str, list[float]]
    ) -> None:
        """Saves evaluation progress to a JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"index": index, "results": results}, f, indent=4)

    def load_checkpoint(self, filepath: Path) -> dict[str, Any]:
        """Loads evaluation progress from a JSON file."""
        with open(filepath) as f:
            return json.load(f)


def save_results(results: dict[str, Any], filepath: str | Path) -> None:
    """Saves evaluation results to a file.

    Args:
        results: The dictionary of results (containing 'summary' and 'details').
        filepath: The path to save the file. Supports .json and .npz formats.

    Raises:
        ValueError: If the given file format is unsupported.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.suffix == ".json":

        def serialize(obj):
            if isinstance(obj, (np.ndarray, jnp.ndarray)):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(x) for x in obj]
            return obj

        with open(filepath, "w") as f:
            json.dump(serialize(results), f, indent=4)
    elif filepath.suffix == ".npz":
        details = results.get("details", {})
        summary = results.get("summary", {})
        arrays = {f"details_{k}": np.array(v) for k, v in details.items()}
        arrays["summary_json"] = json.dumps(summary)
        np.savez(filepath, **arrays)
    else:
        raise ValueError("Unsupported file format. Must be .json or .npz")
