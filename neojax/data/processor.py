"""Implementation of the BundleProcessor."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.normalizers.base_normalizer import BaseNormalizer
from neojax.data.schemas.base_schema import BaseSchema


def _squeeze_stats_batch_dim(normalizer: BaseNormalizer) -> BaseNormalizer:
    """Recursively squeezes the keepdims batch axis out of a normalizer's stats.

    Recurses into `ComposedNormalizer` so every normalizer's stats get squeezed,
    not just the outermost one. Only squeezes when the leading axis is actually
    size 1: not every normalizer's `compute_stats` reduces with `keepdims=True`
    (e.g. `PhysicsNormalizer`'s stat matches the input shape exactly, with no
    batch axis to strip), so a bare leading-axis squeeze would break those.

    Args:
        normalizer: A normalizer whose `compute_stats` has just been called.

    Returns:
        The normalizer with its stats' leading batch axis squeezed out.
    """

    def _squeeze_value(v):
        if isinstance(v, BaseNormalizer):
            return _squeeze_stats_batch_dim(v)
        if isinstance(v, tuple):
            return tuple(_squeeze_value(item) for item in v)
        # Only squeeze size 1 dims
        if eqx.is_array(v) and v.ndim > 0 and v.shape[0] == 1:
            return jnp.squeeze(v, axis=0)
        return v

    squeezed_stats = {k: _squeeze_value(v) for k, v in normalizer.stats.items()}
    return eqx.tree_at(lambda n: n.stats, normalizer, squeezed_stats)


class BundleProcessor(eqx.Module):
    """Orchestrates normalization and structural transformations for PDE data.

    Args:
        normalizers: A dictionary mapping DataBundle attribute names (e.g. 'fields')
            to their respective BaseNormalizer instances. Default is None.

    ??? info "Internal Attributes"
        These fields store the internal state of the processor.

        * **normalizers** (`dict[str, BaseNormalizer] | None`): Normalizer dictionary.
    """

    normalizers: dict[str, BaseNormalizer] | None = eqx.field(default=None)

    def transform(self, bundle: DataBundle, schema: BaseSchema) -> PyTree:
        """Prepares a DataBundle for the neural operator.

        Args:
            bundle: The raw DataBundle to be processed.
            schema: The schema that defines the structural transformation to model inputs.

        Returns:
            PyTree containing the model-ready inputs.
        """
        normalized_bundle = self._apply_normalizers(bundle, inverse=False)
        return schema.transform(normalized_bundle)

    def inverse_transform(
        self, model_outputs: PyTree, schema: BaseSchema, reference_bundle: DataBundle
    ) -> DataBundle:
        """Converts model outputs back to a physical DataBundle.

        Args:
            model_outputs: The raw predictions from the model.
            schema: The schema that maps model outputs back into a physical bundle format.
            reference_bundle: The original input bundle to inherit geometries
                and unmodified fields from.

        Returns:
            DataBundle representing the physical predictions in original scale.
        """
        pred_bundle = schema.transform(model_outputs, reference_bundle=reference_bundle)
        return self._apply_normalizers(pred_bundle, inverse=True)

    def _apply_normalizers(
        self, bundle: DataBundle, inverse: bool = False
    ) -> DataBundle:
        """Applies or inverts normalizers on the bundle's fields.

        Args:
            bundle: The DataBundle to normalize or de-normalize.
            inverse: Whether to apply the inverse transformation.

        Returns:
            A new DataBundle with scaled or de-scaled fields.
        """
        if self.normalizers is None:
            return bundle

        updates = {}
        for attr, normalizer in self.normalizers.items():
            val = getattr(bundle, attr)
            if val is not None:
                if inverse:
                    updates[attr] = normalizer.inverse_transform(val)
                else:
                    updates[attr] = normalizer.transform(val)

        return eqx.tree_at(
            lambda b: [getattr(b, k) for k in updates.keys()],
            bundle,
            list(updates.values()),
        )

    def compute_stats(self, dataset_bundle: DataBundle) -> "BundleProcessor":
        """Fits the normalizers on the provided dataset statistics.

        Args:
            dataset_bundle: A single DataBundle containing the entire dataset
                or training subset, which is used to compute the global statistics.

        Returns:
            A new BundleProcessor instance with updated normalizer statistics.
        """
        if self.normalizers is None:
            return self

        new_normalizers = {}
        for attr, normalizer in self.normalizers.items():
            val = getattr(dataset_bundle, attr)
            if val is not None:
                new_norm = normalizer.compute_stats(val)
                new_normalizers[attr] = _squeeze_stats_batch_dim(new_norm)
            else:
                new_normalizers[attr] = normalizer

        return eqx.tree_at(lambda p: p.normalizers, self, new_normalizers)
