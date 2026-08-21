"""Implementation of the base schema."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import PyTree

from neojax.data.bundles.data_bundle import DataBundle


class BaseSchema(eqx.Module):
    """Abstract base schema.

    All custom schemas should inherit from BaseSchema.
    """

    @abstractmethod
    def transform(
        self,
        bundle_or_model_output: DataBundle | PyTree,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle | PyTree:
        """Transforms a data bundle into the needed model format.

        Args:
            bundle_or_model_output: If schema is an input schema,
                an instance of `DataBundle`,
                if schema is an output schema, any pytree the model outputs.
            reference_bundle: If schema is an output schema, the original input
                `DataBundle` to inherit geometries and unmodified fields from.
                Ignored for input schemas.

        Returns:
            If schema is an input schema, returns a raw array or pytree in model format.
            If schema is an output schema, returns a `DataBundle` instance.
        """
        ...
