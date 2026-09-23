"""Implementation of the base schema."""

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx

from neojax.data.bundles.data_bundle import DataBundle

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class BaseSchema(eqx.Module, Generic[TIn, TOut]):
    """Abstract base schema.

    All custom schemas should inherit from BaseSchema.
    """

    @abstractmethod
    def transform(
        self,
        bundle_or_model_output: TIn,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> TOut:
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
