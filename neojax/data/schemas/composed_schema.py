"""Schema that composes multiple schemas together."""

import warnings
from collections.abc import Sequence
from typing import final

import equinox as eqx
from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.schemas.flatten_to_points_schema import FlattenToPointsSchema
from neojax.data.schemas.mesh_schema import MeshInputSchema
from neojax.data.schemas.time_to_stationary_schema import TimeToStationarySchema


@final
class ComposedSchema(BaseSchema):
    """Schema that applies a sequence of schemas in order.

    Args:
        schemas: A tuple of schemas to compose. They will be applied sequentially
            in the order provided.

    Raises:
        ValueError: If `FlattenToPointsSchema` or `MeshInputSchema` are used as non-final
            schemas.

    Warns:
        UserWarning: If `TimeToStationarySchema` is used as non-final schema.

    ??? info "Internal Attributes"
        * **schemas** (`tuple[BaseSchema]`): Schemas to apply in sequentially in order.

    !!! warning "Known Design Flaw"
        Not all schemas may be used in arbitrary order inside a `ComposedSchema`.
        Have a look at the respective documentations for details on where to use each
        schema.
    """

    schemas: tuple[BaseSchema, ...] = eqx.field(static=True)

    def __init__(self, schemas: Sequence[BaseSchema]) -> None:
        for schema in schemas[:-1]:
            if isinstance(schema, TimeToStationarySchema):
                warnings.warn(
                    "Used 'TimeToStationarySchema' as non-final schema:\n"
                    "It causes divergent transformation branches.\n"
                    "It should only be used as final component of a 'ComposedSchema'.\n"
                    "Have a look at its documentation for details.",
                    stacklevel=1,
                )
            elif isinstance(schema, FlattenToPointsSchema):
                raise ValueError(
                    "Used 'FlattenToPointsSchema' as non-final schema:\n"
                    "Its 'transform' method has an incompatible return signature.\n"
                    "It can only be used as final component of a 'ComposedSchema'.\n"
                    "Have a look at its documentation for details.",
                )
            elif isinstance(schema, MeshInputSchema):
                raise ValueError(
                    "Used 'MeshInputSchema' as non-final schema:\n"
                    "Its 'transform' method has an incompatible return signature.\n"
                    "It can only be used as final component of a 'ComposedSchema'.\n"
                    "Have a look at its documentation for details.",
                )
            else:
                continue
        self.schemas = tuple(schemas)

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle | Inexact[Array, "..."]:
        """Applies each schema's transform sequentially.

        Args:
            bundle: The input DataBundle or PyTree.
            reference_bundle: The original DataBundle to inherit geometries from.

        Returns:
            The transformed bundle or PyTree.
        """
        if reference_bundle is None and isinstance(bundle, DataBundle):
            reference_bundle = bundle

        out = bundle
        for schema in self.schemas:
            out = schema.transform(out, reference_bundle=reference_bundle)
        return out
