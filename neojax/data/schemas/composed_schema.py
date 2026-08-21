"""Schema that composes multiple schemas together."""

from collections.abc import Sequence
from typing import final

import equinox as eqx
from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class ComposedSchema(BaseSchema):
    """Schema that applies a sequence of schemas in order.

    Args:
        schemas: A tuple of schemas to compose. They will be applied sequentially
            in the order provided.

    ??? info "Internal Attributes"
        * **schemas** (`tuple[BaseSchema]`): Schemas to apply in sequentially in order.
    """

    schemas: tuple[BaseSchema, ...] = eqx.field(static=True)

    def __init__(self, schemas: Sequence[BaseSchema]) -> None:
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
