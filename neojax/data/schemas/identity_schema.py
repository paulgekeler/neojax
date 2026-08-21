"""Schema that simply extracts fields without modification."""

from typing import final

from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class IdentitySchema(BaseSchema):
    """Schema that simply returns the fields array without modifying it."""

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[Array, "..."]:
        """Returns the fields array.

        Args:
            bundle: The input DataBundle.
            reference_bundle: Ignored.

        Returns:
            The fields array.
        """
        if isinstance(bundle, DataBundle):
            return bundle.fields
        return bundle
