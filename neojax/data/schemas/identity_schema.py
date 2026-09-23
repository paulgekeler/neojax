"""Schema that simply extracts fields without modification."""

from typing import final

from jaxtyping import Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import BundleOrArray, JaxNpArray


@final
class IdentitySchema(BaseSchema[BundleOrArray, Inexact[JaxNpArray, "..."]]):
    """Schema that simply returns the fields array without modifying it."""

    @override
    def transform(
        self,
        bundle: BundleOrArray,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[JaxNpArray, "..."]:
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
