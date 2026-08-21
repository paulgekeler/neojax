"""Schema that flattens fields into 1D and coordinates into a list of points."""

from typing import final

from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class FlattenToPointsSchema(BaseSchema):
    """Schema that prepares inputs for pointwise models by flattening fields and coordinates.

    Returns a tuple `(u, y)` where `u` is the flattened fields array and `y` is a
    `[N_points, d_dim]` array of coordinates.
    """

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> tuple[Inexact[Array, "m_sensors"], Inexact[Array, "N d"]]:
        """Flattens fields into 1D and coords into point lists.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            Tuple of (flattened_fields, point_coordinates).
        """
        if isinstance(bundle, DataBundle):
            fields = bundle.fields
            coords = bundle.coords
        else:
            fields = bundle
            if reference_bundle is None:
                raise ValueError(
                    "reference_bundle is required when bundle is an array."
                )
            coords = reference_bundle.coords

        if coords is None:
            raise ValueError(
                "DataBundle coords cannot be None for FlattenToPointsSchema."
            )

        # fields shape: [time, channel, *spatial] -> [m_sensors]
        u = fields.reshape(-1)

        # coords shape: [d, *spatial] -> [d, N] -> [N, d]
        d = coords.shape[0]
        y = coords.reshape(d, -1).T

        return (u, y)
