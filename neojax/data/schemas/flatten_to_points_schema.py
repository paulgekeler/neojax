"""Schema that flattens fields into 1D and coordinates into a list of points."""

from typing import final

import equinox as eqx
from jaxtyping import Array, Inexact, Real
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class FlattenToPointsSchema(BaseSchema):
    """Schema that prepares inputs for pointwise models by flattening fields and coordinates.

    Returns a tuple `(u, y)` where `u` has shape `(*leading, m_sensors)`, combining the
    time, channel and spatial dimensions into a single flat axis, and `y` has shape
    `(*leading, N, d)` with `N` the number of spatial points. The batch axis before `time_axis`
    in `fields` is preserved. The batch axis of `coords` is inferred from its own rank
    as it may be shared across a batch.

    ??? info "Internal Attributes"
        * **time_axis** (`int`): Axis index of time dimensions.
        * **channel_axis** (`int`): Axis index of channel dimensions.

    !!! warning "Use in ComposedSchema"
        `FlattenToPointsSchema` can only be used as final schema in a `ComposedSchema`.
        It doesn't fulfill the contract w.r.t. the return signature of the `BaseSchema.transform` method
        in its current implementation. This may be improved/changed in future versions.
    """

    time_axis: int = eqx.field(static=True)
    channel_axis: int = eqx.field(static=True)

    def __init__(self, time_axis: int = 1, channel_axis: int = 2) -> None:
        if time_axis >= channel_axis:
            raise ValueError(
                "Time axis must be strictly before channel axis for simple flattening."
            )
        self.time_axis = time_axis
        self.channel_axis = channel_axis

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> tuple[Inexact[Array, "..."], Real[Array, "..."]]:
        """Flattens fields into 1D and coords into point lists.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            Tuple of (flattened_fields, point_coordinates).

        Raises:
            ValueError: If `reference_bundle` is `None` if `bundle` is an array.
            ValueError: If `bundle.coords` or `reference_bundle.coords` are `None`.
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

        # fields shape: [*leading, time, channel, *spatial] -> [*leading, m_sensors]
        shape = list(fields.shape)
        leading_shape = shape[: self.time_axis]
        u = fields.reshape(*leading_shape, -1)

        # coords shape: [*leading, d, *spatial] -> [*leading, d, N] -> [*leading, N, d]
        # Get coords d axis using the number of spatial dims of fields
        num_spatial = len(shape) - self.channel_axis - 1
        coords_shape = list(coords.shape)
        d_axis = len(coords_shape) - num_spatial - 1
        d = coords_shape[d_axis]
        coords_leading = coords_shape[:d_axis]

        y = coords.reshape(*coords_leading, d, -1).swapaxes(-1, -2)

        return (u, y)
