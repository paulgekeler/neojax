"""Implementation of the MeshInputSchema."""

from typing import final

import equinox as eqx
from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class MeshInputSchema(BaseSchema):
    """Schema for mesh-based operators that maps a DataBundle to a dictionary of fields and coordinates.

    Returns a dictionary `{"u": u, "x_in": x_in}` where `u` has shape `(#batch, channels, num_nodes)`
    and `x_in` has shape `(#batch, num_nodes, dim)`. The batch axis before `time_axis`
    in `fields` is preserved. The batch axis of `coords` is inferred from its own rank
    as it may be shared across a batch.

    Args:
        time_axis: The axis index in the fields array corresponding to time. Defaults to 1.
        channel_axis: The axis index corresponding to the channel dimension. Defaults to 2.

    ??? info "Internal Attributes"
        * **time_axis** (`int`): Axis index of time dimensions.
        * **channel_axis** (`int`): Axis index of channel dimensions.

    !!! warning "Use in ComposedSchema"
        `MeshInputSchema` can only be used as final schema in a `ComposedSchema`.
        It doesn't fulfill the contract w.r.t. the return signature of the `BaseSchema.transform` method
        in its current implementation. This may be improved/changed in future versions.
    """

    time_axis: int = eqx.field(static=True)
    channel_axis: int = eqx.field(static=True)

    def __init__(self, time_axis: int = 1, channel_axis: int = 2) -> None:
        self.time_axis = time_axis
        self.channel_axis = channel_axis

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> dict[str, Inexact[Array, "..."]]:
        """Maps fields and coordinates to mesh-based operator format.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            Dictionary with inputs 'u' and 'x_in'.

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
            raise ValueError("bundle coords cannot be None for MeshInputSchema.")

        # fields shape: [*batch, time, channel, *spatial] -> [*batch, time * channel, num_nodes]
        shape = list(fields.shape)
        time_dim = shape[self.time_axis]
        channel_dim = shape[self.channel_axis]
        leading_shape = shape[: self.time_axis]

        # Flatten spatial dimensions
        num_nodes = 1
        for s in shape[self.channel_axis + 1 :]:
            num_nodes *= s

        u = fields.reshape(*leading_shape, time_dim * channel_dim, num_nodes)

        # coords shape: [*leading, d, *spatial] -> [*leading, d, num_nodes] -> [*leading, num_nodes, d]
        # Get coords d axis using the number of spatial dims of fields
        num_spatial = len(shape) - self.channel_axis - 1
        coords_shape = list(coords.shape)
        d_axis = len(coords_shape) - num_spatial - 1
        d = coords_shape[d_axis]
        coords_leading = coords_shape[:d_axis]

        x_in = coords.reshape(*coords_leading, d, -1).swapaxes(-1, -2)

        return {"u": u, "x_in": x_in}
