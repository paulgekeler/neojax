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

    Returns a dictionary `{"u": u, "x_in": x_in}` where `u` has shape `(channels, num_nodes)`
    and `x_in` has shape `(num_nodes, dim)`.

    Args:
        time_axis: The axis index in the fields array corresponding to time. Defaults to 1.
        channel_axis: The axis index corresponding to the channel dimension. Defaults to 2.

    ??? info "Internal Attributes"
        * **time_axis** (`int`): Axis index of time dimensions.
        * **channel_axis** (`int`): Axis index of channel dimensions.
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
            raise ValueError("DataBundle coords cannot be None for MeshInputSchema.")

        # fields shape: [time, channel, *spatial] -> [time * channel, num_nodes]
        shape = list(fields.shape)
        time_dim = shape[self.time_axis]
        channel_dim = shape[self.channel_axis]

        # Flatten spatial dimensions
        num_nodes = 1
        for s in shape[self.channel_axis + 1 :]:
            num_nodes *= s

        u = fields.reshape(time_dim * channel_dim, num_nodes)

        # coords shape: [d, *spatial] -> [d, num_nodes] -> [num_nodes, d]
        d = coords.shape[0]
        x_in = coords.reshape(d, -1).T

        return {"u": u, "x_in": x_in}
