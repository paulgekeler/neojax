"""Schema that concatenates coordinates into the field channel dimension."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class ConcatenateCoordsSchema(BaseSchema):
    """Schema that appends coordinate grids to the channel dimension of fields.

    This is commonly used as an input schema for Neural Operators (like FNO or UNO)
    that need explicit positional information.

    Args:
        channel_axis: The axis index in the fields array corresponding to the channel
            dimension where coordinates should be appended. Defaults to 2 (assuming
            [batch, time, channel, *spatial]).

    ??? info "Internal Attributes"
        * **channel_axis** (`int`): Axis index of channel dimensions. Default is 1.
    """

    channel_axis: int = eqx.field(static=True, default=1)

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[Array, "..."]:
        """Concatenates the coordinate grids into the fields' channel dimension.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            The modified fields array with coordinates appended.

        Raises:
            ValueError: If `reference_bundle` is None if `bundle` is an array.
            ValueError: If `bundle` or `reference_bundle` coords are None.
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
                "DataBundle coords cannot be None for ConcatenateCoordsSchema."
            )
        # Broadcast coords to [batch, time, dim, *spatial]

        # Ensure channel_axis is positive for indexing calculations
        chan_ax = (
            self.channel_axis
            if self.channel_axis >= 0
            else len(fields.shape) + self.channel_axis
        )

        # The dimensions after channel_axis are the spatial dimensions
        num_spatial = len(fields.shape) - chan_ax - 1

        if num_spatial < 0:
            raise ValueError(
                f"channel_axis {self.channel_axis} is out of bounds for fields with shape {fields.shape}"
            )

        # Extract d and batch_shape
        d = coords.shape[-num_spatial - 1] if num_spatial > 0 else coords.shape[-1]
        batch_shape = (
            coords.shape[: -num_spatial - 1] if num_spatial > 0 else coords.shape[:-1]
        )

        # Insert 1s for dimensions between batch and channel_axis
        num_ones_to_insert = chan_ax - len(batch_shape)
        if num_ones_to_insert < 0:
            raise ValueError(
                f"Batch shape of coords {batch_shape} is larger than channel_axis {chan_ax}."
            )

        spatial_shape = coords.shape[-num_spatial:] if num_spatial > 0 else ()
        reshape_dims = (*batch_shape, *(1,) * num_ones_to_insert, d, *spatial_shape)
        broadcast_coords = jnp.reshape(coords, reshape_dims)

        target_shape = (*fields.shape[:chan_ax], d, *fields.shape[chan_ax + 1 :])
        broadcast_coords = jnp.broadcast_to(broadcast_coords, target_shape)

        return jnp.concatenate([fields, broadcast_coords], axis=self.channel_axis)
