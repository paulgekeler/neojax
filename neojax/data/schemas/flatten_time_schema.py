"""Schema that flattens the time dimension into the channel dimension."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import JaxNpArray


@final
class FlattenTimeSchema(
    BaseSchema[DataBundle | Inexact[JaxNpArray, "..."], Inexact[JaxNpArray, "..."]]
):
    """Schema that flattens multiple time steps into the channel dimension.

    Often used for models predicting the next time step from a history of past
    time steps by stacking the past states as channels.

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
        if time_axis >= channel_axis:
            raise ValueError(
                "Time axis must be strictly before channel axis for simple flattening."
            )
        self.time_axis = time_axis
        self.channel_axis = channel_axis

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[JaxNpArray, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[JaxNpArray, "..."]:
        """Flattens time into channels.

        Args:
            bundle: The `DataBundle` or array.
            reference_bundle: Ignored. Kept for API compatibility.

        Returns:
            The array with the time dimension folded into the channel dimension.
        """
        if isinstance(bundle, DataBundle):
            fields = bundle.fields
        else:
            fields = bundle

        shape = list(fields.shape)
        time_dim = shape[self.time_axis]
        channel_dim = shape[self.channel_axis]

        # e.g. [batch, time, channel, x, y] -> [batch, time * channel, x, y]
        new_shape = (
            shape[: self.time_axis]
            + [time_dim * channel_dim]
            + shape[self.channel_axis + 1 :]
        )

        return jnp.reshape(fields, new_shape)
