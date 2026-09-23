"""Schema that concatenates parameters into the field channels."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Inexact
from typing_extensions import override

from neojax.data.bundles import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import JaxNpArray


@final
class ConcatenateParamsSchema(
    BaseSchema[DataBundle | Inexact[JaxNpArray, "..."], Inexact[JaxNpArray, "..."]]
):
    """Schema that concatenates bundle `parameters` to the channel dimension of fields.

    This is commonly used when one or multiple additional `parameters` are used as
    model inputs next to the bundle `fields` (if they are not already in `fields`).
    E.g. there is a spatially varying velocity field which was passed as `parameters`.

    Args:
        channel_axis: The axis index in the `fields` array corresponding to the channel
            dimension where `parameters` should be appended. Defaults to 2 (assumes
            [batch, time, channel, *spatial]).

    ??? info "Internal Attributes"
        * **channel_axis** (`int`): Axis index of `fields` channel dimensions. Default is 2.

    !!! warning
        Currently only works for parameter fields, that is `parameters` with the same
        spatial dimensions as `fields`. Cannot be used to concatenate parameter scalars
        and `fields`. You need to define your own Schema for scalar `parameters`
        (e.g. broadcast parameters over spatial dimensions as well).
    """

    channel_axis: int = eqx.field(static=True, default=2)

    @override
    def transform(
        self,
        bundle: DataBundle | Inexact[JaxNpArray, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> Inexact[JaxNpArray, "..."]:
        """Concatenates the parameters into the fields channel dimension.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for `parameters` if bundle is an array.

        Returns:
            The modified fields array with `parameters` appended.

        Raises:
            ValueError: If `reference_bundle` is None if `bundle` is an array.
            ValueError: If `bundle` or `reference_bundle` `parameters` are None.
            ValueError: If channel_axis is out of bounds for given fields shape.
        """
        if isinstance(bundle, DataBundle):
            fields = bundle.fields
            parameters = bundle.parameters
        else:
            fields = bundle
            if reference_bundle is None:
                raise ValueError(
                    "reference_bundle is required when bundle is an array."
                )
            parameters = reference_bundle.parameters

        if parameters is None:
            raise ValueError(
                "DataBundle parameters cannot be None for ConcatenateParamsSchema."
            )

        # Ensure channel_axis is positive for indexing calculations
        chan_ax = (
            self.channel_axis
            if self.channel_axis >= 0
            else len(fields.shape) + self.channel_axis
        )

        # The dimensions after channel_axis are the spatial dimensions
        num_spatial = len(fields.shape) - chan_ax - 1

        has_channel_axis = len(parameters.shape) - num_spatial >= 2
        has_batch_axis = len(parameters.shape) - num_spatial >= 1
        has_time_axis = len(parameters.shape) - num_spatial >= 3

        if num_spatial < 0:
            raise ValueError(
                f"channel_axis {self.channel_axis} is out of bounds for fields with shape {fields.shape}"
            )

        # Assemble broadcastable parameter shape
        broadcast_shape = parameters.shape
        if not has_batch_axis:
            broadcast_shape = (1,) + broadcast_shape
        if not has_time_axis:
            broadcast_shape = (broadcast_shape[0], 1, *broadcast_shape[1:])
        if not has_channel_axis:
            broadcast_shape = (*broadcast_shape[:2], 1, *broadcast_shape[2:])

        broadcast_params = jnp.reshape(parameters, broadcast_shape)

        return jnp.concatenate([fields, broadcast_params], axis=self.channel_axis)
