"""Schema that reconstructs a DataBundle from raw array predictions."""

from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema


@final
class BundleReconstructSchema(BaseSchema):
    """Schema that takes raw model outputs and places them back into a DataBundle.

    Args:
        unflatten_time_steps: If provided, unflattens the channel dimension back
            into (time_steps, channels). Use this if your model output is flattened
            time*channels. Default is None, meaning do not unflatten time steps.
        channel_axis: The axis index of the channel dimension. Defaults to 1 (assuming
            [batch, time*channel, *spatial] since time was flattened).
            Default is 1, since time axis is 0.

    ??? info "Internal Attributes"
        * **unflatten_time_steps** (`int | None`): Number of time steps to unflatten. Default is None.
        * **channel_axis** (`int`): Axis index of the channel dimension. Default is 1.
    """

    unflatten_time_steps: int | None = eqx.field(static=True, default=None)
    channel_axis: int = eqx.field(static=True, default=1)

    @override
    def transform(
        self,
        model_output: Inexact[Array, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle:
        """Reconstructs a DataBundle from an array.

        Args:
            model_output: The predicted raw array fields.
            reference_bundle: The bundle containing original geometry (coords, edges, masks).

        Returns:
            A new DataBundle containing the predicted fields and reference geometry.

        Raises:
            ValueError: If `reference_bundle` is None.
            ValueError: If channel dimension is not divisible by time steps.
        """
        if reference_bundle is None:
            raise ValueError("BundleReconstructSchema requires a reference_bundle.")

        if self.unflatten_time_steps is not None:
            # [batch, time*channel, *spatial] -> [batch, time, channel, *spatial]
            shape = list(model_output.shape)
            time_chan_dim = shape[self.channel_axis]

            if time_chan_dim % self.unflatten_time_steps != 0:
                raise ValueError(
                    f"Channel dimension {time_chan_dim} is not divisible by time steps {self.unflatten_time_steps}."
                )

            channels = time_chan_dim // self.unflatten_time_steps
            new_shape = (
                shape[: self.channel_axis]
                + [self.unflatten_time_steps, channels]
                + shape[self.channel_axis + 1 :]
            )
            model_output = jnp.reshape(model_output, new_shape)

        # Create a new DataBundle with the new fields and everything else from reference
        return DataBundle(
            fields=model_output,
            coords=reference_bundle.coords,
            edge_indices=reference_bundle.edge_indices,
            bc_masks=reference_bundle.bc_masks,
            bc_values=reference_bundle.bc_values,
            parameters=reference_bundle.parameters,
        )
