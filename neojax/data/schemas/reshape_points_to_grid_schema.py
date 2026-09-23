"""Schema that reconstructs a physical grid from pointwise predictions."""

from typing import final

import equinox as eqx
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import BundleOrArray


@final
class ReshapePointsToGridSchema(BaseSchema[BundleOrArray, DataBundle]):
    """Schema that reconstructs the spatial grid from a flattened array of point predictions."""

    time_steps: int = eqx.field(static=True, default=1)
    channels: int = eqx.field(static=True, default=1)

    @override
    def transform(
        self,
        bundle: BundleOrArray,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle:
        """Reshapes flat point outputs into a DataBundle grid.

        Args:
            bundle: The input array of point predictions (or existing DataBundle).
            reference_bundle: The original bundle, used for coordinates and other metadata.

        Returns:
            The reconstructed DataBundle.
        """
        if isinstance(bundle, DataBundle):
            return bundle

        outputs = bundle
        if reference_bundle is None or reference_bundle.coords is None:
            raise ValueError(
                "reference_bundle with coords is required to infer spatial shape."
            )

        # reference_bundle.coords is [d, *spatial]
        spatial_shape = reference_bundle.coords.shape[1:]

        # Outputs is [N] or [N, channels]
        # Reshape to [time_steps, channels, *spatial]
        fields = outputs.reshape(self.time_steps, self.channels, *spatial_shape)

        return DataBundle(
            coords=reference_bundle.coords,
            fields=fields,
            parameters=reference_bundle.parameters,
            bc_masks=reference_bundle.bc_masks,
            bc_values=reference_bundle.bc_values,
            edge_indices=reference_bundle.edge_indices,
        )
