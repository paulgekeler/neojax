"""Implementation of unstructured graph input and output tuple schemas."""

from typing import Any, final

import jax.numpy as jnp
from jaxtyping import Inexact
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import BundleOrArray, JaxNpArray


@final
class GraphTupleInputSchema(BaseSchema[BundleOrArray, dict[str, Any]]):
    """Schema to prepare inputs for external graph-based or region interaction neural operators.

    Transforms a DataBundle into a dictionary containing a structured `Inputs` NamedTuple.

    ??? info "Internal Attributes"
        None
    """

    @override
    def transform(
        self,
        bundle: BundleOrArray,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> dict[str, Any]:
        """Prepares structured model input NamedTuple.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            Dictionary containing 'inputs' (Inputs NamedTuple).
        """
        # Lazy imports without rigno installation
        from rigno.models.operator import Inputs

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
                "DataBundle coords cannot be None for GraphTupleInputSchema."
            )

        # Flat coordinates representation (num_nodes, dim)
        x_in = coords.T
        if x_in.shape[1] == 1:
            # Promote 1D mesh to 2D for Delaunay graph building
            coords_2d = jnp.stack(
                [x_in[:, 0], jnp.sin(x_in[:, 0] * 2 * jnp.pi) * 0.1], axis=1
            )
        else:
            coords_2d = x_in

        # fields shape: [time, channel, nodes] -> [batch=1, time, nodes, channel]
        u = jnp.expand_dims(jnp.moveaxis(fields, 1, -1), axis=0)
        x_inp = coords_2d[None, None, ...]
        x_out = coords_2d[None, None, ...]

        inputs = Inputs(u=u, c=None, x_inp=x_inp, x_out=x_out, t=0.0, tau=1.0)
        return {"inputs": inputs}


@final
class GraphTupleOutputSchema(BaseSchema[Inexact[JaxNpArray, "..."], DataBundle]):
    """Schema to reconstruct a DataBundle from the output of external graph/unstructured models.

    Translates an output tensor of shape `[batch, time, num_nodes, channel]`
    back into a DataBundle with fields of shape `[time, channel, num_nodes]`.
    """

    @override
    def transform(
        self,
        outputs: Inexact[JaxNpArray, "..."],
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle:
        """Translates unstructured graph output back to a DataBundle with standard fields shape.

        Args:
            outputs: Outputs returned by the model (shape `[1, time, num_nodes, channel]`).
            reference_bundle: Reference bundle used to restore coordinates and metadata.

        Returns:
            A `DataBundle` with fields of shape `[time, channel, num_nodes]`.

        Raises:
            ValueError: If `reference_bundle` is None.
        """
        if reference_bundle is None:
            raise ValueError("GraphTupleOutputSchema requires a reference_bundle.")

        # output shape: [1, time, num_nodes, channel] -> [time, channel, num_nodes]
        fields = jnp.moveaxis(outputs.squeeze(axis=0), -1, 1)

        return DataBundle(
            fields=fields,
            coords=reference_bundle.coords,
            edge_indices=reference_bundle.edge_indices,
            bc_masks=reference_bundle.bc_masks,
            bc_values=reference_bundle.bc_values,
            parameters=reference_bundle.parameters,
        )
