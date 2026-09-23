"""Schema that extracts slices from time axis for stationary problems."""

from collections.abc import Sequence
from copy import deepcopy
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import override

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.base_schema import BaseSchema
from neojax.data.types import BundleOrArray


@final
class TimeToStationarySchema(BaseSchema[BundleOrArray, DataBundle]):
    """Schema that extracts two time slices from all time axes of a `DataBundle`.

    Can be used to convert a time-dependent problem or problem with multiple time steps
    into a stationary problem, that is one time step as model input and another as ground truth.

    Args:
        time_axis: The axis index in all arrays corresponding to the time
            dimension. Defaults to 1 (assumes [batch, time, channel, *spatial]).
        time_slice_indices: The two indices along `time_axis` to extract.
            Defaults to the first and last index (0, -1). The first index
            has to preceed the second even for negative indices,
            e.g. (-2, -5) are invalid slices.

    ??? info "Internal Attributes"
        * **time_axis** (`int`): The axis index in all arrays corresponding to the time
            dimension. Defaults to 1.
        * **time_slice_indices** (`Sequence[int]`): The two indices along `time_axis` to extract.
            Defaults to the first and last index (0, -1).

    !!! warning "Validation of time slice indices"
        The first index of `time_slice_indices` must preceed the second after modulo time axis length.
        Also, they must not index outside of time axis length.
        Otherwise, the behavior will be undefined (uses JAX out-of-bounds fill-in with `NaN`s).
        E.g. `(-5, -2)` or `(2, -2)` correct for axis length 6, `(-3, -4)`, `(-1, 3)` or `(0, 7)` not correct.
        The last of which would result in an NaN-filled slice.

    !!! warning "Use in ComposedSchema"
        `TimeToStationarySchema` in its current implementation causes divergent tranformation branches:
        We extract two time slices, the first of which we would like to use as model input, the second as
        ground truth. Now, if we needed to apply further transformations to the input only
        (e.g. concatenating `parameters` to `fields`), we'd also apply these transformations to the ground truth
        inside a `ComposedSchema`! This is undesirable. Until `BundleProcessor` supports divergent
        branches, `TimeToStationarySchema` should only be used as a final component in a `ComposedSchema`.
        Or apply transformation schemas separately, as in example notebook 5.
    """

    time_axis: int = eqx.field(static=True, default=1)
    time_slice_indices: Sequence[int] = eqx.field(static=True, default=(0, -1))

    @override
    def transform(
        self,
        bundle: BundleOrArray,
        /,
        reference_bundle: DataBundle | None = None,
    ) -> DataBundle:
        """Extracts two time slices from a `DataBundle` or `fields` at the given `time_slice_indices` from `time_axis`.

        Args:
            bundle: The input `DataBundle` or array.
            reference_bundle: The original bundle, used for coordinates if bundle is an array.

        Returns:
            The modified `DataBundle` where each array time axis
            only has the extracted time index slices.

        Raises:
            ValueError: If length of `time_slice_indices` is not 2.
            ValueError: If `reference_bundle` is `None` if `bundle` is an array.
        """
        if len(self.time_slice_indices) != 2:
            raise ValueError(
                "Length of 'time_slice_indices' has to be 2 for stationary problems."
            )

        attrs_w_time_axis = ["fields", "bc_values", "parameters"]

        if isinstance(bundle, DataBundle):
            num_spatial = len(bundle.fields.shape[3:])
        else:
            num_spatial = len(bundle.shape[3:])

        def has_time_axis(arr: Array, attr_name: str) -> bool:
            if attr_name in attrs_w_time_axis[:-1]:
                # fields and bc_values type hints mandate time axis
                return True
            else:
                # parameters time axis depends on shape
                return len(arr.shape) - num_spatial >= 3

        if isinstance(bundle, DataBundle):
            bundle_kwargs = deepcopy(vars(bundle))
        else:
            # Handle fields separately and bundle same as before
            if reference_bundle is None:
                raise ValueError(
                    "'reference_bundle' cannot be None if 'bundle' is an array."
                )

            bundle_kwargs = deepcopy(vars(reference_bundle))
            bundle_kwargs.update({"fields": bundle})
        for attr_name, attr in bundle_kwargs.items():
            if attr is not None and has_time_axis(attr, attr_name):
                # Slice both indices with fancy indexing
                bundle_kwargs[attr_name] = jnp.take(
                    attr, jnp.array(self.time_slice_indices), axis=self.time_axis
                )
            else:
                continue
        # Return attrs wo time axis unchanged
        return DataBundle(**bundle_kwargs)
