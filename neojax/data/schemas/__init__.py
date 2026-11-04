# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Schema components for preprocessing DataBundles into neural network inputs."""

from neojax.data.schemas.base_schema import BaseSchema as BaseSchema
from neojax.data.schemas.bundle_reconstruct_schema import (
    BundleReconstructSchema as BundleReconstructSchema,
)
from neojax.data.schemas.composed_schema import ComposedSchema as ComposedSchema
from neojax.data.schemas.concatenate_coords_schema import (
    ConcatenateCoordsSchema as ConcatenateCoordsSchema,
)
from neojax.data.schemas.concatenate_params_schema import (
    ConcatenateParamsSchema as ConcatenateParamsSchema,
)
from neojax.data.schemas.flatten_time_schema import (
    FlattenTimeSchema as FlattenTimeSchema,
)
from neojax.data.schemas.flatten_to_points_schema import (
    FlattenToPointsSchema as FlattenToPointsSchema,
)
from neojax.data.schemas.graph_tuple_schema import (
    GraphTupleInputSchema as GraphTupleInputSchema,
)
from neojax.data.schemas.graph_tuple_schema import (
    GraphTupleOutputSchema as GraphTupleOutputSchema,
)
from neojax.data.schemas.identity_schema import IdentitySchema as IdentitySchema
from neojax.data.schemas.mesh_schema import MeshInputSchema as MeshInputSchema
from neojax.data.schemas.reshape_points_to_grid_schema import (
    ReshapePointsToGridSchema as ReshapePointsToGridSchema,
)
from neojax.data.schemas.time_to_stationary_schema import (
    TimeToStationarySchema as TimeToStationarySchema,
)

__all__ = [
    "BaseSchema",
    "BundleReconstructSchema",
    "ComposedSchema",
    "ConcatenateCoordsSchema",
    "FlattenTimeSchema",
    "IdentitySchema",
    "FlattenToPointsSchema",
    "ReshapePointsToGridSchema",
    "MeshInputSchema",
    "GraphTupleInputSchema",
    "GraphTupleOutputSchema",
    "ConcatenateParamsSchema",
    "TimeToStationarySchema",
]
