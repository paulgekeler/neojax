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

import jax.numpy as jnp

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.mesh_schema import MeshInputSchema
from neojax.tests.conftest import assert_filter_jittable


class TestMeshInputSchema:
    def test_mesh_input_schema(self):
        coords = jnp.arange(20).reshape(2, 10).astype(jnp.float32)
        # fields shape: [time=2, channel=3, nodes=10]
        fields = jnp.ones((2, 3, 10))
        bundle = DataBundle(fields=fields, coords=coords)

        # Set time_axis=0, channel_axis=1 to match fields shape
        schema = MeshInputSchema(time_axis=0, channel_axis=1)
        res = schema.transform(bundle)
        assert isinstance(res, dict)
        assert "u" in res
        assert "x_in" in res
        assert res["u"].shape == (6, 10)
        assert res["x_in"].shape == (10, 2)
        assert_filter_jittable(lambda b: schema.transform(b)["u"], bundle)
