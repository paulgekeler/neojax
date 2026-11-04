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
from neojax.data.schemas.identity_schema import IdentitySchema
from neojax.tests.conftest import assert_filter_jittable


class TestIdentitySchema:
    def test_bundle_input(self):
        coords = jnp.arange(8).reshape(1, 8).astype(jnp.float32)
        fields = jnp.ones((2, 1, 8))
        bundle = DataBundle(fields=fields, coords=coords)

        schema = IdentitySchema()
        out = schema.transform(bundle)
        assert jnp.array_equal(out, fields)

    def test_array_input(self):
        fields = jnp.arange(8).reshape(2, 4).astype(jnp.float32)
        schema = IdentitySchema()
        out = schema.transform(fields)
        assert jnp.array_equal(out, fields)

    def test_jittable(self):
        coords = jnp.arange(8).reshape(1, 8).astype(jnp.float32)
        fields = jnp.ones((2, 1, 8))
        bundle = DataBundle(fields=fields, coords=coords)
        schema = IdentitySchema()

        assert_filter_jittable(schema.transform, bundle)
