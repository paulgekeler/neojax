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
import jax.random as jr
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.concatenate_coords_schema import ConcatenateCoordsSchema
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def schema():
    return ConcatenateCoordsSchema(channel_axis=1)


class TestConcatenateCoordsSchema:
    def test_bundle_input(self, schema):
        key = jr.key(0)
        # fields shape: [time=1, channel=2, x=4, y=4], coords shape: [d=2, x=4, y=4]
        fields = jr.normal(key=key, shape=(1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        out = schema.transform(bundle)
        # channels grow from 2 to 2 (fields) + 2 (coords) = 4
        assert out.shape == (1, 4, 4, 4)
        assert jnp.array_equal(out[:, :2], fields)
        assert jnp.array_equal(out[:, 2:], coords[None])

    def test_array_input_with_reference_bundle(self, schema):
        fields = jnp.ones((1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        out = schema.transform(fields, reference_bundle=bundle)
        assert out.shape == (1, 4, 4, 4)

    def test_missing_reference_bundle_raises(self, schema):
        fields = jnp.ones((1, 2, 4, 4))
        with pytest.raises(ValueError):
            schema.transform(fields)

    def test_jittable(self, schema):
        fields = jnp.ones((1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        assert_filter_jittable(schema.transform, bundle)
