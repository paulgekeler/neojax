import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.concatenate_coords_schema import ConcatenateCoordsSchema
from neojax.tests.conftest import assert_filter_jittable


class TestConcatenateCoordsSchema:
    def test_bundle_input(self):
        # fields shape: [time=1, channel=2, x=4, y=4], coords shape: [d=2, x=4, y=4]
        fields = jnp.ones((1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = ConcatenateCoordsSchema(channel_axis=1)
        out = schema.transform(bundle)
        # channels grow from 2 to 2 (fields) + 2 (coords) = 4
        assert out.shape == (1, 4, 4, 4)
        assert jnp.array_equal(out[:, :2], fields)
        assert jnp.array_equal(out[:, 2:], coords[None])

    def test_array_input_with_reference_bundle(self):
        fields = jnp.ones((1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = ConcatenateCoordsSchema(channel_axis=1)
        out = schema.transform(fields, reference_bundle=bundle)
        assert out.shape == (1, 4, 4, 4)

    def test_missing_reference_bundle_raises(self):
        fields = jnp.ones((1, 2, 4, 4))
        schema = ConcatenateCoordsSchema(channel_axis=1)
        with pytest.raises(ValueError):
            schema.transform(fields)

    def test_jittable(self):
        fields = jnp.ones((1, 2, 4, 4))
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)
        schema = ConcatenateCoordsSchema(channel_axis=1)

        assert_filter_jittable(schema.transform, bundle)
