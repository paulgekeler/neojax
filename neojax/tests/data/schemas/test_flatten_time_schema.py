import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.flatten_time_schema import FlattenTimeSchema
from neojax.tests.conftest import assert_filter_jittable


class TestFlattenTimeSchema:
    def test_bundle_input(self):
        # fields shape: [time=2, channel=3, spatial=4]
        fields = jnp.arange(24).reshape(2, 3, 4).astype(jnp.float32)
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = FlattenTimeSchema(time_axis=0, channel_axis=1)
        out = schema.transform(bundle)
        assert out.shape == (6, 4)
        assert jnp.array_equal(out, fields.reshape(6, 4))

    def test_array_input(self):
        fields = jnp.arange(24).reshape(2, 3, 4).astype(jnp.float32)
        schema = FlattenTimeSchema(time_axis=0, channel_axis=1)
        out = schema.transform(fields)
        assert jnp.array_equal(out, fields.reshape(6, 4))

    def test_default_axes(self):
        # default time_axis=1, channel_axis=2 assumes a leading batch dim
        fields = jnp.ones((5, 2, 3, 4))
        schema = FlattenTimeSchema()
        out = schema.transform(fields)
        assert out.shape == (5, 6, 4)

    def test_invalid_axis_order(self):
        with pytest.raises(ValueError):
            FlattenTimeSchema(time_axis=2, channel_axis=1)

    def test_jittable(self):
        fields = jnp.ones((2, 3, 4))
        schema = FlattenTimeSchema(time_axis=0, channel_axis=1)
        assert_filter_jittable(schema.transform, fields)
