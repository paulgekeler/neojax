import jax.numpy as jnp

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.composed_schema import ComposedSchema
from neojax.data.schemas.concatenate_coords_schema import ConcatenateCoordsSchema
from neojax.data.schemas.flatten_time_schema import FlattenTimeSchema
from neojax.tests.conftest import assert_filter_jittable


class TestComposedSchema:
    def test_transform_applies_schemas_in_order(self):
        # fields shape: [time=2, channel=1, spatial=4]
        fields = jnp.arange(8).reshape(2, 1, 4).astype(jnp.float32)
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = ComposedSchema(
            [
                FlattenTimeSchema(time_axis=0, channel_axis=1),
                ConcatenateCoordsSchema(channel_axis=0),
            ]
        )
        out = schema.transform(bundle)

        # FlattenTimeSchema: [2, 1, 4] -> [2, 4]
        # ConcatenateCoordsSchema: [2, 4] + coords [1, 4] -> [3, 4]
        assert out.shape == (3, 4)

    def test_reference_bundle_propagates_through_chain(self):
        fields = jnp.arange(8).reshape(2, 1, 4).astype(jnp.float32)
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        flatten = FlattenTimeSchema(time_axis=0, channel_axis=1)
        concat = ConcatenateCoordsSchema(channel_axis=0)
        schema = ComposedSchema([flatten, concat])

        composed_out = schema.transform(bundle)
        manual_out = concat.transform(
            flatten.transform(bundle), reference_bundle=bundle
        )
        assert jnp.array_equal(composed_out, manual_out)

    def test_jittable(self):
        fields = jnp.ones((2, 1, 4))
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = ComposedSchema(
            [
                FlattenTimeSchema(time_axis=0, channel_axis=1),
                ConcatenateCoordsSchema(channel_axis=0),
            ]
        )
        assert_filter_jittable(schema.transform, bundle)
