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
