import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.flatten_to_points_schema import FlattenToPointsSchema
from neojax.tests.conftest import assert_filter_jittable


class TestFlattenToPointsSchema:
    def test_bundle_input(self):
        # fields shape: [time=1, channel=2, nodes=5], coords shape: [d=2, nodes=5]
        fields = jnp.arange(10).reshape(1, 2, 5).astype(jnp.float32)
        coords = jnp.arange(10).reshape(2, 5).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = FlattenToPointsSchema()
        u, y = schema.transform(bundle)

        assert u.shape == (10,)
        assert jnp.array_equal(u, fields.reshape(-1))
        assert y.shape == (5, 2)
        assert jnp.array_equal(y, coords.T)

    def test_array_input_with_reference_bundle(self):
        fields = jnp.arange(10).reshape(1, 2, 5).astype(jnp.float32)
        coords = jnp.arange(10).reshape(2, 5).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)

        schema = FlattenToPointsSchema()
        u, y = schema.transform(fields, reference_bundle=bundle)
        assert u.shape == (10,)
        assert y.shape == (5, 2)

    def test_missing_reference_bundle_raises(self):
        fields = jnp.ones((1, 2, 5))
        schema = FlattenToPointsSchema()
        with pytest.raises(ValueError):
            schema.transform(fields)

    def test_jittable(self):
        fields = jnp.arange(10).reshape(1, 2, 5).astype(jnp.float32)
        coords = jnp.arange(10).reshape(2, 5).astype(jnp.float32)
        bundle = DataBundle(fields=fields, coords=coords)
        schema = FlattenToPointsSchema()

        assert_filter_jittable(schema.transform, bundle)
