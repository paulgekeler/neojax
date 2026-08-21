import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.reshape_points_to_grid_schema import ReshapePointsToGridSchema
from neojax.tests.conftest import assert_filter_jittable


class TestReshapePointsToGridSchema:
    def test_bundle_input_passthrough(self):
        coords = jnp.arange(16).reshape(1, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=jnp.ones((1, 1, 4, 4)), coords=coords)

        schema = ReshapePointsToGridSchema()
        out = schema.transform(bundle)
        assert out is bundle

    def test_reshapes_flat_predictions_to_grid(self):
        # coords define a 4x4 spatial grid
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((1, 1, 4, 4)), coords=coords)

        outputs = jnp.arange(16).astype(jnp.float32)
        schema = ReshapePointsToGridSchema(time_steps=1, channels=1)
        out = schema.transform(outputs, reference_bundle=reference)

        assert isinstance(out, DataBundle)
        assert out.fields.shape == (1, 1, 4, 4)
        assert jnp.array_equal(out.fields, outputs.reshape(1, 1, 4, 4))
        assert jnp.array_equal(out.coords, coords)

    def test_reshapes_with_multiple_time_steps_and_channels(self):
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((1, 1, 4, 4)), coords=coords)

        outputs = jnp.arange(2 * 3 * 16).astype(jnp.float32)
        schema = ReshapePointsToGridSchema(time_steps=2, channels=3)
        out = schema.transform(outputs, reference_bundle=reference)

        assert out.fields.shape == (2, 3, 4, 4)

    def test_missing_reference_bundle_raises(self):
        outputs = jnp.arange(16).astype(jnp.float32)
        schema = ReshapePointsToGridSchema()
        with pytest.raises(ValueError):
            schema.transform(outputs)

    def test_jittable(self):
        coords = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((1, 1, 4, 4)), coords=coords)
        outputs = jnp.arange(16).astype(jnp.float32)
        schema = ReshapePointsToGridSchema(time_steps=1, channels=1)

        assert_filter_jittable(
            lambda o: schema.transform(o, reference_bundle=reference).fields, outputs
        )
