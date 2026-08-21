import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.bundle_reconstruct_schema import BundleReconstructSchema
from neojax.tests.conftest import assert_filter_jittable


class TestBundleReconstructSchema:
    def test_reconstruct_without_unflatten(self):
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((1, 1, 4)), coords=coords)

        model_output = jnp.arange(4).reshape(1, 1, 4).astype(jnp.float32)
        schema = BundleReconstructSchema()
        out = schema.transform(model_output, reference_bundle=reference)

        assert isinstance(out, DataBundle)
        assert jnp.array_equal(out.fields, model_output)
        assert jnp.array_equal(out.coords, coords)

    def test_reconstruct_with_unflatten_time_steps(self):
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((2, 3, 4)), coords=coords)

        # model_output shape: [time*channel=6, spatial=4]
        model_output = jnp.arange(24).reshape(6, 4).astype(jnp.float32)
        schema = BundleReconstructSchema(unflatten_time_steps=2, channel_axis=0)
        out = schema.transform(model_output, reference_bundle=reference)

        assert out.fields.shape == (2, 3, 4)
        assert jnp.array_equal(out.fields, model_output.reshape(2, 3, 4))

    def test_non_divisible_channel_dim_raises(self):
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((2, 3, 4)), coords=coords)

        model_output = jnp.arange(20).reshape(5, 4).astype(jnp.float32)
        schema = BundleReconstructSchema(unflatten_time_steps=2, channel_axis=0)
        with pytest.raises(ValueError):
            schema.transform(model_output, reference_bundle=reference)

    def test_missing_reference_bundle_raises(self):
        model_output = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        schema = BundleReconstructSchema()
        with pytest.raises(ValueError):
            schema.transform(model_output)

    def test_jittable(self):
        coords = jnp.arange(4).reshape(1, 4).astype(jnp.float32)
        reference = DataBundle(fields=jnp.ones((1, 1, 4)), coords=coords)
        model_output = jnp.ones((1, 1, 4))
        schema = BundleReconstructSchema()

        assert_filter_jittable(
            lambda m: schema.transform(m, reference_bundle=reference).fields,
            model_output,
        )
