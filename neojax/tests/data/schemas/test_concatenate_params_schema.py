import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.concatenate_params_schema import ConcatenateParamsSchema
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def schema():
    return ConcatenateParamsSchema(channel_axis=2)


class TestConcatenateParamsSchema:
    def test_bundle_input(self, schema):
        # fields shape (batch=1, time=1, channel=1, x=4, y=4)
        key = jr.key(0)
        fields = jr.normal(key=key, shape=(1, 1, 1, 4, 4))
        # parameters shape (batch=1, x=4, y=4) with optional channel dim
        # -> (batch=1, channel=1, x=4, y=4)
        parameters_w_channel = jnp.ones((1, 1, 4, 4))
        parameters = parameters_w_channel.squeeze(axis=1)
        # Define necessary coords -> with batch dim to match batch axis type hint
        coords = jnp.arange(32).reshape(1, 2, 4, 4).astype(jnp.float32)
        bundle = DataBundle(fields=fields, parameters=parameters, coords=coords)
        bundle_w_channel = DataBundle(
            fields=fields, parameters=parameters_w_channel, coords=coords
        )

        out = schema.transform(bundle)
        out_w_channel = schema.transform(bundle_w_channel)
        # channels increase by 1 -> (1, 1, 2, 4, 4)
        assert out.shape == (1, 1, 2, 4, 4)
        assert jnp.array_equal(out, out_w_channel)
        assert jnp.array_equal(out[:, :, :1], fields)

    def test_array_input_with_reference_bundle(self, schema):
        # fields shape (batch=1, time=1, channel=2, x=4, y=4)
        fields = jnp.ones((1, 1, 2, 4, 4))
        # parameters shape (batch=1, x=4, y=4) with optional channel dim
        # -> (batch=1, channel=1, x=4, y=4)
        parameters_w_channel = jnp.ones((1, 2, 4, 4))
        parameters = jnp.ones((1, 4, 4))
        # Define necessary coords
        coords = jnp.ones((1, 2, 4, 4))
        bundle = DataBundle(fields=fields, parameters=parameters, coords=coords)
        bundle_w_channel = DataBundle(
            fields=fields, coords=coords, parameters=parameters_w_channel
        )

        out = schema.transform(fields, reference_bundle=bundle)
        out_w_channel = schema.transform(fields, reference_bundle=bundle_w_channel)
        assert out.shape == (1, 1, 3, 4, 4)
        assert out_w_channel.shape == (1, 1, 4, 4, 4)

    def test_multi_batch_fields_and_params(self, schema):
        # fields shape (batch=5, time=1, channel=1, x=4, y=4)
        fields = jnp.ones((5, 1, 1, 4, 4))
        parameters = jnp.ones((5, 4, 4))
        # Define necessary coords
        coords = jnp.ones((5, 1, 4, 4))
        bundle = DataBundle(fields=fields, parameters=parameters, coords=coords)
        out = schema.transform(bundle)
        assert out.shape == (5, 1, 2, 4, 4)

    def test_missing_reference_bundle_raises(self, schema):
        fields = jnp.ones((1, 1, 1, 4, 4))
        with pytest.raises(ValueError):
            schema.transform(fields)

    def test_jittable(self, schema):
        fields = jnp.ones((2, 1, 2, 4, 4))
        parameters = jnp.arange(32).reshape(2, 4, 4).astype(jnp.float32)
        coords = jnp.ones((1, 2, 4, 4))
        bundle = DataBundle(fields=fields, parameters=parameters, coords=coords)

        assert_filter_jittable(schema.transform, bundle)
