import jax.numpy as jnp

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.mesh_schema import MeshInputSchema
from neojax.tests.conftest import assert_filter_jittable


class TestMeshInputSchema:
    def test_mesh_input_schema(self):
        coords = jnp.arange(20).reshape(2, 10).astype(jnp.float32)
        # fields shape: [time=2, channel=3, nodes=10]
        fields = jnp.ones((2, 3, 10))
        bundle = DataBundle(fields=fields, coords=coords)

        # Set time_axis=0, channel_axis=1 to match fields shape
        schema = MeshInputSchema(time_axis=0, channel_axis=1)
        res = schema.transform(bundle)
        assert isinstance(res, dict)
        assert "u" in res
        assert "x_in" in res
        assert res["u"].shape == (6, 10)
        assert res["x_in"].shape == (10, 2)
        assert_filter_jittable(lambda b: schema.transform(b)["u"], bundle)
