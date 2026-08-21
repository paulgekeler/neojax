# Include the cloned rigno_repo in sys.path to resolve namedtuple type assertions

import jax.numpy as jnp
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.graph_tuple_schema import (
    GraphTupleInputSchema,
    GraphTupleOutputSchema,
)
from neojax.tests.conftest import assert_filter_jittable

rigno = pytest.importorskip("rigno")


class TestGraphTupleSchema:
    def test_graph_tuple_schemas(self):
        from rigno.models.operator import Inputs

        coords = jnp.arange(20).reshape(2, 10).astype(jnp.float32)
        # fields shape: [time=1, channel=1, nodes=10]
        fields = jnp.ones((1, 1, 10))
        bundle = DataBundle(fields=fields, coords=coords)

        in_schema = GraphTupleInputSchema()
        inputs_res = in_schema.transform(bundle)
        assert isinstance(inputs_res, dict)
        assert "inputs" in inputs_res

        inputs_val = inputs_res["inputs"]
        assert isinstance(inputs_val, Inputs)
        # GraphTuple Inputs structure shapes
        assert inputs_val.u.shape == (1, 1, 10, 1)  # [1, time, nodes, channel]
        assert inputs_val.x_inp.shape == (1, 1, 10, 2)
        assert inputs_val.x_out.shape == (1, 1, 10, 2)

        out_schema = GraphTupleOutputSchema()
        # Outputs shape: [1, time=1, nodes=10, channel=1]
        model_out = jnp.zeros((1, 1, 10, 1))
        out_bundle = out_schema.transform(model_out, reference_bundle=bundle)
        assert isinstance(out_bundle, DataBundle)
        assert out_bundle.fields.shape == (1, 1, 10)  # [time, channel, nodes]
        assert jnp.allclose(out_bundle.coords, coords)

        assert_filter_jittable(lambda b: in_schema.transform(b)["inputs"].u, bundle)
        assert_filter_jittable(
            lambda m: out_schema.transform(m, reference_bundle=bundle).fields, model_out
        )
