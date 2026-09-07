import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas.time_to_stationary_schema import TimeToStationarySchema
from neojax.tests.conftest import assert_jittable


@pytest.fixture
def schema():
    return TimeToStationarySchema(time_axis=1)


@pytest.fixture
def schema_w_indices():
    return TimeToStationarySchema(time_axis=1, time_slice_indices=(0, 5))


@pytest.fixture
def attrs_w_time_axis():
    return ["fields", "bc_values", "parameters"]


@pytest.fixture
def bundle_all_fields():
    # 5 time steps
    fkey, ckey, pkey, bcv_key, bcm_key = jr.split(jr.key(0), 5)
    b = 1
    t = 10
    c_bc = 1
    c = 1
    d1 = 4
    fields = jr.normal(fkey, (b, t, c, d1))
    coords = jr.normal(ckey, (b, 1, d1))
    parameters = jr.normal(pkey, (b, t, c, d1))
    bc_values = jr.normal(bcv_key, (b, t, c_bc, d1))
    bc_mask = jr.choice(bcm_key, jnp.array([0, 1], dtype=jnp.int32), (b, c_bc, d1))
    edge_inds = jnp.arange(20, dtype=jnp.int32).reshape((b, 2, 10))
    return DataBundle(
        fields=fields,
        coords=coords,
        parameters=parameters,
        bc_values=bc_values,
        bc_masks=bc_mask,
        edge_indices=edge_inds,
    )


@pytest.fixture
def bundle():
    # 5 time steps
    fields = jnp.ones((1, 5, 1, 4))
    coords = jnp.ones((1, 1, 4))
    return DataBundle(fields=fields, coords=coords)


class TestTimeToStationarySchema:
    def assert_time_slices_match(
        self, in_bundle, out_bundle, time_axis, time_slice_indices, attrs
    ):
        for attr_name in attrs:
            in_attr = getattr(in_bundle, attr_name)
            out_attr = getattr(out_bundle, attr_name)
            in_attr_sliced = jnp.take(
                in_attr, jnp.array(time_slice_indices), axis=time_axis
            )
            assert jnp.array_equal(in_attr_sliced, out_attr)

    def test_wrong_time_indices(self, bundle_all_fields, attrs_w_time_axis):
        schema_more_indices = TimeToStationarySchema(
            time_axis=1, time_slice_indices=(1, 2, 3)
        )
        schema_rev_ind_order = TimeToStationarySchema(
            time_axis=1, time_slice_indices=(5, 2)
        )
        schema_rev_neg_ind_order = TimeToStationarySchema(
            time_axis=1, time_slice_indices=(-1, 3)
        )
        schmema_out_of_range = TimeToStationarySchema(
            time_axis=1, time_slice_indices=(0, 11)
        )

        with pytest.raises(ValueError):
            _ = schema_more_indices.transform(bundle_all_fields)

        for s in [schema_rev_ind_order, schema_rev_neg_ind_order]:
            out = s.transform(bundle_all_fields)
            self.assert_time_slices_match(
                bundle_all_fields,
                out,
                s.time_axis,
                s.time_slice_indices,
                attrs_w_time_axis,
            )
        # Out of range fills with NaNs
        out = schmema_out_of_range.transform(bundle_all_fields)
        for attr_name in attrs_w_time_axis:
            in_attr = getattr(bundle_all_fields, attr_name)
            out_attr = getattr(out, attr_name)
            nan_array = jnp.full_like(in_attr[:, 0], fill_value=jnp.nan)
            assert jnp.array_equal(
                out_attr,
                jnp.stack(
                    [in_attr[:, 0], nan_array], axis=schmema_out_of_range.time_axis
                ),
                equal_nan=True,
            )

    def test_missing_reference_bundle_raises(self, schema):
        fields = jnp.ones((1, 1, 1, 4))
        with pytest.raises(ValueError):
            _ = schema.transform(fields)

    def test_neg_time_slice_indices(self, bundle_all_fields, attrs_w_time_axis):
        schema = TimeToStationarySchema(time_axis=1, time_slice_indices=(-8, -3))
        schema2 = TimeToStationarySchema(time_axis=1, time_slice_indices=(0, -3))
        out = schema.transform(bundle_all_fields)
        out2 = schema2.transform(bundle_all_fields)
        self.assert_time_slices_match(
            bundle_all_fields,
            out,
            schema.time_axis,
            schema.time_slice_indices,
            attrs_w_time_axis,
        )
        self.assert_time_slices_match(
            bundle_all_fields,
            out2,
            schema2.time_axis,
            schema2.time_slice_indices,
            attrs_w_time_axis,
        )

    def test_bundle_input(self, bundle_all_fields, schema_w_indices, attrs_w_time_axis):
        out = schema_w_indices.transform(bundle_all_fields)
        self.assert_time_slices_match(
            bundle_all_fields,
            out,
            schema_w_indices.time_axis,
            schema_w_indices.time_slice_indices,
            attrs_w_time_axis,
        )

    def test_array_input(self, bundle_all_fields, schema_w_indices, attrs_w_time_axis):
        fields = bundle_all_fields.fields
        out = schema_w_indices.transform(fields, reference_bundle=bundle_all_fields)
        self.assert_time_slices_match(
            bundle_all_fields,
            out,
            schema_w_indices.time_axis,
            schema_w_indices.time_slice_indices,
            attrs_w_time_axis,
        )

    def test_jittable(self, bundle_all_fields, schema_w_indices):
        assert_jittable(schema_w_indices.transform, bundle_all_fields)
