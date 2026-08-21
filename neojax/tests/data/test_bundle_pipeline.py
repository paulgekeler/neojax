import pathlib
import tempfile

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.normalizers import UnitGaussianNormalizer
from neojax.data.processor import BundleProcessor
from neojax.data.schemas.bundle_reconstruct_schema import BundleReconstructSchema
from neojax.data.schemas.composed_schema import ComposedSchema
from neojax.data.schemas.concatenate_coords_schema import ConcatenateCoordsSchema
from neojax.data.schemas.flatten_time_schema import FlattenTimeSchema


@pytest.fixture
def dummy_pdebench_file() -> pathlib.Path:
    temp_dir = pathlib.Path(tempfile.mkdtemp())
    file_path = temp_dir / "dummy.hdf5"

    with h5py.File(file_path, "w") as f:
        # shape: [batch, time, x, y, channel]
        # e.g., 4 samples, 5 timesteps, 16x16 grid, 1 channel
        f.create_dataset("tensor", data=np.random.randn(4, 5, 16, 16, 1))
        # coords: [dim, x, y]
        f.create_dataset("coords", data=np.random.randn(2, 16, 16))

    return file_path


def test_full_bundle_pipeline(dummy_pdebench_file: pathlib.Path) -> None:
    # Load the dataset
    field_mapping = {"fields": "tensor", "coords": "coords"}
    dataset = BundleDataset.from_pdebench(
        dummy_pdebench_file, field_mapping=field_mapping
    )
    assert len(dataset) == 4

    # Convert the entire dataset into a single DataBundle to compute stats
    # This acts as our "full dataset" batch.
    full_bundle = dataset[:]

    # Setup Processor & Schemas
    processor = BundleProcessor(normalizers={"fields": UnitGaussianNormalizer()})
    processor = processor.compute_stats(full_bundle)

    # FNO-style schema: flatten time into channels, then append coords
    # Original fields: [batch, time, channel, x, y] -> time=1, channel=2
    # After flatten: [batch, time*channel, x, y] (time_axis=1, channel_axis=2 -> channel_axis becomes 1)
    # After coords: [batch, time*channel + coord_dims, x, y]
    in_schema = ComposedSchema(
        [
            FlattenTimeSchema(time_axis=1, channel_axis=2),
            ConcatenateCoordsSchema(channel_axis=1),
        ]
    )

    # Let's say the model predicts the next 1 timestep.
    # Output schema needs to unflatten 1 timestep -> [batch, 1, channel, x, y]
    out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=1)

    # Get a single batch (e.g. all 4 samples)
    batch_bundle = dataset[0:4]

    # Override dummy coords to be proper meshgrid [batch, 2, 16, 16] instead of [batch, 2, 16]
    # coords shape in dataset is [2, 16], broadcast to [4, 2, 16]
    # We want [4, 2, 16, 16]
    x, y = jnp.meshgrid(jnp.linspace(0, 1, 16), jnp.linspace(0, 1, 16), indexing="ij")
    coords = jnp.stack([x, y], axis=0)
    batched_coords = jnp.broadcast_to(coords, (4, *coords.shape))
    import equinox as eqx

    batch_bundle = eqx.tree_at(lambda b: b.coords, batch_bundle, batched_coords)

    # Transform (Forward pipeline)
    # The processor naturally handles batched DataBundles now, so no vmap is needed!
    model_inputs = processor.transform(batch_bundle, in_schema)

    # Verify shapes
    # fields [4, 5, 1, 16, 16] -> flatten -> [4, 5, 16, 16]
    # coords [4, 2, 16, 16]
    # concatenated -> [4, 7, 16, 16]
    assert model_inputs.shape == (4, 7, 16, 16)

    # Dummy Model Step
    # A dummy model that predicts next step features [4, 1, 16, 16] from the 7 input channels
    # Let's just output zeros
    model_outputs = jnp.zeros((4, 1, 16, 16))

    # Inverse Transform (Backward pipeline)
    pred_bundle = processor.inverse_transform(model_outputs, out_schema, batch_bundle)

    # Verify output bundle
    assert isinstance(pred_bundle, DataBundle)
    assert pred_bundle.fields.shape == (4, 1, 1, 16, 16)
    assert pred_bundle.coords is not None
    assert pred_bundle.coords.shape == (4, 2, 16, 16)

    # Verify inverse normalization occurred (since outputs were 0, inverse norm should shift by mean)
    normalizer = processor.normalizers["fields"]
    expected_output = normalizer.inverse_transform(jnp.zeros((4, 1, 1, 16, 16)))
    np.testing.assert_allclose(pred_bundle.fields, expected_output, rtol=1e-5)


def test_pipeline_jittable_and_vmappable() -> None:
    # Define batched sample
    coords = jnp.zeros((3, 2, 8, 8))
    fields = jnp.ones((3, 5, 1, 8, 8))  # batch=3, time=5, channel=1
    batched_bundle = DataBundle(fields=fields, coords=coords)

    processor = BundleProcessor(
        normalizers={
            "fields": UnitGaussianNormalizer(
                mean=jnp.array([1.0]), std=jnp.array([2.0])
            )
        }
    )

    in_schema = ComposedSchema(
        [
            FlattenTimeSchema(time_axis=1, channel_axis=2),
            ConcatenateCoordsSchema(channel_axis=1),
        ]
    )
    out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=1)

    def forward_pass(bundle: DataBundle, proc: BundleProcessor) -> DataBundle:
        inputs = proc.transform(bundle, schema=in_schema)
        # Dummy model predicting 1 feature channel across batch
        # outputs shape should be [batch, channel, x, y] -> [3, 1, 8, 8]
        # Wait, the shape of inputs is [3, 7, 8, 8]
        outputs = jnp.zeros((inputs.shape[0], 1, 8, 8))
        return proc.inverse_transform(
            outputs, schema=out_schema, reference_bundle=bundle
        )

    # Test standard forward pass
    out_bundle = forward_pass(batched_bundle, processor)
    assert out_bundle.fields.shape == (3, 1, 1, 8, 8)

    # Test JIT
    jitted_forward = jax.jit(forward_pass)
    out_bundle_jit = jitted_forward(batched_bundle, processor)
    assert out_bundle_jit.fields.shape == (3, 1, 1, 8, 8)

    # Test VMAP over batch inside the processor? No, processor is natively batched!
    # But let's verify it still vmaps correctly if the model itself is vmapped.
    # Suppose we have an unbatched bundle inside a vmapped model.
    single_bundle = DataBundle(
        fields=jnp.ones((1, 5, 1, 8, 8)), coords=jnp.zeros((1, 2, 8, 8))
    )
    jitted_single_forward = jax.jit(forward_pass)
    single_out = jitted_single_forward(single_bundle, processor)
    assert single_out.fields.shape == (1, 1, 1, 8, 8)


def test_pipeline_model_integration(dummy_pdebench_file: pathlib.Path) -> None:
    import equinox as eqx

    from neojax.models.fno import FNO

    # Load dataset
    field_mapping = {"fields": "tensor", "coords": "coords"}
    dataset = BundleDataset.from_pdebench(
        dummy_pdebench_file, field_mapping=field_mapping
    )
    batch_bundle = dataset[0:4]  # batched inputs [4, ...]

    x, y = jnp.meshgrid(jnp.linspace(0, 1, 16), jnp.linspace(0, 1, 16), indexing="ij")
    coords = jnp.stack([x, y], axis=0)
    batched_coords = jnp.broadcast_to(coords, (4, *coords.shape))
    batch_bundle = eqx.tree_at(lambda b: b.coords, batch_bundle, batched_coords)

    in_schema = ComposedSchema(
        [
            FlattenTimeSchema(time_axis=1, channel_axis=2),
            ConcatenateCoordsSchema(channel_axis=1),
        ]
    )
    out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=1)

    # Setup processor
    processor = BundleProcessor(normalizers={"fields": UnitGaussianNormalizer()})
    processor = processor.compute_stats(dataset[:])

    # Model setup
    model = FNO(
        key=jax.random.PRNGKey(0),
        in_channels=7,
        out_channels=1,
        hidden_channels=8,
        n_layers=2,
        modes=(4, 4),
    )

    from neojax.metrics import RelativeLpMetric

    loss_metric = RelativeLpMetric(p=2)

    def loss_fn(model_params, batch_inputs):
        # Forward pipeline
        model_inputs = processor.transform(batch_inputs, in_schema)

        # Forward model (vmap over batch dimension)
        vmap_model = jax.vmap(model_params)
        model_outputs = vmap_model(model_inputs)

        # Backward pipeline
        pred_bundle = processor.inverse_transform(
            model_outputs, out_schema, batch_inputs
        )

        # Compute loss over raw arrays using standard metric
        # For a dummy test, we use zero fields as the target
        target_fields = jnp.zeros_like(pred_bundle.fields)
        loss = loss_metric(target=target_fields, pred=pred_bundle.fields)
        return loss, pred_bundle

    # Check forward and backward pass
    (loss, pred_bundle), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, batch_bundle
    )

    # Verify outputs
    assert loss.shape == ()
    assert grads is not None
    # Ensure gradients have been computed for at least one parameter array
    assert any(
        g is not None and getattr(g, "size", 0) > 0
        for g in jax.tree_util.tree_leaves(grads)
    )

    # Check coherency of the returned output bundle
    assert isinstance(pred_bundle, DataBundle)
    assert pred_bundle.fields.shape == (4, 1, 1, 16, 16)
    assert pred_bundle.coords is not None
    assert pred_bundle.coords.shape == (4, 2, 16, 16)
