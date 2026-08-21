import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from neojax.benchmark.evaluators.base_evaluator import save_results
from neojax.benchmark.evaluators.steady_state_eval import SteadyStateEvaluator
from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.normalizers import UnitGaussianNormalizer
from neojax.data.processor import BundleProcessor
from neojax.data.schemas.bundle_reconstruct_schema import BundleReconstructSchema
from neojax.data.schemas.flatten_time_schema import FlattenTimeSchema
from neojax.metrics import LpMetric, MSEMetric


class DummyModel(eqx.Module):
    out_channels: int = eqx.field(static=True)

    def __call__(self, x):
        return x[: self.out_channels]


def test_steady_state_evaluator():
    # Setup data: 4 samples, 1 time step, 1 channel, spatial size 8
    coords = jnp.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    fields = jnp.ones((4, 1, 1, 8)) * 2.0
    dataset = BundleDataset(coords=coords, fields=fields)

    in_schema = FlattenTimeSchema(time_axis=0, channel_axis=1)
    out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=0)

    processor = BundleProcessor(
        normalizers={
            "fields": UnitGaussianNormalizer(
                mean=jnp.array([1.0]), std=jnp.array([1.0])
            )
        }
    )

    metrics = {
        "l2": LpMetric(p=2.0),
        "mse": MSEMetric(),
    }

    evaluator = SteadyStateEvaluator(
        metrics=metrics,
        in_schema=in_schema,
        out_schema=out_schema,
        processor=processor,
    )

    model = DummyModel(out_channels=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.json"

        results = evaluator.evaluate_dataset(
            model=model,
            dataset=dataset,
            batch_size=2,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
        )

        assert "summary" in results
        assert "details" in results
        assert "l2_mean" in results["summary"]
        assert "mse_mean" in results["summary"]
        assert len(results["details"]["l2"]) == 4
        assert len(results["details"]["mse"]) == 4

        json_path = Path(tmpdir) / "results.json"
        save_results(results, json_path)
        assert json_path.exists()

        npz_path = Path(tmpdir) / "results.npz"
        save_results(results, npz_path)
        assert npz_path.exists()
