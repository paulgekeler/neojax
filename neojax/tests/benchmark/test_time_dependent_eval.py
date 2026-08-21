import tempfile
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

from neojax.benchmark.evaluators.base_evaluator import save_results
from neojax.benchmark.evaluators.time_dependent_eval import TimeDependentEvaluator
from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.schemas.bundle_reconstruct_schema import BundleReconstructSchema
from neojax.data.schemas.flatten_time_schema import FlattenTimeSchema
from neojax.metrics import LpMetric, MSEMetric


class DummyTimeModel(eqx.Module):
    def __call__(self, x):
        # x shape: [history_steps, c, spatial]
        # output shape: [1, c, spatial] (persist last time step)
        return x[-1:]


def test_time_dependent_evaluator():
    # Setup data: 4 samples, 5 time steps, 1 channel, spatial size 8
    coords = jnp.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    fields = jnp.ones((4, 5, 1, 8)) * 2.0
    dataset = BundleDataset(coords=coords, fields=fields)

    in_schema = FlattenTimeSchema(time_axis=0, channel_axis=1)
    out_schema = BundleReconstructSchema(unflatten_time_steps=1, channel_axis=0)

    metrics = {
        "l2": LpMetric(p=2.0),
        "mse": MSEMetric(),
    }

    evaluator = TimeDependentEvaluator(
        metrics=metrics,
        in_schema=in_schema,
        out_schema=out_schema,
        history_steps=2,
        processor=None,
    )

    model = DummyTimeModel()

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
