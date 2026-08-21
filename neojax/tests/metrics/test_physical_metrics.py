import jax.numpy as jnp

from neojax.metrics import (
    BoundaryConsistencyMetric,
    ConservationMetric,
    ResidualMetric,
)
from neojax.tests.conftest import assert_jittable


class TestPhysicalMetrics:
    def test_boundary_consistency(self):
        # Shape: [batch, time, channels, spatial] = [2, 3, 2, 4]
        pred = jnp.arange(48, dtype=jnp.float32).reshape(2, 3, 2, 4)

        # bc_masks shape: [batch, c_bc, spatial] = [2, 1, 4]
        # let's mask index 0 and 3 of the spatial grid
        bc_masks = jnp.array([[[1, 0, 0, 1]], [[1, 0, 0, 1]]], dtype=jnp.int32)

        # bc_values shape: [batch, time, c_bc, spatial] = [2, 3, 1, 4]
        # let's make it match pred exactly at masked locations
        # pred channel 0 has values:
        # sample 0, time 0: [0, 1, 2, 3] -> masked is 0, 3
        # sample 0, time 1: [8, 9, 10, 11] -> masked is 8, 11
        # sample 0, time 2: [16, 17, 18, 19] -> masked is 16, 19
        # sample 1, time 0: [24, 25, 26, 27] -> masked is 24, 27
        # sample 1, time 1: [32, 33, 34, 35] -> masked is 32, 35
        # sample 1, time 2: [40, 41, 42, 43] -> masked is 40, 43
        bc_values = jnp.array(
            [
                [
                    [[0.0, 0.0, 0.0, 3.0]],
                    [[8.0, 0.0, 0.0, 11.0]],
                    [[16.0, 0.0, 0.0, 19.0]],
                ],
                [
                    [[24.0, 0.0, 0.0, 27.0]],
                    [[32.0, 0.0, 0.0, 35.0]],
                    [[40.0, 0.0, 0.0, 43.0]],
                ],
            ],
            dtype=jnp.float32,
        )

        metric = BoundaryConsistencyMetric()

        # Prediction matches bc_values at the mask, consistency error must be 0
        actual = metric(pred=pred, bc_masks=bc_masks, bc_values=bc_values)
        assert jnp.allclose(actual, 0.0)

        # sample 0, time 0, channel 0, pos 0 is 0.0, let's change bc_value to 2.0
        # For sample 0: masked elements = 6, squared error = 4. Error = sqrt(4/6) = sqrt(2/3)
        # For sample 1: masked elements = 6, squared error = 0. Error = 0.0
        # Batch mean = (sqrt(2/3) + 0) / 2 = sqrt(2/3) / 2
        bc_values_err = bc_values.at[0, 0, 0, 0].set(2.0)
        actual_err = metric(pred=pred, bc_masks=bc_masks, bc_values=bc_values_err)
        assert jnp.allclose(actual_err, jnp.sqrt(2.0 / 3.0) / 2.0)

        # Verify zero if None
        assert jnp.allclose(metric(pred=pred, bc_masks=None, bc_values=None), 0.0)

    def test_conservation_metric(self):
        # coords shape: [1, 4] (1D space with 4 grid points)
        coords = jnp.array([[0.0, 1.0, 2.0, 3.0]])

        # Conservation function: sum of fields
        def conservation_fn(f, x):
            return jnp.sum(f)

        metric_abs = ConservationMetric(
            conservation_fn=conservation_fn, mode="absolute"
        )
        metric_rel = ConservationMetric(
            conservation_fn=conservation_fn, mode="relative"
        )

        # Conserved trajectory (sum is 10.0 at all time steps)
        # shape: [batch, time, channels, spatial] = [1, 3, 1, 4]
        pred_conserved = jnp.array(
            [[[[1.0, 2.0, 3.0, 4.0]], [[2.5, 2.5, 2.5, 2.5]], [[4.0, 3.0, 2.0, 1.0]]]]
        )

        actual_abs = metric_abs(pred=pred_conserved, coords=coords)
        actual_rel = metric_rel(pred=pred_conserved, coords=coords)
        assert jnp.allclose(actual_abs, 0.0, atol=1e-6)
        assert jnp.allclose(actual_rel, 0.0, atol=1e-6)

        # Drifting trajectory
        # t0 sum = 10.0, t1 sum = 11.0, t2 sum = 8.0
        # absolute deviations: [1.0, 2.0]
        # mean squared deviation: (1^2 + 2^2)/2 = 2.5
        # expected absolute: sqrt(2.5)
        # expected relative: sqrt(( (1/10)^2 + (2/10)^2 )/2) = sqrt(0.025)
        pred_drift = jnp.array(
            [[[[1.0, 2.0, 3.0, 4.0]], [[2.5, 2.5, 3.0, 3.0]], [[2.0, 2.0, 2.0, 2.0]]]]
        )

        actual_drift_abs = metric_abs(pred=pred_drift, coords=coords)
        assert jnp.allclose(actual_drift_abs, jnp.sqrt(2.5))

        actual_drift_rel = metric_rel(pred=pred_drift, coords=coords)
        assert jnp.allclose(actual_drift_rel, jnp.sqrt(0.025))

    def test_residual_metric(self):
        coords = jnp.array([[0.0, 1.0, 2.0, 3.0]])

        # Residual function: computes temporal derivative
        def residual_fn(p, x):
            # p shape is [t, c, spatial]
            # Returns central difference in time
            return p[1:] - p[:-1]

        metric = ResidualMetric(residual_fn=residual_fn)

        # Steady trajectory -> residual is 0
        pred_steady = jnp.ones((1, 3, 1, 4))
        actual = metric(pred=pred_steady, coords=coords)
        assert jnp.allclose(actual, 0.0)

        # Non-steady trajectory
        # shape [1, 2, 1, 4] -> residual shape: [1, 1, 4]
        # p_0 is ones, p_1 is ones * 2.0 -> residual is ones * 1.0
        pred_unsteady = jnp.array([[[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]]]])
        actual_unsteady = metric(pred=pred_unsteady, coords=coords)
        assert jnp.allclose(actual_unsteady, 1.0)

    def test_jit_and_grad(self):
        coords = jnp.array([[0.0, 1.0]])
        pred = jnp.array(
            [[[1.0, 2.0], [3.0, 4.0]]]
        )  # [1, 2, 2] -> [batch, time, spatial]

        # For BoundaryConsistency
        bc_masks = jnp.array([[[1, 0]]], dtype=jnp.int32)
        bc_values = jnp.array([[[[1.0, 0.0]], [[3.0, 0.0]]]])  # [1, 2, 1, 2]

        metric_bc = BoundaryConsistencyMetric()
        assert_jittable(
            lambda p, m, v: metric_bc(pred=p, bc_masks=m, bc_values=v),
            pred,
            bc_masks,
            bc_values,
        )

        metric_cons = ConservationMetric(conservation_fn=lambda f, x: jnp.sum(f))
        assert_jittable(lambda p, c: metric_cons(pred=p, coords=c), pred, coords)

        metric_res = ResidualMetric(residual_fn=lambda p, x: p[1:] - p[:-1])
        assert_jittable(lambda p, c: metric_res(pred=p, coords=c), pred, coords)
