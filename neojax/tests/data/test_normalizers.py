import jax.numpy as jnp
import pytest

from neojax.data.normalizers import (
    ComposedNormalizer,
    MinMaxNormalizer,
    PhysicsNormalizer,
    RobustNormalizer,
    UnitGaussianNormalizer,
)
from neojax.data.scales import CharacteristicLengthScale


class TestUnitGaussianNormalizer:
    def test_global_stats(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = UnitGaussianNormalizer()
        norm = norm.compute_stats(data)

        assert jnp.allclose(norm.mean, 3.0)
        assert jnp.allclose(norm.std, jnp.std(data))

        out = norm(data)
        assert jnp.allclose(jnp.mean(out), 0.0, atol=1e-6)
        assert jnp.allclose(jnp.std(out), 1.0, atol=1e-6)
        assert jnp.allclose(norm.inverse_transform(out), data)

    def test_axis_stats(self):
        # Shape (2, 3) - compute stats over axis 1 (per-channel)
        data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        norm = UnitGaussianNormalizer().compute_stats(data, axis=1)

        # Expected means: [2.0, 5.0] but kept with keepdims=True -> [[2.0], [5.0]]
        assert norm.mean.shape == (2, 1)
        assert jnp.allclose(norm.mean, jnp.array([[2.0], [5.0]]))


class TestMinMaxNormalizer:
    def test_scaling(self):
        data = jnp.array([-1.0, 0.0, 1.0])
        norm = MinMaxNormalizer().compute_stats(data)

        out = norm.transform(data, mode="scale")
        assert jnp.all(out >= 0.0) and jnp.all(out <= 1.0)
        assert jnp.allclose(out, jnp.array([0.0, 0.5, 1.0]))
        assert jnp.allclose(norm.inverse_transform(out), data)

    def test_clipping(self):
        data = jnp.array([0.0, 0.5, 1.0])
        norm = MinMaxNormalizer(minima=0.2, maxima=0.8)

        with pytest.warns(UserWarning):
            out = norm.transform(data, mode="clip")
            assert jnp.allclose(out, jnp.array([0.2, 0.5, 0.8]))

    def test_axis_stats(self):
        data = jnp.ones((4, 10, 10))  # (C, H, W)
        norm = MinMaxNormalizer().compute_stats(data, axis=(1, 2))
        assert norm.minima.shape == (4, 1, 1)


class TestRobustNormalizer:
    def test_robust_scaling(self):
        # Data with outlier
        data = jnp.array([1.0, 1.1, 1.2, 100.0])
        norm = RobustNormalizer().compute_stats(data)

        # Median of [1.0, 1.1, 1.2, 100.0] is between 1.1 and 1.2 -> 1.15
        assert jnp.allclose(norm.median, 1.15)

        out = norm(data)
        assert jnp.allclose(norm.inverse_transform(out), data)


class TestPhysicsNormalizer:
    def test_multiple_scales(self):
        data = jnp.ones((10,)) * 10.0
        s1 = CharacteristicLengthScale(L_ref=2.0)
        s2 = CharacteristicLengthScale(L_ref=5.0)

        norm = PhysicsNormalizer(s1, s2).compute_stats(data)
        # Total scale = 2.0 * 5.0 = 10.0
        # Result = 10.0 / 10.0 = 1.0

        assert jnp.allclose(norm(data), 1.0)
        assert jnp.allclose(norm.inverse_transform(norm(data)), data)


class TestComposedNormalizer:
    def test_sequential_vs_parallel(self):
        data = jnp.array([10.0, 20.0, 30.0])

        # Sequential (Default):
        # 1. UnitGaussian: mean=20, std=~8.16 -> Out is approx [-1.2, 0, 1.2]
        # 2. MinMax: Should scale [-1.2, 1.2] to [0, 1]
        norm_seq = ComposedNormalizer(
            UnitGaussianNormalizer(), MinMaxNormalizer()
        ).compute_stats(data, sequential=True)

        out_seq = norm_seq(data)
        assert jnp.allclose(jnp.min(out_seq), 0.0, atol=1e-5)
        assert jnp.allclose(jnp.max(out_seq), 1.0, atol=1e-5)

        # Parallel (Non-sequential):
        # 1. UnitGaussian: mean=20, std=~8.16
        # 2. MinMax: min=10, max=30
        # If we apply this non-sequentially, the MinMax won't produce [0, 1]
        # because it's using the range of the ORIGINAL data to scale the
        # ALREADY Gaussian-normalized data.
        norm_par = ComposedNormalizer(
            UnitGaussianNormalizer(), MinMaxNormalizer()
        ).compute_stats(data, sequential=False)

        out_par = norm_par(data)
        # Min of Gaussian out was approx -1.2.
        # MinMax scale step: (-1.2 - 10) / (30 - 10) = -11.2 / 20 = -0.56
        assert jnp.min(out_par) < 0.0
