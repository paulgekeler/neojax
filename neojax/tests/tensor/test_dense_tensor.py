import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.tensor.dense_tensor import DenseTensor


class TestDenseTensor:
    def test_dense_tensor_standard(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        tensor = DenseTensor(key, in_c, out_c, modes, separable=False)

        assert len(tensor.weights) == 2  # 2 ** (2 - 1)
        assert tensor.weights[0].shape == (out_c, in_c, 4, 4)
        assert not tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (out_c, 4, 4)

    def test_dense_tensor_separable(self):
        key = jr.key(0)
        in_c = 3
        modes = (4, 4)
        tensor = DenseTensor(key, in_c, in_c, modes, separable=True)

        assert len(tensor.weights) == 2
        assert tensor.weights[0].shape == (in_c, 4, 4)
        assert tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (in_c, 4, 4)
        # Elementwise check
        expected = tensor.weights[0] * x_slice
        assert jnp.allclose(out, expected)

    def test_dense_tensor_separable_invalid_channels(self):
        key = jr.key(0)
        with pytest.raises(ValueError):
            DenseTensor(key, 3, 5, (4, 4), separable=True)

    def test_dense_tensor_from_weights(self):
        key = jr.key(0)
        in_c = 3
        modes = (4, 4)
        tensor = DenseTensor(key, in_c, in_c, modes, separable=True)
        dense_weights = jnp.stack(tensor.weights, axis=0)

        new_tensor = DenseTensor.from_weights(dense_weights, separable=True)
        assert new_tensor.separable
        assert len(new_tensor.weights) == 2
        assert jnp.allclose(new_tensor.weights[0], tensor.weights[0])

    def test_jittable(self):
        from neojax.tests.conftest import assert_filter_jittable
        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)

        # Standard
        tensor_std = DenseTensor(key, in_c, out_c, modes, separable=False)
        assert_filter_jittable(tensor_std, 0, x_slice)

        # Separable
        tensor_sep = DenseTensor(key, in_c, in_c, modes, separable=True)
        assert_filter_jittable(tensor_sep, 0, x_slice)

