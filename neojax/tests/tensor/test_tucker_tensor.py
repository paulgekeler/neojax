import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.tensor.tucker_tensor import TuckerTensor


class TestTuckerTensor:
    def test_tucker_tensor_standard(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        ranks = (2, 2, 3, 3)
        tensor = TuckerTensor(key, in_c, out_c, modes, ranks, separable=False)

        assert len(tensor.core_tensors) == 2  # 2 ** (2 - 1)
        assert tensor.core_tensors[0].shape == ranks
        assert not tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (out_c, 4, 4)

        # Test to_dense
        dense = tensor.to_dense()
        assert dense.shape == (2, out_c, in_c, 4, 4)

        # Verify variance is within range
        init_std = (2.0 / (in_c * out_c)) ** 0.5
        target_var = init_std**2
        actual_var = jnp.var(dense.real)
        assert 0.05 * (target_var / 2.0) < actual_var < 20.0 * (target_var / 2.0)

    def test_tucker_tensor_separable(self):
        key = jr.key(0)
        in_c = 3
        modes = (4, 4)
        ranks = (2, 3, 3)  # channel rank + 2 spatial ranks
        tensor = TuckerTensor(key, in_c, in_c, modes, ranks, separable=True)

        assert len(tensor.core_tensors) == 2
        assert tensor.core_tensors[0].shape == ranks
        assert tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (in_c, 4, 4)

        # Test to_dense
        dense = tensor.to_dense()
        assert dense.shape == (2, in_c, 4, 4)

        # Verify variance is within range
        init_std = (2.0 / in_c) ** 0.5
        target_var = init_std**2
        actual_var = jnp.var(dense.real)
        assert 0.05 * (target_var / 2.0) < actual_var < 20.0 * (target_var / 2.0)

    def test_tucker_tensor_separable_invalid_channels(self):
        key = jr.key(0)
        with pytest.raises(ValueError):
            TuckerTensor(key, 3, 5, (4, 4), ranks=2, separable=True)

    def test_tucker_tensor_separable_invalid_ranks(self):
        key = jr.key(0)
        with pytest.raises(ValueError):
            # Expects 3 ranks for separable 2D, but passes 4 ranks
            TuckerTensor(key, 3, 3, (4, 4), ranks=(2, 2, 2, 2), separable=True)

    def test_jittable(self):
        from neojax.tests.conftest import assert_filter_jittable

        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)

        # Standard
        ranks_std = (2, 2, 3, 3)
        tensor_std = TuckerTensor(key, in_c, out_c, modes, ranks_std, separable=False)
        assert_filter_jittable(tensor_std, 0, x_slice)

        # Separable
        ranks_sep = (2, 3, 3)
        tensor_sep = TuckerTensor(key, in_c, in_c, modes, ranks_sep, separable=True)
        assert_filter_jittable(tensor_sep, 0, x_slice)
