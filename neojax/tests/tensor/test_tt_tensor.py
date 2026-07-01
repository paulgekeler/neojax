import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.tensor.tt_tensor import TTTensor


class TestTTTensor:
    def test_tt_tensor_standard(self):
        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        ranks = (2, 3, 2)
        tensor = TTTensor(key, in_c, out_c, modes, ranks, separable=False)

        assert len(tensor.lr_tensors) == 8  # 2 * 4
        assert tensor.lr_tensors[0].shape == (out_c, ranks[0])
        assert not tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (out_c, 4, 4)

        # Test to_dense
        dense = tensor.to_dense()
        assert dense.shape == (2, out_c, in_c, 4, 4)

        # Verify variance is within range
        init_std = (2.0 / (in_c * out_c)) ** 0.5
        target_var = init_std ** 2
        actual_var = jnp.var(dense.real)
        assert 0.05 * (target_var / 2.0) < actual_var < 20.0 * (target_var / 2.0)

        # Verify that contraction matches manually reconstructed weight
        lr = tensor.lr_tensors[0:4]
        reconstructed_weight = jnp.einsum(tensor.to_dense_einsum_str, *lr)
        assert reconstructed_weight.shape == (out_c, in_c, 4, 4)
        expected_out = jnp.einsum("o i x y, i x y -> o x y", reconstructed_weight, x_slice)
        assert jnp.allclose(out, expected_out)

        # Verify to_dense matches corner slices
        assert jnp.allclose(dense[0], reconstructed_weight)

    def test_tt_tensor_separable(self):
        key = jr.key(0)
        in_c = 3
        modes = (4, 4)
        ranks = (2, 3)
        tensor = TTTensor(key, in_c, in_c, modes, ranks, separable=True)

        assert len(tensor.lr_tensors) == 6  # 2 * 3
        assert tensor.lr_tensors[0].shape == (in_c, ranks[0])
        assert tensor.separable

        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)
        out = tensor(0, x_slice)
        assert out.shape == (in_c, 4, 4)

        # Test to_dense
        dense = tensor.to_dense()
        assert dense.shape == (2, in_c, 4, 4)

        # Verify variance is within range
        init_std = (2.0 / in_c) ** 0.5
        target_var = init_std ** 2
        actual_var = jnp.var(dense.real)
        assert 0.05 * (target_var / 2.0) < actual_var < 20.0 * (target_var / 2.0)

        # Verify that contraction matches manually reconstructed weight
        lr = tensor.lr_tensors[0:3]
        reconstructed_weight = jnp.einsum(tensor.to_dense_einsum_str, *lr)
        assert reconstructed_weight.shape == (in_c, 4, 4)
        expected_out = reconstructed_weight * x_slice
        assert jnp.allclose(out, expected_out)

        # Verify to_dense matches corner slices
        assert jnp.allclose(dense[0], reconstructed_weight)

    def test_tt_tensor_separable_invalid_channels(self):
        key = jr.key(0)
        with pytest.raises(ValueError):
            TTTensor(key, 3, 5, (4, 4), ranks=2, separable=True)

    def test_tt_tensor_invalid_ranks(self):
        key = jr.key(0)
        with pytest.raises(ValueError):
            # Expects 3 ranks for standard 2D, but passes 2 ranks
            TTTensor(key, 3, 5, (4, 4), ranks=(2, 2), separable=False)

        with pytest.raises(ValueError):
            # Expects 2 ranks for separable 2D, but passes 3 ranks
            TTTensor(key, 3, 3, (4, 4), ranks=(2, 2, 2), separable=True)

    def test_jittable(self):
        from neojax.tests.conftest import assert_filter_jittable
        key = jr.key(0)
        in_c, out_c = 3, 5
        modes = (4, 4)
        x_slice = jnp.ones((in_c, 4, 4), dtype=jnp.complex64)

        # Standard
        tensor_std = TTTensor(key, in_c, out_c, modes, ranks=(2, 3, 2), separable=False)
        assert_filter_jittable(tensor_std, 0, x_slice)

        # Separable
        tensor_sep = TTTensor(key, in_c, in_c, modes, ranks=(2, 3), separable=True)
        assert_filter_jittable(tensor_sep, 0, x_slice)

