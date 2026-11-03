import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.patching import patchify, unpatchify


class TestPatching:
    @pytest.mark.parametrize(
        "grid, patch_size",
        [((8,), 2), ((8, 6), 2), ((9, 6), 3), ((4, 4, 4), 2), ((6, 4, 2), 2)],
    )
    def test_roundtrip_and_shape(self, grid, patch_size):
        x = jr.normal(jr.key(0), (3, *grid))
        tokens = patchify(x, patch_size)
        ndim = len(grid)
        assert tokens.shape == (
            3 * patch_size**ndim,
            int(np.prod([g // patch_size for g in grid])),
        )
        np.testing.assert_array_equal(unpatchify(tokens, patch_size, grid), x)

    def test_2d_token_content_and_order(self):
        # channel 0 holds the flat index of every grid point
        grid = (4, 6)
        x = jnp.arange(24, dtype=jnp.float32).reshape(1, *grid)
        tokens = np.asarray(patchify(x, 2))  # (4, 6)
        for t in range(6):
            pi, pj = divmod(t, 3)  # first grid axis varies slowest
            expected = x[0, 2 * pi : 2 * pi + 2, 2 * pj : 2 * pj + 2]
            assert set(tokens[:, t]) == set(np.asarray(expected).ravel())
        # features are ordered (in-patch axis 1, in-patch axis 2)
        np.testing.assert_array_equal(tokens[:, 0], [0, 1, 6, 7])

    def test_patch_size_one_is_a_reshape(self):
        x = jr.normal(jr.key(0), (2, 3, 4))
        np.testing.assert_array_equal(patchify(x, 1), x.reshape(2, 12))

    def test_invalid_arguments_raise(self):
        x = jnp.ones((2, 6, 6))
        with pytest.raises(ValueError):
            patchify(x, 4)
        with pytest.raises(ValueError):
            patchify(x, 0)
        with pytest.raises(ValueError):
            patchify(jnp.ones((3,)), 1)
        tokens = patchify(x, 2)
        with pytest.raises(ValueError):
            unpatchify(tokens, 2, (6, 4))
        with pytest.raises(ValueError):
            unpatchify(tokens, 4, (6, 6))
        with pytest.raises(ValueError):
            unpatchify(tokens, 2, ())
