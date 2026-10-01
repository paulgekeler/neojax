from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from neojax.nn.pointwise_mlp import PointwiseMLP


class TestPointwiseMLP:
    def test_dimensions(self):
        key = jr.key(0)
        in_c, hidden, out_c = 16, 32, 8
        mlp = PointwiseMLP(
            key=key, layers=(in_c, hidden, out_c), activations=jax.nn.gelu
        )

        # 1D
        assert mlp(jnp.ones((in_c, 2))).shape == (out_c, 2)
        # 2D
        assert mlp(jnp.ones((in_c, 4, 4))).shape == (out_c, 4, 4)
        # 3D
        assert mlp(jnp.ones((in_c, 4, 4, 4))).shape == (out_c, 4, 4, 4)
        # 4D
        assert mlp(jnp.ones((in_c, 2, 2, 2, 2))).shape == (out_c, 2, 2, 2, 2)

    @pytest.mark.parametrize("activations", [jax.nn.gelu, [jax.nn.gelu, jax.nn.relu]])
    def test_activations(self, activations):
        key = jr.key(0)
        in_c, hidden, out_c = 16, 32, 8
        # Single activation
        mlp_single = PointwiseMLP(
            key=key, layers=(in_c, hidden, out_c), activations=activations
        )
        assert mlp_single(jnp.ones((in_c, 2))).shape == (out_c, 2)
        if isinstance(activations, Callable):
            assert (
                mlp_single.activations[0] == jax.nn.gelu
                and mlp_single.activations[1] is None
            )
        else:
            assert (
                len(mlp_single.activations) == 2
                and mlp_single.activations[1] is not None
            )

    @pytest.mark.parametrize("shape", [(1,), (1, 5), (1, 4, 4), (8, 5), (32,), (3, 2)])
    def test_channel_mismatch_raises(self, shape):
        # Don't broadcast size-1 channel axes to `in_c`
        mlp = PointwiseMLP(key=jr.key(0), layers=(16, 32, 8))
        with pytest.raises(ValueError):
            mlp(jnp.ones(shape))

    def test_single_channel_input_works(self):
        # Matching size-1 input dimension is still valid
        mlp = PointwiseMLP(key=jr.key(0), layers=(1, 4, 2))
        assert mlp(jnp.ones((1,))).shape == (2,)
        assert mlp(jnp.ones((1, 5))).shape == (2, 5)
