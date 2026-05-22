import jax
import jax.numpy as jnp
import jax.random as jr

from neojax.nn.pointwise_mlp import PointwiseMLP


class TestPointwiseMLP:
    def test_dimensions(self):
        key = jr.PRNGKey(0)
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

    def test_activations(self):
        key = jr.PRNGKey(0)
        in_c, hidden, out_c = 16, 32, 8
        # Single activation
        mlp_single = PointwiseMLP(
            key=key, layers=(in_c, hidden, out_c), activations=jax.nn.relu
        )
        assert mlp_single(jnp.ones((in_c, 2))).shape == (out_c, 2)
