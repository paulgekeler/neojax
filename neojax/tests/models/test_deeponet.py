import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from neojax.models.deeponet import DeepONet, MLPDeepONet
from neojax.tests.conftest import assert_filter_jittable


class TestDeepONet:
    def test_basic_forward(self):
        # Test the flexible base class with manual nets
        key = jr.key(0)
        m, d, p = 10, 2, 5

        # Simple linear layers (weights only)
        branch = eqx.nn.Linear(m, p, key=key)
        trunk = eqx.nn.Linear(d, p, key=key)

        model = DeepONet(branch, trunk)

        u = jnp.zeros((m,))
        y = jnp.zeros((d,))
        out = model(u, y)

        assert out.shape == ()

    def test_bias_and_activation(self):
        key = jr.key(0)
        m, d, p = 10, 2, 5

        branch = eqx.nn.Linear(m, p, key=key)
        trunk = eqx.nn.Linear(d, p, key=key)
        bias = jnp.array([1.0])

        # With bias
        model = DeepONet(branch, trunk, bias=bias)

        u = jnp.zeros((m,))
        y = jnp.zeros((d,))
        out = model(u, y)

        assert out.shape == ()

        # With activation
        model_act = DeepONet(branch, trunk, out_activation=jax.nn.relu)

        out = model_act(u, y)
        assert out.shape == ()


class TestMLPDeepONet:
    def test_mlp_forward(self):
        key = jr.key(0)
        m, d, p = 20, 1, 10
        model = MLPDeepONet(
            key=key,
            m_sensors=m,
            d_dim=d,
            p_latent=p,
            branch_hidden_dims=[32, 32],
            trunk_hidden_dims=[16],
        )

        u = jnp.ones((m,))
        y = jnp.ones((d,))
        out = model(u, y)
        assert out.shape == ()

    def test_vmap_evaluation(self):
        key = jr.key(0)
        m, d, p = 20, 1, 10
        model = MLPDeepONet(key, m, d, p, [16], [16])

        # Evaluate 1 function over 50 points
        u = jnp.ones((m,))
        y_batch = jnp.ones((50, d))

        vmap_model = jax.vmap(model, in_axes=(None, 0))
        preds = vmap_model(u, y_batch)

        assert preds.shape == (50,)

    def test_jit_compatibility(self):
        key = jr.key(0)
        model = MLPDeepONet(key, 10, 1, 5, [8], [8])
        u, y = jnp.ones((10,)), jnp.ones((1,))

        assert_filter_jittable(model, u, y)
