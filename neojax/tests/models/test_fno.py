import jax.numpy as jnp
import jax.random as jr

from neojax.models.fno import FNO


class TestFNO:
    def test_dimensions(self):
        key = jr.PRNGKey(0)

        # 1D
        model_1d = FNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4,),
        )
        assert model_1d(jnp.ones((1, 32))).shape == (1, 32)

        # 2D
        model_2d = FNO(
            key=key,
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            n_layers=2,
            modes=(4, 4),
            domain_padding=0.1,
        )
        assert model_2d(jnp.ones((2, 16, 16))).shape == (2, 16, 16)

        # 3D
        model_3d = FNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4, 4, 4),
        )
        assert model_3d(jnp.ones((1, 8, 8, 8))).shape == (1, 8, 8, 8)

    def test_mlp_layers_incorporation(self):
        key = jr.PRNGKey(0)
        n_lift = 3
        n_proj = 4
        model = FNO(
            key=jr.PRNGKey(0),
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4,),
            n_lift_layers=n_lift,
            n_proj_layers=n_proj,
        )
        # PointwiseMLP weight tuples should have n_layers elements
        assert len(model.lifting.weights) == n_lift
        assert len(model.projection.weights) == n_proj

        # 4D
        model_4d = FNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            n_layers=1,
            modes=(2, 2, 2, 2),
        )
        assert model_4d(jnp.ones((1, 4, 4, 4, 4))).shape == (1, 4, 4, 4, 4)
