import jax.numpy as jnp
import jax.random as jr

from neojax.models.tfno import TFNO
from neojax.tests.conftest import assert_filter_jittable


class TestTFNO:
    def test_dimensions(self):
        key = jr.key(0)

        # 1D
        model_1d = TFNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4,),
            ranks=(2, 2, 2),
        )
        assert model_1d(jnp.ones((1, 32))).shape == (1, 32)

        # 2D
        model_2d = TFNO(
            key=key,
            in_channels=2,
            out_channels=2,
            hidden_channels=16,
            n_layers=2,
            modes=(4, 4),
            ranks=(2, 2, 2, 2),
            share_factor_matrices=False,
            domain_padding=0.1,
        )
        assert model_2d(jnp.ones((2, 16, 16))).shape == (2, 16, 16)

        # 3D
        model_3d = TFNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4, 4, 4),
            ranks=2,
        )
        assert model_3d(jnp.ones((1, 8, 8, 8))).shape == (1, 8, 8, 8)

    def test_mlp_layers_incorporation(self):
        key = jr.key(0)
        n_lift = 3
        n_proj = 4
        model = TFNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=1,
            modes=(4,),
            ranks=2,
            n_lift_layers=n_lift,
            n_proj_layers=n_proj,
        )
        # PointwiseMLP weight tuples should have n_layers elements
        assert len(model.lifting.weights) == n_lift
        assert len(model.projection.weights) == n_proj

    def test_normalization_and_residuals(self):
        key = jr.key(0)
        in_c, out_c, hidden = 1, 1, 8

        # Test different normalization types
        for norm in ["layer", "instance", "group", None]:
            model = TFNO(
                key=key,
                in_channels=in_c,
                out_channels=out_c,
                hidden_channels=hidden,
                n_layers=2,
                modes=(4,),
                ranks=2,
                normalization=norm,
                norm_groups=2 if norm == "group" else 1,
            )
            x = jnp.ones((in_c, 16))
            assert model(x).shape == (out_c, 16)
            assert_filter_jittable(model, x)

        # Test residual connections toggle
        model_no_res = TFNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4,),
            ranks=2,
            use_fno_residual=False,
            use_channel_mlp=False,
        )
        assert model_no_res(jnp.ones((in_c, 16))).shape == (out_c, 16)

    def test_dimensional_extension(self):
        key = jr.key(0)
        # 4D
        model_4d = TFNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            n_layers=1,
            modes=(2, 2, 2, 2),
            ranks=(1, 1, 1, 1, 1, 1),
        )
        assert model_4d(jnp.ones((1, 4, 4, 4, 4))).shape == (1, 4, 4, 4, 4)
