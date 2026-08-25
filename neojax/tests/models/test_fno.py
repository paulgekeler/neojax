import jax.numpy as jnp
import jax.random as jr

from neojax.models.fno import FNO
from neojax.tests.conftest import assert_filter_jittable


class TestFNO:
    def test_dimensions(self):
        key = jr.key(0)

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
        key = jr.key(0)
        n_lift = 3
        n_proj = 4
        model = FNO(
            key=key,
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

    def test_normalization_and_residuals(self):
        key = jr.key(0)
        in_c, out_c, hidden = 1, 1, 8

        # Test different normalization types
        for norm in ["layer", "instance", "group", None]:
            model = FNO(
                key=key,
                in_channels=in_c,
                out_channels=out_c,
                hidden_channels=hidden,
                n_layers=2,
                modes=(4,),
                normalization=norm,
                norm_groups=2 if norm == "group" else 1,
            )
            x = jnp.ones((in_c, 16))
            assert model(x).shape == (out_c, 16)
            assert_filter_jittable(model, x)

        # Test residual connections toggle
        model_no_res = FNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4,),
            use_fno_residual=False,
            use_channel_mlp=False,
        )
        assert model_no_res(jnp.ones((in_c, 16))).shape == (out_c, 16)

    def test_dimensional_extension(self):
        key = jr.key(0)
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

    def test_parameters_propagation(self):
        key = jr.key(0)
        in_c, out_c, hidden = 2, 2, 8
        model = FNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4, 4),
            enforce_hermitian_symmetry=False,
            fft_norm="ortho",
            is_complex_data=True,
        )

        for fno_layer in model.fno_blocks.fno_layers:
            assert fno_layer.spectral_conv.enforce_hermitian_symmetry is False
            assert fno_layer.spectral_conv.fft_norm == "ortho"
            assert fno_layer.spectral_conv.is_complex_data is True

        x = jr.normal(key, (in_c, 16, 16), dtype=jnp.complex64)
        assert model(x).shape == (out_c, 16, 16)

    def test_resolution_scaling(self):
        key = jr.key(0)
        in_c, out_c, hidden = 2, 2, 8

        # Test resolution scaling without domain padding
        # 2 layers: 16 -> 24 -> 36 spatial size
        model_no_pad = FNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4, 4),
            resolution_scaling_factor=1.5,
        )
        x = jnp.ones((in_c, 16, 16))
        assert model_no_pad(x).shape == (out_c, 36, 36)

        # Test resolution scaling with domain padding
        model_pad = FNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4, 4),
            domain_padding=0.1,
            resolution_scaling_factor=1.5,
        )
        assert model_pad(x).shape == (out_c, 36, 36)

    def test_dropout(self):
        key = jr.key(0)
        in_c, out_c, hidden = 2, 2, 8
        model = FNO(
            key=key,
            in_channels=in_c,
            out_channels=out_c,
            hidden_channels=hidden,
            n_layers=2,
            modes=(4, 4),
            channel_mlp_dropout=0.2,
        )

        for mlp in model.fno_blocks.channel_mlps:
            assert mlp.dropout == 0.2

        x = jnp.ones((in_c, 16, 16))
        out_inf1 = model(x, inference=True)
        out_inf2 = model(x, key=jr.key(1), inference=True)
        assert jnp.allclose(out_inf1, out_inf2)

        out_drop = model(x, key=jr.key(2), inference=False)
        assert out_drop.shape == (out_c, 16, 16)
