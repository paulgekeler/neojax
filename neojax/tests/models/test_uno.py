import jax.numpy as jnp
import jax.random as jr

from neojax.models.uno import UNO
from neojax.tests.conftest import assert_filter_jittable


class TestUNO:
    def test_basic_2d(self):
        """Basic 2D UNO with default horizontal skip map."""
        key = jr.key(0)
        model = UNO(
            key=key,
            in_channels=3,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4], [4, 4], [4, 4], [4, 4]],
            uno_scalings=[[1, 1], [1, 1], [1, 1], [1, 1]],
            n_fno_layers=4,
        )
        x = jnp.ones((3, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_basic_1d(self):
        """Basic 1D UNO."""
        key = jr.key(0)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4], [4], [4], [4]],
            uno_scalings=[[1], [1], [1], [1]],
            n_fno_layers=4,
        )
        x = jnp.ones((1, 32))
        out = model(x)
        assert out.shape == (1, 32)

    def test_scaling(self):
        """UNO with resolution scaling (down then up)."""
        key = jr.key(1)
        model = UNO(
            key=key,
            in_channels=2,
            out_channels=2,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4], [4, 4], [4, 4], [4, 4]],
            uno_scalings=[[0.5, 0.5], [0.5, 0.5], [2, 2], [2, 2]],
            n_fno_layers=4,
        )
        x = jnp.ones((2, 16, 16))
        out = model(x)
        # cumulative scaling: 0.5*0.5*2*2 = 1.0
        assert out.shape == (2, 16, 16)

    def test_custom_skip_map(self):
        """UNO with custom horizontal skip map."""
        key = jr.key(2)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 32, 16, 8],
            uno_modes=[[4, 4]] * 5,
            uno_scalings=[[1, 1]] * 5,
            n_fno_layers=5,
            horizontal_skip_map={4: 0, 3: 1},
        )
        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_no_horizontal_skip(self):
        """UNO with empty horizontal skip map (no skip connections)."""
        key = jr.key(3)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
            horizontal_skip_map={},
        )
        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_default_skip_map_construction(self):
        """Verify default horizontal skip map mirrors encoder to decoder."""
        key = jr.key(4)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
        )
        # For 4 layers: skip map should be {3: 0, 2: 1}
        assert model.horizontal_skip_map == {3: 0, 2: 1}

    def test_odd_layers_default_skip_map(self):
        """Default skip map with odd number of layers."""
        key = jr.key(5)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 32, 16, 8],
            uno_modes=[[4, 4]] * 5,
            uno_scalings=[[1, 1]] * 5,
            n_fno_layers=5,
        )
        # For 5 layers: skip map should be {4: 0, 3: 1}
        assert model.horizontal_skip_map == {4: 0, 3: 1}

    def test_no_channel_mlp(self):
        """UNO without channel MLPs."""
        key = jr.key(6)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
            use_channel_mlp=False,
        )
        assert model.channel_mlps is None
        assert model.channel_mlp_residuals is None

        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_with_domain_padding(self):
        """UNO with domain padding."""
        key = jr.key(7)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
            domain_padding=0.1,
        )
        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_with_positional_embedding(self):
        """UNO with grid positional embedding."""
        key = jr.key(8)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
            positional_embedding="grid",
        )
        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_normalization_variants(self):
        """Test all normalization types."""
        key = jr.key(9)
        for norm in ["layer", "instance", "group", None]:
            model = UNO(
                key=key,
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                uno_out_channels=[8, 16, 16, 8],
                uno_modes=[[4, 4]] * 4,
                uno_scalings=[[1, 1]] * 4,
                n_fno_layers=4,
                normalization=norm,
                norm_groups=2 if norm == "group" else 1,
            )
            x = jnp.ones((1, 16, 16))
            assert model(x).shape == (1, 16, 16)

    def test_horizontal_residual_variants(self):
        """Test horizontal skip projection types."""
        key = jr.key(10)
        for hr in ["linear", "soft-gating", "identity", None]:
            model = UNO(
                key=key,
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                uno_out_channels=[8, 16, 16, 8],
                uno_modes=[[4, 4]] * 4,
                uno_scalings=[[1, 1]] * 4,
                n_fno_layers=4,
                horizontal_residual=hr,
            )
            x = jnp.ones((1, 16, 16))
            assert model(x).shape == (1, 16, 16)

    def test_dropout(self):
        """UNO with channel MLP dropout."""
        key = jr.key(11)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
            channel_mlp_dropout=0.2,
        )
        x = jnp.ones((1, 16, 16))
        # Inference mode: deterministic
        out_inf1 = model(x, inference=True)
        out_inf2 = model(x, key=jr.key(1), inference=True)
        assert jnp.allclose(out_inf1, out_inf2)

        # Training mode with dropout
        out_drop = model(x, key=jr.key(2), inference=False)
        assert out_drop.shape == (1, 16, 16)

    def test_jittable(self):
        """Verify UNO is jittable."""
        key = jr.key(12)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
        )
        x = jnp.ones((1, 16, 16))
        assert_filter_jittable(model, x)

    def test_end_to_end_scaling_factor(self):
        """Verify cumulative scaling factor computation."""
        key = jr.key(13)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[0.5, 0.5], [0.5, 0.5], [2, 2], [2, 2]],
            n_fno_layers=4,
        )
        assert model.end_to_end_scaling_factor == [1.0, 1.0]

    def test_scalar_scalings(self):
        """UNO with scalar (non-nested) scalings."""
        key = jr.key(14)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[1, 1, 1, 1],
            n_fno_layers=4,
        )
        x = jnp.ones((1, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)

    def test_size_method(self):
        """Test model size reporting."""
        key = jr.key(15)
        model = UNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            uno_out_channels=[8, 16, 16, 8],
            uno_modes=[[4, 4]] * 4,
            uno_scalings=[[1, 1]] * 4,
            n_fno_layers=4,
        )
        size, n_params = model.size(return_n_params=True)
        assert size > 0
        assert n_params > 0
