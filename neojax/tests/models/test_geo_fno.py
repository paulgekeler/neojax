import jax.numpy as jnp
import jax.random as jr

from neojax.models.geo_fno import GeoFNO
from neojax.tests.conftest import assert_filter_jittable
from neojax.utils.torch_converters import load_torch_weights_into_geo_fno


class TestGeoFNO:
    def test_grid_fallback(self):
        key = jr.key(0)
        # Test standard FNO grid fallback mode (x_in is None)
        model = GeoFNO(
            key=key,
            in_channels=2,
            out_channels=1,
            hidden_channels=8,
            n_layers=3,
            modes=(4, 4),
            grid_resolution=(16, 16),
            use_coord_projection=True,
        )
        x = jnp.ones((2, 16, 16))
        out = model(x)
        assert out.shape == (1, 16, 16)
        assert_filter_jittable(model, x)

    def test_mesh_mode_shapes(self):
        key = jr.key(0)
        # Test irregular mesh mapping mode
        model = GeoFNO(
            key=key,
            in_channels=2,
            out_channels=3,
            hidden_channels=16,
            n_layers=4,
            modes=(4, 4),
            grid_resolution=(8, 8),
            use_coord_projection=True,
        )

        # 100 mesh nodes, 2D coordinates
        x_in = jr.uniform(key, (100, 2))
        u = jr.normal(jr.split(key)[0], (2, 100))

        # Query on 50 different coordinates
        x_out = jr.uniform(jr.split(key)[1], (50, 2))

        out = model(u, x_in=x_in, x_out=x_out)
        assert out.shape == (3, 50)

        # Run with default x_out = x_in
        out_self = model(u, x_in=x_in)
        assert out_self.shape == (3, 100)

    def test_code_conditioning(self):
        key = jr.key(0)
        model = GeoFNO(
            key=key,
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            n_layers=3,
            modes=(4, 4),
            grid_resolution=(8, 8),
            geomap_width=16,
            code_dim=10,
        )

        x_in = jr.uniform(key, (80, 2))
        u = jr.normal(key, (1, 80))
        code = jr.normal(key, (10,))

        out = model(u, x_in=x_in, code=code)
        assert out.shape == (1, 80)

    def test_weight_converter(self):
        key = jr.key(0)
        model = GeoFNO(
            key=key,
            in_channels=2,
            out_channels=1,
            hidden_channels=8,
            n_layers=3,
            modes=(4, 4),
            grid_resolution=(8, 8),
            use_coord_projection=True,
        )

        # Build a dummy state dict mimicking the PyTorch models
        # e.g., models have:
        # model_iphi.fc0.weight
        # fc0.weight, fc0.bias
        # conv0.weights1, conv0.weights2
        # etc.
        dummy_state_dict = {
            "model_iphi.fc0.weight": jnp.ones((8, 4)),
            "model_iphi.fc0.bias": jnp.ones((8,)),
            "model_iphi.fc_no_code.weight": jnp.ones((32, 24)),
            "model_iphi.fc_no_code.bias": jnp.ones((32,)),
            "model_iphi.fc1.weight": jnp.ones((32, 32)),
            "model_iphi.fc1.bias": jnp.ones((32,)),
            "model_iphi.fc2.weight": jnp.ones((32, 32)),
            "model_iphi.fc2.bias": jnp.ones((32,)),
            "model_iphi.fc3.weight": jnp.ones((2, 32)),
            "model_iphi.fc3.bias": jnp.ones((2,)),
            "fc0.weight": jnp.ones((8, 2)),
            "fc0.bias": jnp.ones((8,)),
            "conv0.weights1": jnp.ones((8, 8, 4, 4)),
            "conv0.weights2": jnp.ones((8, 8, 4, 4)),
            "conv1.weights1": jnp.ones((8, 8, 4, 4)),
            "conv1.weights2": jnp.ones((8, 8, 4, 4)),
            "w1.weight": jnp.ones((8, 8, 1, 1)),
            "w1.bias": jnp.ones((8,)),
            "conv4.weights1": jnp.ones((8, 8, 4, 4)),
            "conv4.weights2": jnp.ones((8, 8, 4, 4)),
            "b0.weight": jnp.ones((8, 2, 1, 1)),
            "b0.bias": jnp.ones((8,)),
            "b1.weight": jnp.ones((8, 2, 1, 1)),
            "b1.bias": jnp.ones((8,)),
            "b2.weight": jnp.ones((8, 2, 1)),
            "b2.bias": jnp.ones((8,)),
            "fc1.weight": jnp.ones((128, 8)),
            "fc1.bias": jnp.ones((128,)),
            "fc2.weight": jnp.ones((1, 128)),
            "fc2.bias": jnp.ones((1,)),
        }

        updated_model = load_torch_weights_into_geo_fno(model, dummy_state_dict)

        # Verify a leaf has been correctly updated
        assert jnp.allclose(updated_model.lifting.weights[0], 1.0)
        assert jnp.allclose(updated_model.geomap.lin0.weight, 1.0)
        assert jnp.allclose(updated_model.conv_in.conv.weights.weights[0], 1.0)
        assert jnp.allclose(updated_model.projection.weights[0], 1.0)
