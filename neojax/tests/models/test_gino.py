import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.models.gino import GiNO
from neojax.nn.neighbor_search import radius_neighbor_search
from neojax.tests.conftest import assert_filter_jittable

IN_C, OUT_C, N, M, GRID = 2, 3, 40, 25, 6


@pytest.fixture
def make_model():
    def f(coord_dim=2, **kwargs):
        defaults = dict(
            key=jr.key(0),
            in_channels=IN_C,
            out_channels=OUT_C,
            modes=(3,) * coord_dim,
            hidden_channels=8,
            n_layers=2,
            coord_dim=coord_dim,
            in_gno_radius=0.4,
            out_gno_radius=0.4,
            in_gno_max_neighbors=8,
            out_gno_max_neighbors=8,
            gno_embed_channels=4,
            in_gno_channel_mlp_hidden_layers=(8,),
            out_gno_channel_mlp_hidden_layers=(8,),
        )
        defaults.update(kwargs)
        return GiNO(**defaults)

    return f


@pytest.fixture
def make_inputs():
    def f(coord_dim=2, latent_feature_channels=None, seed=0):
        k1, k2, k3, k4, k5 = jr.split(jr.key(seed), 5)
        axes = [jnp.linspace(0.0, 1.0, GRID)] * coord_dim
        return {
            "u": jr.normal(k1, (IN_C, N)),
            "x_in": jr.uniform(k2, (N, coord_dim)),
            "x_out": jr.uniform(k3, (M, coord_dim)),
            "x_latent": jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1),
            "latent_features": (
                None
                if latent_feature_channels is None
                else jr.normal(k4, (latent_feature_channels, *([GRID] * coord_dim)))
            ),
        }

    return f


@pytest.mark.usefixtures("make_model", "make_inputs")
class TestGiNO:
    @pytest.mark.parametrize("coord_dim", [2, 3])
    def test_shapes_and_default_x_out(self, coord_dim, make_model, make_inputs):
        model = make_model(coord_dim)
        inputs = make_inputs(coord_dim)
        out = model(**inputs)
        assert out.shape == (OUT_C, M)
        assert bool(jnp.all(jnp.isfinite(out)))

        inputs.pop("x_out")
        assert model(**inputs).shape == (OUT_C, N)
        assert_filter_jittable(model, **make_inputs(coord_dim))

    @pytest.mark.parametrize(
        "in_type, out_type",
        [
            ("linear", "linear"),
            ("nonlinear", "nonlinear"),
            ("nonlinear_kernelonly", "nonlinear_kernelonly"),
            ("linear_kernelonly", "linear_kernelonly"),
        ],
    )
    @pytest.mark.parametrize("pos_embedding_type", ["transformer", None])
    def test_transform_types(
        self, in_type, out_type, pos_embedding_type, make_model, make_inputs
    ):
        model = make_model(
            in_gno_transform_type=in_type,
            out_gno_transform_type=out_type,
            in_gno_pos_embedding_type=pos_embedding_type,
            out_gno_pos_embedding_type=pos_embedding_type,
        )
        inputs = make_inputs()
        if in_type == "linear_kernelonly":
            inputs["u"] = None
        assert model(**inputs).shape == (OUT_C, M)

    def test_in_gno_out_channels_for_kernelonly(self, make_model, make_inputs):
        model = make_model(
            in_gno_transform_type="nonlinear_kernelonly", in_gno_out_channels=5
        )
        assert model.lifting.weights[0].shape[1] == 5
        assert model(**make_inputs()).shape == (OUT_C, M)

    def test_latent_features(self, make_model, make_inputs):
        model = make_model(latent_feature_channels=2)
        assert model.lifting.weights[0].shape[1] == IN_C + 2
        inputs = make_inputs(latent_feature_channels=2)
        assert model(**inputs).shape == (OUT_C, M)

        with pytest.raises(ValueError):
            model(**make_inputs())
        with pytest.raises(ValueError):
            make_model()(**inputs)
        inputs["latent_features"] = inputs["latent_features"][:, :-1]
        with pytest.raises(ValueError):
            model(**inputs)

    def test_dict_output_queries(self, make_model, make_inputs):
        model = make_model()
        inputs = make_inputs()
        x_out = {"volume": inputs["x_out"], "surface": inputs["x_out"][:7]}
        out = model(**{**inputs, "x_out": x_out})
        assert set(out) == {"volume", "surface"}
        assert out["volume"].shape == (OUT_C, M)
        assert out["surface"].shape == (OUT_C, 7)
        np.testing.assert_allclose(out["volume"], model(**inputs), rtol=1e-5, atol=1e-6)

    def test_precomputed_neighbors(self, make_model, make_inputs):
        model = make_model()
        inputs = make_inputs()
        flat = inputs["x_latent"].reshape(-1, 2)
        in_nb = radius_neighbor_search(inputs["x_in"], flat, 0.4, 8)
        out_nb = radius_neighbor_search(flat, inputs["x_out"], 0.4, 8)
        np.testing.assert_allclose(
            model(**inputs),
            model(**inputs, in_neighbors=in_nb, out_neighbors=out_nb),
            rtol=1e-5,
            atol=1e-6,
        )
        out_dict = model(
            **{**inputs, "x_out": {"a": inputs["x_out"]}},
            out_neighbors={"a": out_nb},
        )
        assert out_dict["a"].shape == (OUT_C, M)

        with pytest.raises(ValueError):
            model(**inputs, out_neighbors={"a": out_nb})
        with pytest.raises(ValueError):
            model(**{**inputs, "x_out": {"a": inputs["x_out"]}}, out_neighbors=out_nb)

    def test_vmap_batch_shared_and_per_sample_geometry(self, make_model, make_inputs):
        model = make_model()
        inputs = make_inputs()
        u_batch = jr.normal(jr.key(9), (3, IN_C, N))

        def call(u, x_in, x_latent, x_out):
            return model(u, x_in=x_in, x_latent=x_latent, x_out=x_out)

        shared = jax.vmap(call, in_axes=(0, None, None, None))(
            u_batch, inputs["x_in"], inputs["x_latent"], inputs["x_out"]
        )
        assert shared.shape == (3, OUT_C, M)
        np.testing.assert_allclose(
            shared[1],
            call(u_batch[1], inputs["x_in"], inputs["x_latent"], inputs["x_out"]),
            rtol=1e-5,
            atol=1e-6,
        )

        x_in_batch = jr.uniform(jr.key(10), (3, N, 2))
        per_sample = jax.vmap(call, in_axes=(0, 0, None, None))(
            u_batch, x_in_batch, inputs["x_latent"], inputs["x_out"]
        )
        assert per_sample.shape == (3, OUT_C, M)

    def test_grads_reach_all_layers(self, make_model, make_inputs):
        import equinox as eqx

        model = make_model()
        inputs = make_inputs()

        def loss(m):
            return jnp.sum(m(**inputs) ** 2)

        grads = eqx.filter_grad(loss)(model)
        for name in ["in_gno", "lifting", "fno_blocks", "out_gno", "projection"]:
            leaves = jax.tree_util.tree_leaves(
                eqx.filter(getattr(grads, name), eqx.is_array)
            )
            assert leaves
            assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
            assert any(float(jnp.abs(leaf).sum()) > 0 for leaf in leaves), name

    def test_mollified_gno_and_tanh(self, make_model, make_inputs):
        model = make_model(
            gno_weighting_function="bump",
            gno_weight_function_scale=2.0,
            out_gno_tanh=True,
        )
        assert model.out_gno.integral_transform.weighting_fn is not None
        assert model.out_gno.integral_transform.reduction == "sum"
        out = model(**make_inputs())
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_quadrature_weights(self, make_model, make_inputs):
        model = make_model()
        inputs = make_inputs()
        in_w = jr.uniform(jr.key(20), (N,)) + 0.5
        lat_w = jnp.full((GRID, GRID), 0.1)
        out = model(
            **inputs, in_quadrature_weights=in_w, latent_quadrature_weights=lat_w
        )
        assert out.shape == (OUT_C, M)
        assert not np.allclose(out, model(**inputs))
        assert_filter_jittable(
            lambda m: m(
                **inputs, in_quadrature_weights=in_w, latent_quadrature_weights=lat_w
            ),
            model,
        )
        with pytest.raises(ValueError, match="latent_quadrature_weights"):
            model(**inputs, latent_quadrature_weights=jnp.ones((GRID,)))

    def test_mollified_output_is_differentiable_in_queries(
        self, make_model, make_inputs
    ):
        model = make_model(
            gno_weighting_function="half_cos", out_gno_max_neighbors=GRID * GRID
        )
        inputs = make_inputs()
        flat = inputs["x_latent"].reshape(-1, 2)
        out_nb = radius_neighbor_search(flat, inputs["x_out"], 0.4, GRID * GRID)

        live = jax.grad(lambda xo: model(**{**inputs, "x_out": xo}).sum())(
            inputs["x_out"]
        )
        cached = jax.grad(
            lambda xo: model(**{**inputs, "x_out": xo}, out_neighbors=out_nb).sum()
        )(inputs["x_out"])
        np.testing.assert_allclose(cached, live, rtol=1e-4, atol=1e-5)
        assert float(jnp.abs(live).max()) > 0

    def test_dropout_and_inference(self, make_model, make_inputs):
        model = make_model(channel_mlp_dropout=0.5, n_layers=1)
        inputs = make_inputs()
        det1 = model(**inputs, inference=True)
        det2 = model(**inputs, inference=True)
        np.testing.assert_array_equal(det1, det2)
        a = model(**inputs, key=jr.key(1), inference=False)
        b = model(**inputs, key=jr.key(2), inference=False)
        assert not np.allclose(a, b)

    def test_factorized_spectral_weights(self, make_model, make_inputs):
        model = make_model(factorization="tucker", ranks=2)
        assert model(**make_inputs()).shape == (OUT_C, M)

    def test_save_load_and_astype(self, tmp_path, make_inputs):
        hyperparams = dict(
            in_channels=IN_C,
            out_channels=OUT_C,
            modes=[3, 3],
            hidden_channels=8,
            n_layers=1,
            coord_dim=2,
            in_gno_radius=0.4,
            out_gno_radius=0.4,
            gno_embed_channels=4,
            in_gno_channel_mlp_hidden_layers=[8],
            out_gno_channel_mlp_hidden_layers=[8],
        )
        model = GiNO(key=jr.key(0), **hyperparams)
        path = tmp_path / "gino.eqx"
        model.save(file=path, hyperparams=hyperparams)
        loaded = GiNO.load(file=path, make_fn=lambda **hp: GiNO(key=jr.key(5), **hp))
        inputs = make_inputs()
        np.testing.assert_allclose(model(**inputs), loaded(**inputs), rtol=1e-6)

        assert model.astype(jnp.bfloat16).lifting.weights[0].dtype == jnp.bfloat16
        assert model.size() > 0

    def test_invalid_arguments(self, make_model):
        with pytest.raises(ValueError):
            make_model(modes=())
        with pytest.raises(ValueError):
            make_model(modes=(3, 0))
        with pytest.raises(ValueError):
            make_model(hidden_channels=0)
        with pytest.raises(ValueError):
            make_model(latent_feature_channels=0)
        # linear transforms multiply kernel and function elementwise
        with pytest.raises(ValueError):
            make_model(in_gno_out_channels=IN_C + 1)

    def test_latent_grid_dimension_mismatch_raises(self, make_model, make_inputs):
        model = make_model()
        inputs = make_inputs()
        inputs["x_latent"] = inputs["x_latent"].reshape(-1, 2)
        with pytest.raises(ValueError):
            model(**inputs)
