import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from neojax.data.bundles.data_bundle import DataBundle
from neojax.data.schemas import (
    BundleReconstructSchema,
    ConcatenateCoordsSchema,
    MeshInputSchema,
)
from neojax.models.transolver import Transolver
from neojax.nn.positional_embedding import GridEmbeddingNd
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def small_params():
    return {"hidden_channels": 16, "n_layers": 2, "n_heads": 2, "slice_num": 4}


@pytest.fixture
def make_model(small_params):
    def f(**kwargs):
        kwargs = {
            "key": jr.key(0),
            "in_channels": 2,
            "out_channels": 3,
            "ndim": 2,
            **small_params,
            **kwargs,
        }
        return Transolver(**kwargs)

    return f


@pytest.mark.usefixtures("make_model")
class TestTransolver:
    def test_mesh_shapes(self, make_model):
        model = make_model(use_mesh=True)
        u = jr.normal(jr.key(1), (2, 50))
        x_in = jr.uniform(jr.key(2), (50, 2))
        out = model(u, x_in=x_in)
        assert out.shape == (3, 50)
        assert model.n_extra_channels == 2
        assert model.lifting.weights[0].shape[1] == 4
        assert_filter_jittable(lambda m, u, x: m(u, x_in=x), model, u, x_in)

    @pytest.mark.parametrize(
        "ndim,shape", [(1, (2, 20)), (2, (2, 8, 9)), (3, (2, 4, 4, 3))]
    )
    def test_grid_shapes(self, ndim, shape, make_model):
        model = make_model(ndim=ndim)
        out = model(jr.normal(jr.key(1), shape))
        assert out.shape == (3, *shape[1:])
        assert model.n_extra_channels == 0
        assert_filter_jittable(model, jnp.ones(shape))

    def test_grid_resolution_independent_weights(self, make_model):
        model = make_model()
        assert model(jnp.ones((2, 8, 8))).shape == (3, 8, 8)
        assert model(jnp.ones((2, 11, 5))).shape == (3, 11, 5)

    @pytest.mark.parametrize("pe", ["grid", GridEmbeddingNd(2, 2)])
    def test_grid_positional_embedding(self, pe, make_model):
        model = make_model(positional_embedding=pe)
        assert model.n_extra_channels == 2
        assert model.lifting.weights[0].shape[1] == 4
        assert model(jnp.ones((2, 6, 6))).shape == (3, 6, 6)

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_unified_pos(self, use_mesh, make_model):
        model = make_model(use_mesh=use_mesh, unified_pos=True, ref=3)
        assert model.n_extra_channels == 9
        assert model.lifting.weights[0].shape[1] == 2 + 9
        if use_mesh:
            out = model(jnp.ones((2, 30)), x_in=jr.uniform(jr.key(1), (30, 2)))
            assert out.shape == (3, 30)
        else:
            assert model(jnp.ones((2, 6, 7))).shape == (3, 6, 7)

    def test_mesh_equals_grid_on_flat_grid_with_linear_attention(self, make_model):
        # With use_mesh=True, a flattened regular grid and its coordinates give
        # the same result as feeding the points in any order (permutation equivariance).
        model = make_model(use_mesh=True)
        u = jr.normal(jr.key(1), (2, 40))
        x_in = jr.uniform(jr.key(2), (40, 2))
        perm = jr.permutation(jr.key(3), 40)
        out = model(u, x_in=x_in)
        out_perm = model(u[:, perm], x_in=x_in[perm])
        np.testing.assert_allclose(out[:, perm], out_perm, rtol=1e-4, atol=1e-5)

    def test_vmap_over_batch_with_shared_coords(self, make_model):
        model = make_model(use_mesh=True)
        u = jr.normal(jr.key(1), (4, 2, 30))
        x_in = jr.uniform(jr.key(2), (30, 2))
        batched = jax.vmap(lambda u_: model(u_, x_in=x_in))(u)
        assert batched.shape == (4, 3, 30)
        np.testing.assert_allclose(
            batched[1], model(u[1], x_in=x_in), rtol=1e-5, atol=1e-6
        )

    def test_placeholder_is_used(self, make_model):
        model = make_model()
        x = jr.normal(jr.key(1), (2, 6, 6))
        shifted = eqx.tree_at(lambda m: m.placeholder, model, model.placeholder + 1.0)
        assert not jnp.allclose(model(x), shifted(x))

    def test_init(self, make_model):
        model = make_model(init_std=0.02)
        for w in jtu.tree_leaves(
            [model.lifting.weights, model.projection.weights, model.blocks[0].mlp]
        ):
            if hasattr(w, "ndim") and w.ndim == 2:
                assert float(jnp.std(w)) < 0.05
                assert float(jnp.max(jnp.abs(w))) <= 2.0
        assert model.placeholder.shape == (16,)
        assert float(jnp.max(model.placeholder)) <= 1.0 / 16
        default_init = make_model(init_std=None)
        assert float(jnp.std(default_init.lifting.weights[0])) > 0.1

    def test_grad_finite(self, make_model):
        model = make_model(use_mesh=True)
        u = jr.normal(jr.key(1), (2, 30))
        x_in = jr.uniform(jr.key(2), (30, 2))
        grads = eqx.filter_grad(lambda m: jnp.sum(m(u, x_in=x_in) ** 2))(model)
        for leaf in jtu.tree_leaves(eqx.filter(grads, eqx.is_array)):
            assert jnp.all(jnp.isfinite(leaf))

    def test_dropout(self, make_model):
        model = make_model(dropout=0.3)
        x = jr.normal(jr.key(1), (2, 6, 6))
        det = model(x, inference=True)
        assert not jnp.allclose(model(x, key=jr.key(2)), det)
        np.testing.assert_allclose(model(x, key=jr.key(2), inference=True), det)
        with pytest.raises(ValueError, match="PRNG key"):
            model(x)

    def test_save_load_roundtrip(self, tmp_path, small_params):
        hyperparams = {
            "in_channels": 2,
            "out_channels": 3,
            "ndim": 2,
            "use_mesh": True,
            **small_params,
        }
        model = Transolver(key=jr.key(0), **hyperparams)
        path = tmp_path / "transolver.eqx"
        model.save(file=path, hyperparams=hyperparams)
        loaded = Transolver.load(
            file=path, make_fn=functools.partial(Transolver, key=jr.key(1))
        )
        u = jr.normal(jr.key(2), (2, 20))
        x_in = jr.uniform(jr.key(3), (20, 2))
        np.testing.assert_allclose(model(u, x_in=x_in), loaded(u, x_in=x_in))

    def test_astype_and_size(self, make_model):
        model = make_model().astype(jnp.bfloat16)
        assert model.placeholder.dtype == jnp.bfloat16
        size, n_params = make_model().size(return_n_params=True)
        assert size > 0 and n_params > 0

    def test_schema_pipeline_mesh(self, make_model):
        # Batch of 3 samples on a 5x4 grid flattened to 20 mesh nodes
        coords = jnp.broadcast_to(jr.uniform(jr.key(0), (1, 2, 5, 4)), (3, 2, 5, 4))
        fields = jr.normal(jr.key(1), (3, 1, 2, 5, 4))  # (b, t, c, 5, 4)
        bundle = DataBundle(fields=fields, coords=coords)
        batch = MeshInputSchema(time_axis=1, channel_axis=2).transform(bundle)
        assert batch["u"].shape == (3, 2, 20) and batch["x_in"].shape == (3, 20, 2)

        model = make_model(use_mesh=True)
        out = jax.vmap(lambda u, x: model(u, x_in=x))(batch["u"], batch["x_in"])
        assert out.shape == (3, 3, 20)
        # Dict batches can be passed as keyword arguments, as the evaluators do
        out_kw = jax.vmap(lambda **kw: model(**kw))(**batch)
        np.testing.assert_allclose(out, out_kw, rtol=1e-5, atol=1e-6)

        pred = BundleReconstructSchema(unflatten_time_steps=1).transform(
            out.reshape(3, 3, 5, 4), reference_bundle=bundle
        )
        assert pred.fields.shape == (3, 1, 3, 5, 4)

    def test_schema_pipeline_grid(self, make_model):
        coords = jnp.stack(
            jnp.meshgrid(jnp.linspace(0, 1, 6), jnp.linspace(0, 1, 6), indexing="ij")
        )
        fields = jr.normal(jr.key(1), (1, 1, 6, 6))  # (t, c, x, y)
        bundle = DataBundle(fields=fields, coords=coords)
        x = ConcatenateCoordsSchema(channel_axis=1).transform(bundle)  # (1, 3, 6, 6)
        model = make_model(in_channels=3)
        assert model(x[0]).shape == (3, 6, 6)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param({"in_channels": 0}, "in_channels", id="in-channels-zero"),
            pytest.param({"out_channels": 0}, "out_channels", id="out-channels-zero"),
            pytest.param({"ndim": 0}, "ndim", id="ndim-zero"),
            pytest.param(
                {"hidden_channels": 0}, "hidden_channels", id="hidden-channels-zero"
            ),
            pytest.param({"n_layers": 0}, "n_layers", id="n-layers-zero"),
            pytest.param({"ref": 0}, "ref", id="ref-zero"),
            pytest.param({"init_std": 0.0}, "init_std", id="init-std-zero"),
            pytest.param({"n_heads": 5}, "divisible", id="heads-not-divisible"),
            pytest.param(
                {"positional_embedding": "grid", "use_mesh": True},
                "use_mesh",
                id="positional-embedding-with-mesh",
            ),
            pytest.param(
                {"positional_embedding": "grid", "unified_pos": True},
                "unified_pos",
                id="positional-embedding-with-unified-pos",
            ),
            pytest.param(
                {"positional_embedding": GridEmbeddingNd(2, 3)},
                "ndim",
                id="positional-embedding-ndim-mismatch",
            ),
            pytest.param({"kernel_size": 4}, "kernel_size", id="kernel-size-even"),
        ],
    )
    def test_init_raises(self, kwargs, match, make_model):
        with pytest.raises(ValueError, match=match):
            make_model(**kwargs)

    @pytest.mark.parametrize(
        "use_mesh, u_shape, x_in_shape, match",
        [
            pytest.param(True, (2, 10), None, "x_in", id="mesh-missing-x-in"),
            pytest.param(True, (2, 10), (9, 2), "shaped", id="mesh-x-in-mismatch"),
            pytest.param(False, (2, 6, 6), (36, 2), "use_mesh", id="grid-with-x-in"),
            pytest.param(False, (2, 6), None, "spatial axes", id="grid-wrong-ndim"),
            pytest.param(False, (5, 6, 6), None, "channels", id="grid-wrong-channels"),
        ],
    )
    def test_call_raises(self, use_mesh, u_shape, x_in_shape, match, make_model):
        model = make_model(use_mesh=use_mesh)
        x_in = None if x_in_shape is None else jnp.ones(x_in_shape)
        with pytest.raises(ValueError, match=match):
            model(jnp.ones(u_shape), x_in=x_in)
