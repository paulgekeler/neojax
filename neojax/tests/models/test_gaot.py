import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.models.gaot import GAOT
from neojax.tests.conftest import (
    assert_filter_jittable,
    assert_reduced_precision_close_to_full,
)

N, M = 40, 25


@pytest.fixture
def make_model():
    def f(ndim=2, **kwargs):
        defaults = dict(
            key=jr.key(0),
            in_channels=2,
            out_channels=3,
            latent_resolution=(8,) * ndim if ndim == 2 else (4,) * ndim,
            patch_size=2,
            lifting_channels=4 if ndim == 2 else 3,  # 2**ndim * c divisible by 2*ndim
            hidden_channels=16,
            n_layers=3,
            n_heads=4,
            radius=0.45,
            max_neighbors=16,
            gno_hidden_channels=8,
            gno_n_layers=2,
            attention_dim=8,
        )
        return GAOT(**(defaults | kwargs))

    return f


@pytest.fixture
def make_data():
    def f(ndim=2, n=N, m=M):
        k1, k2, k3 = jr.split(jr.key(1), 3)
        x_in = jr.uniform(k1, (n, ndim), minval=-1.0, maxval=1.0)
        x_out = jr.uniform(k2, (m, ndim), minval=-1.0, maxval=1.0)
        return jr.normal(k3, (2, n)), x_in, x_out

    return f


@pytest.mark.usefixtures("make_model", "make_data")
class TestGAOT:
    @pytest.mark.parametrize("ndim", [2, 3])
    def test_shapes_and_jit(self, ndim, make_model, make_data):
        model = make_model(ndim)
        u, x_in, x_out = make_data(ndim)
        assert model(u, x_in=x_in).shape == (3, N)
        assert model(u, x_in=x_in, x_out=x_out).shape == (3, M)
        assert bool(jnp.all(jnp.isfinite(model(u, x_in=x_in, x_out=x_out))))

    def test_jittable(self, make_model, make_data):
        ndim = 2
        model = make_model(ndim)
        u, x_in, x_out = make_data(ndim)
        assert_filter_jittable(lambda m: m(u, x_in=x_in, x_out=x_out), model)

    @pytest.mark.parametrize("positional_embedding", ["absolute", "rope"])
    @pytest.mark.parametrize("scales", [(1.0,), (0.5, 2.0)])
    def test_positional_embeddings_and_scales(
        self, positional_embedding, scales, make_model, make_data
    ):
        model = make_model(positional_embedding=positional_embedding, scales=scales)
        u, x_in, _ = make_data()
        out = model(u, x_in=x_in)
        assert out.shape == (3, N) and bool(jnp.all(jnp.isfinite(out)))
        assert (model.pos_embedding is None) == (positional_embedding == "rope")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"attention_type": "dot_product"},
            {"use_attention": False},
            {"use_geoembed": False},
            {"embedding_method": "pointnet", "pooling": "mean"},
            {"use_scale_weights": True, "scales": (1.0, 2.0)},
            {"node_embedding": True, "node_embedding_frequencies": 2},
            {"n_kv_heads": 2, "n_layers": 4, "use_long_range_skip": False},
            {"gno_transform_type": "nonlinear"},
            {"gno_query_chunk_size": 7},
        ],
    )
    def test_variants_run(self, kwargs, make_model, make_data):
        model = make_model(**kwargs)
        u, x_in, x_out = make_data()
        out = model(u, x_in=x_in, x_out=x_out)
        assert out.shape == (3, M) and bool(jnp.all(jnp.isfinite(out)))

    def test_absolute_embedding_matches_reference_formula(self, make_model):
        model = make_model()
        positions = np.asarray(model._patch_positions(jnp.float32))  # (16, 2)
        ndim, embed_dim = 2, 2**2 * 4

        # reference: GAOT._compute_absolute_embeddings
        freqs_per_dim = embed_dim // (2 * ndim)
        inv_freq = 1.0 / (10000 ** (np.arange(freqs_per_dim) / freqs_per_dim))
        angles = positions[:, :, None] * inv_freq[None, None, :]
        reference = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
        reference = reference.reshape(positions.shape[0], -1)

        ours = np.asarray(model.pos_embedding(jnp.asarray(positions)))
        assert ours.shape == reference.shape
        # same encoding up to the order of the channels
        np.testing.assert_allclose(
            np.sort(ours, axis=-1), np.sort(reference, axis=-1), atol=1e-4
        )

    def test_precomputed_neighbors(self, make_model, make_data):
        model = make_model(scales=(1.0, 2.0))
        u, x_in, x_out = make_data()
        in_nb, out_nb = model.build_neighbors(x_in, x_out)
        assert len(in_nb) == len(out_nb) == 2
        np.testing.assert_allclose(
            model(u, x_in=x_in, x_out=x_out),
            model(u, x_in=x_in, x_out=x_out, in_neighbors=in_nb, out_neighbors=out_nb),
            rtol=1e-5,
            atol=1e-6,
        )
        # default queries at x_in
        in_nb, out_nb = model.build_neighbors(x_in)
        assert out_nb[0].index.shape[0] == N

    def test_domain_bounds_rescale_coordinates(self, make_model, make_data):
        u, x_in, x_out = make_data()
        bounds = ((0.0, 2.0), (-1.0, 3.0))
        low = jnp.array([0.0, -1.0])
        width = jnp.array([2.0, 4.0])
        to_domain = lambda x: low + (x + 1.0) / 2.0 * width  # noqa: E731

        plain = make_model()
        bounded = make_model(domain_bounds=bounds)
        np.testing.assert_allclose(
            plain(u, x_in=x_in, x_out=x_out),
            bounded(u, x_in=to_domain(x_in), x_out=to_domain(x_out)),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_invariant_to_input_point_order(self, make_model, make_data):
        model = make_model(max_neighbors=N)  # no truncation
        u, x_in, x_out = make_data()
        perm = jr.permutation(jr.key(5), N)
        np.testing.assert_allclose(
            model(u, x_in=x_in, x_out=x_out),
            model(u[:, perm], x_in=x_in[perm], x_out=x_out),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_equivariant_to_query_order(self, make_model, make_data):
        model = make_model()
        u, x_in, x_out = make_data()
        perm = jr.permutation(jr.key(6), M)
        np.testing.assert_allclose(
            model(u, x_in=x_in, x_out=x_out)[:, perm],
            model(u, x_in=x_in, x_out=x_out[perm]),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_vmap_over_functions_and_geometries(self, make_model, make_data):
        model = make_model()
        u, x_in, x_out = make_data()
        u_batch = jr.normal(jr.key(7), (3, 2, N))
        out = jax.vmap(lambda u_: model(u_, x_in=x_in, x_out=x_out))(u_batch)
        assert out.shape == (3, 3, M)
        for b in range(3):
            np.testing.assert_allclose(
                out[b], model(u_batch[b], x_in=x_in, x_out=x_out), rtol=1e-4, atol=1e-5
            )

        # a different mesh per sample
        x_batch = jr.uniform(jr.key(8), (3, N, 2), minval=-1.0, maxval=1.0)
        out_geo = jax.vmap(lambda u_, x_: model(u_, x_in=x_, x_out=x_out))(
            u_batch, x_batch
        )
        assert out_geo.shape == (3, 3, M)
        np.testing.assert_allclose(
            out_geo[1],
            model(u_batch[1], x_in=x_batch[1], x_out=x_out),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_gradients_are_finite_and_nonzero(self, make_model, make_data):
        model = make_model(use_scale_weights=True, scales=(1.0, 2.0))
        u, x_in, x_out = make_data()
        grads = eqx.filter_grad(lambda m: jnp.sum(m(u, x_in=x_in, x_out=x_out) ** 2))(
            model
        )
        leaves = jtu.tree_leaves(grads)
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
        # every sub-network receives a gradient
        for part in [
            grads.lifting,
            grads.encoder,
            grads.patch_linear,
            grads.processor,
            grads.decoder,
            grads.projection,
        ]:
            assert sum(float(jnp.abs(g).sum()) for g in jtu.tree_leaves(part)) > 0

    def test_dropout_and_edge_drop_need_a_key(self, make_model, make_data):
        u, x_in, _ = make_data()
        model = make_model(
            attention_dropout=0.3,
            edge_drop_strategy="max_neighbors",
            edge_drop_max_neighbors=3,
        )
        clean = model(u, x_in=x_in, inference=True)
        np.testing.assert_array_equal(
            clean, model(u, x_in=x_in, key=jr.key(1), inference=True)
        )
        with pytest.raises(ValueError):
            model(u, x_in=x_in)

        dropped = model(u, x_in=x_in, key=jr.key(1))
        assert not np.allclose(clean, dropped)
        np.testing.assert_array_equal(dropped, model(u, x_in=x_in, key=jr.key(1)))
        assert not np.allclose(dropped, model(u, x_in=x_in, key=jr.key(2)))

    def test_save_and_load_weights(self, make_model, make_data, tmp_path):
        model = make_model()
        u, x_in, _ = make_data()
        path = tmp_path / "gaot.eqx"
        model.save_weights(path)
        loaded = make_model(key=jr.key(99)).load_weights(path)
        np.testing.assert_allclose(
            model(u, x_in=x_in), loaded(u, x_in=x_in), rtol=1e-6, atol=1e-6
        )
        size, n_params = model.size(return_n_params=True)
        assert size > 0 and n_params > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"in_channels": 0},
            {"out_channels": 0},
            {"lifting_channels": 0},
            {"patch_size": 0},
            {"latent_resolution": ()},
            {"latent_resolution": (8, 7)},  # not divisible by the patch size
            {"domain_bounds": ((0.0, 1.0),)},  # wrong number of axes
            {"domain_bounds": ((0.0, 1.0), (1.0, 1.0))},  # low >= high
            {"positional_embedding": "invalid"},
            # 3**2 * 1 not divisible by 2 * 2
            {"patch_size": 3, "latent_resolution": (9, 9), "lifting_channels": 1},
            {"positional_embedding": "rope", "hidden_channels": 20, "n_heads": 4},
            {"scales": ()},
            {"n_heads": 3},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, make_model):
        with pytest.raises((ValueError, TypeCheckError)):
            make_model(**kwargs)

    def test_invalid_inputs_raise(self, make_model, make_data):
        model = make_model()
        u, x_in, x_out = make_data()
        with pytest.raises((ValueError, TypeCheckError)):
            model(u[:1], x_in=x_in)
        with pytest.raises((ValueError, TypeCheckError)):
            model(u[:, :-1], x_in=x_in)
        with pytest.raises((ValueError, TypeCheckError)):
            model(u, x_in=jnp.ones((N, 3)))
        with pytest.raises((ValueError, TypeCheckError)):
            model(u, x_in=x_in, x_out=jnp.ones((M, 3)))
        with pytest.raises(ValueError):
            in_nb, out_nb = model.build_neighbors(x_in)
            model(u, x_in=x_in, in_neighbors=in_nb + in_nb, out_neighbors=out_nb)

    @pytest.mark.parametrize("reduced_dtype", [jnp.float16, jnp.bfloat16])
    def test_build_neighbors_mixed_precision(
        self, make_model, make_data, reduced_dtype
    ):
        model = make_model()
        _, x_in, x_out = make_data()
        # assert with zero tolerance (want exact match because of int/bool outputs)
        assert_reduced_precision_close_to_full(
            model.build_neighbors,
            filter_kwarg_names=(),
            reduced_dtype=reduced_dtype,
            rtol=0.0,
            atol=0.0,
            x_in=x_in.astype(reduced_dtype),
            x_out=x_out.astype(reduced_dtype),
        )
