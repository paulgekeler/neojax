import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.geometric_embedding import GeometricEmbedding
from neojax.nn.neighbor_search import Neighbors, radius_neighbor_search
from neojax.tests.conftest import assert_filter_jittable

N, M, K, OUT = 60, 20, 10, 6


@pytest.fixture
def make_data():
    def f(d, radius=0.3):
        k1, k2 = jr.split(jr.key(0))
        data = jr.uniform(k1, (N, d))
        # the last query is far from all points: an empty neighborhood
        queries = jnp.concatenate([jr.uniform(k2, (M - 1, d)), jnp.full((1, d), 9.0)])
        return data, queries, radius_neighbor_search(data, queries, radius, K)

    return f


def reference_statistics(data, queries, neighbors):
    """Straightforward numpy version of the standardized statistical features."""
    data, queries = np.asarray(data), np.asarray(queries)
    index, mask = np.asarray(neighbors.index), np.asarray(neighbors.mask)
    d = data.shape[1]
    features = np.zeros((queries.shape[0], 3 + 2 * d))
    for i in range(queries.shape[0]):
        nbr = data[index[i][mask[i]]]
        if len(nbr) == 0:
            continue
        dist = np.linalg.norm(nbr - queries[i], axis=1)
        centroid = nbr.mean(axis=0)
        cov = (nbr - centroid).T @ (nbr - centroid) / len(nbr)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        features[i] = np.concatenate(
            [[len(nbr), dist.mean(), dist.var()], centroid - queries[i], eig]
        )
    std = features.std(axis=0, ddof=1)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - features.mean(axis=0)) / std


@pytest.mark.usefixtures("make_data")
class TestGeometricEmbedding:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_statistical_features_match_reference(self, d, make_data):
        data, queries, neighbors = make_data(d, radius=0.4)
        emb = GeometricEmbedding(jr.key(1), d, OUT)
        features = emb._statistical_features(data, queries, neighbors)
        np.testing.assert_allclose(
            features, reference_statistics(data, queries, neighbors), atol=1e-4
        )

    @pytest.mark.parametrize("method", ["statistical", "pointnet"])
    @pytest.mark.parametrize("d", [2, 3])
    def test_shapes_jit_and_non_negative(self, method, d, make_data):
        data, queries, neighbors = make_data(d)
        emb = GeometricEmbedding(jr.key(1), d, OUT, method=method)
        out = emb(data, queries, neighbors)
        assert out.shape == (OUT, M)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert bool(jnp.all(out >= 0))  # final ReLU
        assert_filter_jittable(emb, data, queries, neighbors)

    @pytest.mark.parametrize("method", ["statistical", "pointnet"])
    def test_padding_and_order_do_not_matter(self, method, make_data):
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 2, OUT, method=method)
        out = emb(data, queries, neighbors)

        # arbitrary (valid) indices in the padding
        garbage = jr.randint(jr.key(3), neighbors.index.shape, 0, N)
        garbled = Neighbors(
            index=jnp.where(neighbors.mask, neighbors.index, garbage),
            mask=neighbors.mask,
            sq_dist=neighbors.sq_dist,
        )
        np.testing.assert_allclose(out, emb(data, queries, garbled), atol=1e-5)

        # permuted neighbor order
        perm = jr.permutation(jr.key(4), K)
        permuted = Neighbors(
            index=neighbors.index[:, perm],
            mask=neighbors.mask[:, perm],
            sq_dist=neighbors.sq_dist[:, perm],
        )
        np.testing.assert_allclose(out, emb(data, queries, permuted), atol=1e-4)

    def test_pointnet_empty_neighborhood_is_zero(self, make_data):
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 2, OUT, method="pointnet")
        out = emb(data, queries, neighbors)
        np.testing.assert_array_equal(out[:, -1], 0.0)
        assert float(jnp.abs(out[:, :-1]).sum()) > 0

    def test_pointnet_pooling(self, make_data):
        data, queries, neighbors = make_data(2)
        kwargs = dict(key=jr.key(1), coord_dim=2, out_channels=OUT, method="pointnet")
        out_max = GeometricEmbedding(**kwargs, pooling="max")(data, queries, neighbors)
        out_mean = GeometricEmbedding(**kwargs, pooling="mean")(
            data, queries, neighbors
        )
        assert not np.allclose(out_max, out_mean)

    def test_pointnet_is_translation_invariant(self, make_data):
        # the neighbors enter relative to the query
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 2, OUT, method="pointnet")
        shift = jnp.array([3.0, -2.0])
        np.testing.assert_allclose(
            emb(data, queries, neighbors),
            emb(data + shift, queries + shift, neighbors),
            atol=1e-4,
        )

    def test_gradients(self, make_data):
        data, queries, neighbors = make_data(2)
        for method in ["statistical", "pointnet"]:
            emb = GeometricEmbedding(jr.key(1), 2, OUT, method=method)

            def loss(model, data, queries):
                return jnp.sum(model(data, queries, neighbors) ** 2)

            g_model = eqx.filter_grad(loss)(emb, data, queries)
            g_data = jax.grad(loss, argnums=1)(emb, data, queries)
            g_queries = jax.grad(loss, argnums=2)(emb, data, queries)
            for leaf in jax.tree_util.tree_leaves(g_model):
                assert bool(jnp.all(jnp.isfinite(leaf)))
            assert bool(jnp.all(jnp.isfinite(g_data)))
            assert bool(jnp.all(jnp.isfinite(g_queries)))
            if method == "statistical":
                # features are constant w.r.t. the coordinates
                np.testing.assert_array_equal(g_data, 0.0)
                np.testing.assert_array_equal(g_queries, 0.0)
            else:
                assert float(jnp.abs(g_queries).sum()) > 0

    def test_single_query_is_finite(self, make_data):
        data, queries, neighbors = make_data(2, radius=0.5)
        single = Neighbors(
            index=neighbors.index[:1],
            mask=neighbors.mask[:1],
            sq_dist=neighbors.sq_dist[:1],
        )
        emb = GeometricEmbedding(jr.key(1), 2, OUT)
        out = emb(data, queries[:1], single)
        assert out.shape == (OUT, 1) and bool(jnp.all(jnp.isfinite(out)))

    def test_vmap_over_geometries(self, make_data):
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 2, OUT)
        shifts = jnp.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 0.5]])
        out = jax.vmap(lambda s: emb(data + s, queries + s, neighbors))(shifts)
        assert out.shape == (3, OUT, M)

    def test_float16_cast(self, make_data):
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 2, OUT)
        ref_emb = emb(data, queries, neighbors)
        for dtype in [jnp.float16, jnp.bfloat16]:
            eps = float(jnp.finfo(dtype).eps)
            rtol = 1.0 * eps
            atol = rtol * float(jnp.abs(ref_emb).max())
            cast_emb = emb(
                jnp.astype(data, dtype), jnp.astype(queries, dtype), neighbors
            )
            np.testing.assert_allclose(ref_emb, cast_emb, rtol=rtol, atol=atol)

    def test_coordinate_mismatch_raises(self, make_data):
        data, queries, neighbors = make_data(2)
        emb = GeometricEmbedding(jr.key(1), 3, OUT)
        with pytest.raises(ValueError):
            emb(data, queries, neighbors)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"coord_dim": 0},
            {"out_channels": 0},
            {"hidden_channels": 0},
            {"method": "invalid"},
            {"pooling": "sum"},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        full = dict(coord_dim=2, out_channels=OUT) | kwargs
        with pytest.raises((ValueError, TypeCheckError)):
            GeometricEmbedding(jr.key(0), **full)
