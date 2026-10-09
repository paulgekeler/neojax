import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.positional_embedding import GridEmbeddingNd, SinusoidalEmbedding
from neojax.tests.conftest import assert_filter_jittable


class TestGridEmbeddingNd:
    def test_dimensions(self):
        # 1D: one grid channel appended
        emb_1d = GridEmbeddingNd(in_channels=3, ndim=1)
        assert emb_1d(jnp.ones((3, 12))).shape == (4, 12)

        # 1D with explicit boundaries
        emb_1d_exp = GridEmbeddingNd(in_channels=1, ndim=1, grid_boundaries=((0, 1),))
        assert emb_1d_exp(jnp.ones((1, 12))).shape == (2, 12)

        # 2D with explicit boundaries: one grid per spatial dim
        emb_2d = GridEmbeddingNd(
            in_channels=2, ndim=2, grid_boundaries=((0, 5), (0, 5))
        )
        assert emb_2d(jnp.ones((2, 10, 10))).shape == (4, 10, 10)

        # 3D
        emb_3d = GridEmbeddingNd(in_channels=2, ndim=3)
        assert emb_3d(jnp.ones((2, 4, 4, 4))).shape == (5, 4, 4, 4)

        # 4D
        emb_4d = GridEmbeddingNd(in_channels=1, ndim=4)
        assert emb_4d(jnp.ones((1, 4, 4, 4, 4))).shape == (5, 4, 4, 4, 4)

    def test_n_extra_channels(self):
        assert GridEmbeddingNd(in_channels=1, ndim=2).n_extra_channels == 2
        assert GridEmbeddingNd(in_channels=5, ndim=2).n_extra_channels == 2
        assert GridEmbeddingNd(in_channels=5, ndim=3).n_extra_channels == 3

    def test_default_bounds(self):
        emb = GridEmbeddingNd(in_channels=2, ndim=2)
        out = emb(jnp.zeros((2, 5, 5)))
        assert out.shape == (4, 5, 5)
        # Grid channels are the coordinates on [0, 1] along each axis.
        assert jnp.allclose(out[2, :, 0], jnp.linspace(0, 1, 5))
        assert jnp.allclose(out[2, 0, :], 0.0)
        assert jnp.allclose(out[3, 0, :], jnp.linspace(0, 1, 5))
        assert jnp.allclose(out[3, :, 0], 0.0)

    def test_signal_preserved(self):
        emb = GridEmbeddingNd(in_channels=2, ndim=1)
        x = jnp.arange(2 * 6, dtype=jnp.float32).reshape(2, 6)
        assert jnp.array_equal(emb(x)[:2], x)

    def test_channel_mismatch_raises(self):
        emb = GridEmbeddingNd(in_channels=2, ndim=2)
        with pytest.raises(ValueError):
            emb(jnp.ones((1, 10, 10)))
        with pytest.raises(ValueError):
            emb(jnp.ones((3, 10, 10)))


class TestSinusoidalEmbedding:
    @pytest.mark.parametrize("embedding_type", ["transformer", "nerf"])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_shapes(self, embedding_type, d):
        emb = SinusoidalEmbedding(
            in_channels=d, num_frequencies=4, embedding_type=embedding_type
        )
        assert emb.out_channels == 2 * 4 * d
        x = jr.uniform(jr.key(0), (7, d))
        assert emb(x).shape == (7, 2 * 4 * d)
        # arbitrary leading dims
        assert emb(x.reshape(7, 1, d)).shape == (7, 1, 2 * 4 * d)
        assert_filter_jittable(emb, x)

    def test_nerf_values(self):
        emb = SinusoidalEmbedding(
            in_channels=2, num_frequencies=3, embedding_type="nerf"
        )
        x = jnp.array([[0.1, 0.7], [0.3, 0.2]], dtype=jnp.float32)
        expected = []
        for p in x:
            row = []
            for coord in p:
                for k in range(3):
                    row += [
                        jnp.sin(2.0**k * jnp.pi * coord),
                        jnp.cos(2.0**k * jnp.pi * coord),
                    ]
            expected.append(row)
        np.testing.assert_allclose(emb(jnp.array(x)), jnp.array(expected), atol=1e-5)

    def test_transformer_values(self):
        emb = SinusoidalEmbedding(
            in_channels=1,
            num_frequencies=4,
            embedding_type="transformer",
            max_positions=100,
        )
        x = jnp.array([[0.5], [2.0]], dtype=jnp.float32)
        freqs = (1 / 100) ** (2.0 * jnp.arange(4) / 4)
        expected = jnp.stack([jnp.sin(x * freqs), jnp.cos(x * freqs)], axis=-1).reshape(
            2, 8
        )
        np.testing.assert_allclose(emb(jnp.array(x)), expected, atol=1e-5)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=0, num_frequencies=2)
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=2, num_frequencies=0)
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=2, num_frequencies=2, max_positions=0)

    def test_channel_mismatch_raises(self):
        emb = SinusoidalEmbedding(in_channels=2, num_frequencies=2)
        with pytest.raises(ValueError):
            emb(jnp.ones((5, 3)))
