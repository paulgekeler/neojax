import jax.numpy as jnp
import pytest

from neojax.nn.positional_embedding import GridEmbeddingNd


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
