import jax.numpy as jnp

from neojax.nn.positional_embedding import GridEmbeddingNd


class TestGridEmbeddingNd:
    def test_dimensions(self):
        # 1D
        emb_1d = GridEmbeddingNd(in_channels=1, ndim=1)
        assert emb_1d(jnp.ones((3, 12))).shape == (4, 12)

        # 1D with explicit boundaries
        emb_1d_exp = GridEmbeddingNd(in_channels=1, ndim=1, grid_boundaries=((0, 1),))
        assert emb_1d_exp(jnp.ones((3, 12))).shape == (4, 12)

        # 2D with explicit boundaries
        emb_2d = GridEmbeddingNd(
            in_channels=2, ndim=2, grid_boundaries=((0, 5), (0, 5))
        )
        # Input (2, 10, 10) -> Output (2 + 2*2, 10, 10) = (6, 10, 10)
        assert emb_2d(jnp.ones((2, 10, 10))).shape == (6, 10, 10)

        # 3D
        emb_3d = GridEmbeddingNd(in_channels=1, ndim=3)
        # Input (2, 4, 4, 4) -> Output (2 + 3*1, 4, 4, 4) = (5, 4, 4, 4)
        assert emb_3d(jnp.ones((2, 4, 4, 4))).shape == (5, 4, 4, 4)

        # 4D
        emb_4d = GridEmbeddingNd(in_channels=1, ndim=4)
        # Input (1, 4, 4, 4, 4) -> Output (1 + 4*1, 4, 4, 4, 4) = (5, 4, 4, 4, 4)
        assert emb_4d(jnp.ones((1, 4, 4, 4, 4))).shape == (5, 4, 4, 4, 4)

    def test_default_bounds(self):
        emb = GridEmbeddingNd(in_channels=2, ndim=2)
        # Input (1, 10, 10) -> Output (1 + 2*2, 10, 10) = (5, 10, 10)
        assert emb(jnp.ones((1, 10, 10))).shape == (5, 10, 10)
