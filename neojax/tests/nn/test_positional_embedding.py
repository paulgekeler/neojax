import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.positional_embedding import (
    GridEmbeddingNd,
    ReferenceGridEmbeddingNd,
    RotaryEmbedding,
    SinusoidalEmbedding,
)
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

    def test_linear_values(self):
        emb = SinusoidalEmbedding(
            in_channels=2, num_frequencies=3, embedding_type="linear"
        )
        x = jnp.array([[0.1, 0.7], [0.3, 0.2]], dtype=jnp.float32)
        expected = []
        for p in x:
            row = []
            for coord in p:
                for k in range(3):
                    row += [
                        jnp.sin((k + 1) * jnp.pi * coord),
                        jnp.cos((k + 1) * jnp.pi * coord),
                    ]
            expected.append(row)
        np.testing.assert_allclose(emb(jnp.array(x)), jnp.array(expected), atol=1e-5)

    def test_linear_matches_gaot_node_position_encoding_up_to_sign(self):
        # GAOT: phi = pi * (x + 1), angles = k * phi for k = 1..L
        num_freq = 4
        emb = SinusoidalEmbedding(
            in_channels=2, num_frequencies=num_freq, embedding_type="linear"
        )
        x = jr.uniform(jr.key(0), (9, 2), minval=-1.0, maxval=1.0)
        phi = jnp.pi * (x + 1.0)
        k = jnp.arange(1, num_freq + 1)
        angles = k[None, :, None] * phi[:, None, :]  # (n, freq, dim)
        reference = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=2)
        reference = reference.reshape(9, -1)
        # same multiset of channels per point up to sign
        np.testing.assert_allclose(
            jnp.sort(jnp.abs(emb(x)), axis=-1),
            jnp.sort(jnp.abs(reference), axis=-1),
            atol=1e-4,
        )

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=0, num_frequencies=2)
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=2, num_frequencies=0)
        with pytest.raises(ValueError):
            SinusoidalEmbedding(in_channels=2, num_frequencies=2, max_positions=0)
        with pytest.raises((ValueError, TypeCheckError)):
            SinusoidalEmbedding(
                in_channels=2, num_frequencies=2, embedding_type="invalid"
            )

    def test_channel_mismatch_raises(self):
        emb = SinusoidalEmbedding(in_channels=2, num_frequencies=2)
        with pytest.raises(ValueError):
            emb(jnp.ones((5, 3)))


class TestReferenceGridEmbeddingNd:
    @pytest.mark.parametrize(
        "ndim,ref,spatial",
        [(1, 4, (12,)), (2, 3, (5, 6)), (3, 2, (4, 3, 2))],
    )
    def test_dimensions(self, ndim, ref, spatial):
        emb = ReferenceGridEmbeddingNd(ndim=ndim, ref=ref)
        assert emb.n_features == ref**ndim
        coords = emb.regular_grid(spatial)
        assert coords.shape == (ndim, *spatial)
        assert emb(coords).shape == (ref**ndim, *spatial)

    def test_point_list_matches_grid(self):
        emb = ReferenceGridEmbeddingNd(ndim=2, ref=3)
        grid = emb.regular_grid((4, 5))
        as_points = grid.reshape(2, -1)
        assert jnp.allclose(emb(grid).reshape(9, -1), emb(as_points))

    def test_distances(self):
        emb = ReferenceGridEmbeddingNd(ndim=2, ref=2)
        # Reference nodes are the unit square corners (ij ordering)
        out = emb(jnp.array([[0.0], [0.0]]))
        assert jnp.allclose(out[:, 0], jnp.array([0.0, 1.0, 1.0, jnp.sqrt(2.0)]))

    def test_grid_boundaries(self):
        emb = ReferenceGridEmbeddingNd(ndim=1, ref=2, grid_boundaries=((0, 4),))
        out = emb(jnp.array([[1.0]]))
        assert jnp.allclose(out[:, 0], jnp.array([1.0, 3.0]))

    def test_raises(self):
        with pytest.raises(ValueError):
            ReferenceGridEmbeddingNd(ndim=0)
        with pytest.raises(ValueError):
            ReferenceGridEmbeddingNd(ndim=2, ref=0)
        with pytest.raises(ValueError):
            ReferenceGridEmbeddingNd(ndim=2, grid_boundaries=((0, 1),))
        emb = ReferenceGridEmbeddingNd(ndim=2)
        with pytest.raises(ValueError):
            emb(jnp.ones((3, 4)))
        with pytest.raises(ValueError):
            emb.regular_grid((4,))


class TestRotaryEmbedding:
    def test_matches_manual_rotation(self):
        dim, t, theta = 4, 5, 100.0
        rope = RotaryEmbedding(dim, theta=theta)
        x = jr.normal(jr.key(0), (dim, t))
        out = np.asarray(rope(x))
        xn = np.asarray(x)
        for pos in range(t):
            for i in range(dim // 2):
                angle = pos * theta ** (-2.0 * i / dim)
                c, s = np.cos(angle), np.sin(angle)
                a, b = xn[2 * i, pos], xn[2 * i + 1, pos]
                np.testing.assert_allclose(out[2 * i, pos], a * c - b * s, atol=1e-5)
                np.testing.assert_allclose(
                    out[2 * i + 1, pos], a * s + b * c, atol=1e-5
                )

    def test_position_zero_is_unchanged_and_norm_preserved(self):
        rope = RotaryEmbedding(8)
        x = jr.normal(jr.key(1), (3, 8, 6))  # (heads, dim, t)
        out = rope(x)
        np.testing.assert_allclose(out[..., 0], x[..., 0], atol=1e-6)
        np.testing.assert_allclose(
            jnp.linalg.norm(out, axis=-2), jnp.linalg.norm(x, axis=-2), rtol=1e-5
        )

    def test_inner_products_depend_on_relative_position_only(self):
        rope = RotaryEmbedding(8)
        q = jr.normal(jr.key(2), (8, 1))
        k = jr.normal(jr.key(3), (8, 1))
        q_all = rope(jnp.broadcast_to(q, (8, 7)))
        k_all = rope(jnp.broadcast_to(k, (8, 7)))
        scores = np.asarray(q_all.T @ k_all)  # (query pos, key pos)
        for offset in range(-3, 4):
            diag = np.diagonal(scores, offset=offset)
            np.testing.assert_allclose(diag, diag[0], atol=1e-4)
        # and it does depend on the offset
        assert abs(scores[0, 0] - scores[0, 3]) > 1e-3

    def test_jittable_and_dtype(self):
        rope = RotaryEmbedding(4)
        x = jr.normal(jr.key(4), (2, 4, 5))
        assert_filter_jittable(rope, x)
        assert rope(x.astype(jnp.bfloat16)).dtype == jnp.bfloat16

    def test_invalid_arguments_raise(self):
        for kwargs in [{"dim": 0}, {"dim": 3}, {"dim": 4, "theta": 0.0}]:
            with pytest.raises(ValueError):
                RotaryEmbedding(**kwargs)
        with pytest.raises(ValueError):
            RotaryEmbedding(4)(jnp.ones((6, 5)))
