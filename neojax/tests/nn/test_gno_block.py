import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.gno_block import GNOBlock
from neojax.nn.mollifier_functions import make_weighting_fn
from neojax.nn.neighbor_search import radius_neighbor_search
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tests.conftest import assert_filter_jittable

N, M, C = 50, 20, 3


@pytest.fixture
def make_data():
    def f(d):
        k1, k2, k3 = jr.split(jr.key(0), 3)
        return jr.uniform(k1, (N, d)), jr.uniform(k2, (M, d)), jr.normal(k3, (C, N))

    return f


@pytest.fixture
def make_block():
    def f(**kwargs):
        radius = 0.3
        return GNOBlock(
            key=jr.key(3),
            in_channels=C,
            out_channels=C,
            coord_dim=2,
            radius=radius,
            max_neighbors=N,  # no truncation
            weighting_fn=make_weighting_fn("half_cos", radius**2),
            pos_embedding_channels=4,
            channel_mlp_hidden_layers=(8,),
            **kwargs,
        )

    return f


@pytest.mark.usefixtures("make_data")
class TestGNOBlock:
    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize("pos_embedding_type", ["transformer", "nerf", None])
    @pytest.mark.parametrize(
        "transform_type",
        ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"],
    )
    def test_shapes(self, d, pos_embedding_type, transform_type, make_data):
        y, x, f_y = make_data(d)
        block = GNOBlock(
            key=jr.key(1),
            in_channels=C,
            out_channels=C,
            coord_dim=d,
            radius=0.4,
            max_neighbors=16,
            transform_type=transform_type,
            pos_embedding_type=pos_embedding_type,
            pos_embedding_channels=4,
            channel_mlp_hidden_layers=(16, 16),
        )
        out = block(y, x, f_y)
        assert out.shape == (C, M)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert_filter_jittable(block, y, x, f_y)

    def test_kernel_in_channels(self):
        kwargs = dict(key=jr.key(0), in_channels=C, out_channels=C, coord_dim=2)
        # Embedded y and x: 2 * (2 * L * d), plus f for nonlinear
        block = GNOBlock(**kwargs, radius=0.1, pos_embedding_channels=4)
        assert block.kernel_in_channels == 2 * 2 * 4 * 2
        block = GNOBlock(
            **kwargs, radius=0.1, pos_embedding_channels=4, transform_type="nonlinear"
        )
        assert block.kernel_in_channels == 2 * 2 * 4 * 2 + C
        block = GNOBlock(**kwargs, radius=0.1, pos_embedding_type=None)
        assert block.kernel_in_channels == 2 * 2

    def test_precomputed_neighbors_equal_on_the_fly(self, make_data):
        y, x, f_y = make_data(2)
        block = GNOBlock(
            key=jr.key(1),
            in_channels=C,
            out_channels=C,
            coord_dim=2,
            radius=0.3,
            max_neighbors=8,
            pos_embedding_channels=4,
            channel_mlp_hidden_layers=(8,),
            query_chunk_size=7,
        )
        neighbors = radius_neighbor_search(y, x, 0.3, 8)
        np.testing.assert_allclose(
            block(y, x, f_y),
            block(y, x, f_y, neighbors=neighbors),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_vmap_over_functions_and_geometries(self, make_data):
        y, x, f_y = make_data(2)
        block = GNOBlock(
            key=jr.key(1),
            in_channels=C,
            out_channels=C,
            coord_dim=2,
            radius=0.3,
            max_neighbors=8,
            pos_embedding_channels=4,
            channel_mlp_hidden_layers=(8,),
        )
        f_batch = jr.normal(jr.key(5), (4, C, N))
        out = jax.vmap(block, in_axes=(None, None, 0))(y, x, f_batch)
        assert out.shape == (4, C, M)
        np.testing.assert_allclose(
            out[2], block(y, x, f_batch[2]), rtol=1e-5, atol=1e-6
        )

        # a different mesh per sample
        y_batch = jnp.stack([y, 1.0 - y])
        out = jax.vmap(block)(y_batch, jnp.stack([x, x]), f_batch[:2])
        np.testing.assert_allclose(
            out[1], block(1.0 - y, x, f_batch[1]), rtol=1e-5, atol=1e-6
        )

    def test_weighting_fn_and_mean_reduction(self, make_data):
        y, x, f_y = make_data(2)
        kwargs = dict(
            key=jr.key(1),
            in_channels=C,
            out_channels=C,
            coord_dim=2,
            radius=0.3,
            max_neighbors=8,
            pos_embedding_channels=4,
            channel_mlp_hidden_layers=(8,),
        )
        block = GNOBlock(
            **kwargs,
            weighting_fn=make_weighting_fn("bump", 0.3**2),
            reduction="mean",
        )
        assert block.integral_transform.reduction == "sum"
        assert block(y, x, f_y).shape == (C, M)

    def test_custom_channel_mlp(self, make_data):
        y, x, f_y = make_data(2)
        kwargs = dict(
            key=jr.key(1),
            in_channels=C,
            out_channels=C,
            coord_dim=2,
            radius=0.3,
            pos_embedding_type=None,
        )
        mlp = PointwiseMLP(key=jr.key(2), layers=(4, 8, C))
        block = GNOBlock(**kwargs, channel_mlp=mlp)
        assert block.integral_transform.channel_mlp is mlp
        assert block(y, x, f_y).shape == (C, M)

        with pytest.raises(ValueError):
            GNOBlock(
                **kwargs, channel_mlp=PointwiseMLP(key=jr.key(2), layers=(5, 8, C))
            )
        with pytest.raises(ValueError):
            GNOBlock(
                **kwargs, channel_mlp=PointwiseMLP(key=jr.key(2), layers=(4, 8, C + 1))
            )

    def test_invalid_arguments(self):
        kwargs = dict(key=jr.key(0), in_channels=C, out_channels=C, coord_dim=2)
        with pytest.raises(ValueError):
            GNOBlock(**kwargs, radius=0.0)
        with pytest.raises(ValueError):
            GNOBlock(**kwargs, radius=0.1, max_neighbors=0)
        with pytest.raises(ValueError):
            GNOBlock(**{**kwargs, "coord_dim": 0}, radius=0.1)
        with pytest.raises(ValueError):
            GNOBlock(**{**kwargs, "out_channels": C + 1}, radius=0.1)
        # Kernel-only transforms are free in the channels
        GNOBlock(
            **{**kwargs, "out_channels": C + 1},
            radius=0.1,
            transform_type="nonlinear_kernelonly",
        )


@pytest.mark.usefixtures("make_data", "make_block")
class TestMollifiedGNOBlock:
    def test_grad_wrt_queries_matches_finite_differences(self, make_data, make_block):
        y, x, f_y = make_data(2)
        block = make_block()

        def loss(x_):
            return block(y, x_, f_y).sum()

        grad = jax.grad(loss)(x)
        eps = 1e-3
        fd = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                e = jnp.zeros_like(x).at[i, j].set(eps)
                fd[i, j] = (loss(x + e) - loss(x - e)) / (2 * eps)
        np.testing.assert_allclose(grad, fd, rtol=5e-2, atol=5e-2)
        assert float(jnp.abs(grad).max()) > 0

    def test_grad_wrt_queries_with_precomputed_neighbors(self, make_data, make_block):
        # The weights must stay differentiable in x, although the
        # stored squared distances of the neighbors are constants
        y, x, f_y = make_data(2)
        block = make_block()
        neighbors = radius_neighbor_search(y, x, 0.3, N)

        live = jax.grad(lambda x_: block(y, x_, f_y).sum())(x)
        cached = jax.grad(lambda x_: block(y, x_, f_y, neighbors=neighbors).sum())(x)
        np.testing.assert_allclose(cached, live, rtol=1e-4, atol=1e-5)

    def test_quadrature_weights(self, make_data, make_block):
        y, x, f_y = make_data(2)
        block = make_block()
        delta = jr.uniform(jr.key(4), (N,)) + 0.5
        out = block(y, x, f_y, quadrature_weights=delta)
        assert out.shape == (C, M)
        np.testing.assert_allclose(
            block(y, x, f_y, quadrature_weights=jnp.full((N,), 2.0)),
            2.0 * block(y, x, f_y),
            rtol=1e-5,
            atol=1e-6,
        )
        assert not np.allclose(out, block(y, x, f_y))
