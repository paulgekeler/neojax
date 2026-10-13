import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from jaxtyping import Array

from neojax.nn.physics_attention import PhysicsAttention
from neojax.tests.conftest import assert_filter_jittable


@pytest.fixture
def softmax():
    def f(x, axis):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    return f


@pytest.fixture
def reference_forward():
    def f(attn: PhysicsAttention, x: Array) -> Array:
        """Naive loop implementation of Physics-Attention in (N, C) layout.

        Mirrors the reference implementation of thuml/Transolver with explicit
        per-head loops. Only valid for the pointwise (mesh) variant.
        """
        c, n = x.shape
        h, dh = attn.n_heads, attn.dim_head
        xt = x.T  # (n, c)
        w_x, b_x = (
            jnp.asarray(attn.in_project_x.weights[0]),
            jnp.asarray(attn.in_project_x.biases[0]),
        )
        w_fx, b_fx = (
            jnp.asarray(attn.in_project_fx.weights[0]),
            jnp.asarray(attn.in_project_fx.biases[0]),
        )
        w_s, b_s = (
            jnp.asarray(attn.in_project_slice.weights[0]),
            jnp.asarray(attn.in_project_slice.biases[0]),
        )
        wq, wk, wv = (jnp.asarray(m.weight) for m in (attn.to_q, attn.to_k, attn.to_v))
        w_o, b_o = (
            jnp.asarray(attn.to_out.weights[0]),
            jnp.asarray(attn.to_out.biases[0]),
        )
        temp = jnp.asarray(attn.temperature)[0, :, 0]
        if attn.temperature_range is not None:
            temp = jnp.clip(temp, *attn.temperature_range)

        x_mid = xt @ w_x.T + b_x  # (n, h*dh)
        fx_mid = xt @ w_fx.T + b_fx
        heads_out = []
        for i in range(h):
            xm = x_mid[:, i * dh : (i + 1) * dh]
            fm = fx_mid[:, i * dh : (i + 1) * dh]
            sw = jax.nn.softmax((xm @ w_s.T + b_s) / temp[i], axis=-1)  # (n, g)
            norm = sw.sum(0)  # (g,)
            tok = (sw.T @ fm) / (norm[:, None] + 1e-5)  # (g, dh)
            q, k, v = tok @ wq.T, tok @ wk.T, tok @ wv.T
            att = jax.nn.softmax(q @ k.T * dh**-0.5, axis=-1)
            heads_out.append(sw @ (att @ v))  # (n, dh)
        out = jnp.concatenate(heads_out, axis=-1) @ w_o.T + b_o  # (n, c)
        return out.T

    return f


@pytest.mark.usefixtures("reference_forward")
class TestPhysicsAttention:
    @pytest.mark.parametrize(
        "ndim,shape",
        [
            (None, (16, 50)),
            (None, (16, 6, 5)),
            (1, (16, 20)),
            (2, (16, 7, 6)),
            (3, (16, 4, 4, 3)),
        ],
    )
    def test_dimensions(self, ndim, shape):
        attn = PhysicsAttention(jr.key(0), dim=16, n_heads=4, slice_num=8, ndim=ndim)
        x = jr.normal(jr.key(1), shape)
        assert attn(x).shape == shape

    def test_matches_reference_implementation(self, reference_forward):
        attn = PhysicsAttention(jr.key(0), dim=16, n_heads=4, slice_num=8)
        # Random weights, biases and temperature to exercise all paths
        leaves, treedef = jtu.tree_flatten(eqx.filter(attn, eqx.is_array))
        keys = jr.split(jr.key(7), len(leaves))
        leaves = [
            jr.normal(k, leaf.shape) * 0.3 if leaf.ndim > 0 else leaf
            for k, leaf in zip(keys, leaves, strict=True)
        ]
        attn = eqx.tree_at(
            lambda m: m.temperature,
            eqx.combine(jtu.tree_unflatten(treedef, leaves), attn),
            jnp.array([0.3, 0.5, 1.0, 2.0]).reshape(1, 4, 1),
        )
        x = jr.normal(jr.key(1), (16, 30))
        expected = reference_forward(attn, x)
        np.testing.assert_allclose(attn(x), expected, rtol=1e-4, atol=1e-5)

    def test_slice_weights_are_partition_of_unity(self):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=5)
        x = jr.normal(jr.key(1), (8, 20))
        x_mid = attn._split_heads(attn.in_project_x(x))
        w = jax.nn.softmax(attn.in_project_slice(x_mid) / attn.temperature, axis=0)
        np.testing.assert_allclose(w.sum(0), 1.0, rtol=1e-5)

    def test_permutation_equivariance_on_mesh(self):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=5)
        x = jr.normal(jr.key(1), (8, 20))
        perm = jr.permutation(jr.key(2), 20)
        np.testing.assert_allclose(
            attn(x)[:, perm], attn(x[:, perm]), rtol=1e-4, atol=1e-5
        )

    def test_flat_grid_equals_mesh(self):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=5)
        x = jr.normal(jr.key(1), (8, 4, 5))
        np.testing.assert_allclose(
            attn(x).reshape(8, -1), attn(x.reshape(8, -1)), rtol=1e-5, atol=1e-6
        )

    def test_conv_variant_is_translation_insensitive_to_padding_size(self):
        # Conv projections must preserve spatial size for all odd kernel sizes
        for k in (1, 3, 5):
            attn = PhysicsAttention(
                jr.key(0), dim=8, n_heads=2, slice_num=4, ndim=2, kernel_size=k
            )
            assert attn(jnp.ones((8, 6, 7))).shape == (8, 6, 7)

    @pytest.mark.parametrize("ndim", [None, 2])
    def test_jittable_and_grad(self, ndim):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=4, ndim=ndim)
        x = jr.normal(jr.key(1), (8, 6, 6) if ndim else (8, 36))
        assert_filter_jittable(attn, x)

        grads = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(attn)
        for leaf in jtu.tree_leaves(eqx.filter(grads, eqx.is_array)):
            assert jnp.all(jnp.isfinite(leaf))

    def test_temperature_clipping(self):
        x = jr.normal(jr.key(1), (8, 12))
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=4)
        far = eqx.tree_at(lambda m: m.temperature, attn, jnp.full((1, 2, 1), 100.0))
        clipped = eqx.tree_at(lambda m: m.temperature, attn, jnp.full((1, 2, 1), 5.0))
        np.testing.assert_allclose(far(x), clipped(x), rtol=1e-6)

        # A (negative) temperature produces finite outputs only because of clipping
        neg = eqx.tree_at(lambda m: m.temperature, attn, jnp.full((1, 2, 1), -1.0))
        low = eqx.tree_at(lambda m: m.temperature, attn, jnp.full((1, 2, 1), 0.1))
        np.testing.assert_allclose(neg(x), low(x), rtol=1e-6)

        unclipped = PhysicsAttention(
            jr.key(0), dim=8, n_heads=2, slice_num=4, temperature_range=None
        )
        assert unclipped.temperature_range is None
        assert unclipped(x).shape == x.shape

    def test_dropout(self):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, slice_num=4, dropout=0.5)
        x = jr.normal(jr.key(1), (8, 12))
        det = attn(x, inference=True)
        np.testing.assert_allclose(det, attn(x, key=jr.key(3), inference=True))
        out_a = attn(x, key=jr.key(3))
        out_b = attn(x, key=jr.key(4))
        assert not jnp.allclose(out_a, det)
        assert not jnp.allclose(out_a, out_b)
        np.testing.assert_allclose(out_a, attn(x, key=jr.key(3)))
        with pytest.raises(ValueError):
            attn(x)

    def test_default_dim_head_and_orthogonal_slice_init(self):
        attn = PhysicsAttention(jr.key(0), dim=32, n_heads=4, slice_num=6)
        assert attn.dim_head == 8
        w = attn.in_project_slice.weights[0]  # (g, dh) semi-orthogonal
        np.testing.assert_allclose(w @ w.T, np.eye(6), atol=1e-5)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param(
                {"dim": 10, "n_heads": 4}, "divisible", id="dim-not-divisible"
            ),
            pytest.param({"dim": 0}, "`dim`", id="dim-zero"),
            pytest.param({"n_heads": 0}, "`n_heads`", id="n-heads-zero"),
            pytest.param({"slice_num": 0}, "`slice_num`", id="slice-num-zero"),
            pytest.param({"dim_head": 0}, "`dim_head`", id="dim-head-zero"),
            pytest.param({"dropout": 1.0}, "`dropout`", id="dropout-one"),
            pytest.param({"ndim": 0}, "`ndim`", id="ndim-zero"),
            pytest.param(
                {"ndim": 2, "kernel_size": 4}, "`kernel_size`", id="kernel-size-even"
            ),
            pytest.param(
                {"temperature_range": (0.0, 1.0)},
                "`temperature_range`",
                id="temperature-range-zero-low",
            ),
        ],
    )
    def test_init_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PhysicsAttention(jr.key(0), **{"dim": 8, **kwargs})

    @pytest.mark.parametrize(
        "ndim, shape",
        [
            pytest.param(None, (7, 10), id="mesh-wrong-channels"),
            pytest.param(2, (8, 10), id="grid-wrong-spatial-axes"),
        ],
    )
    def test_call_raises(self, ndim, shape):
        attn = PhysicsAttention(jr.key(0), dim=8, n_heads=2, ndim=ndim)
        with pytest.raises(ValueError):
            attn(jnp.ones(shape))
