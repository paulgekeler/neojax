import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.neighbor_attention import NeighborAttention
from neojax.tests.conftest import (
    assert_filter_jittable,
    assert_reduced_precision_close_to_full,
)

E, C, K = 3, 5, 6


@pytest.fixture
def inputs():
    k1, k2 = jr.split(jr.key(0))
    x = jnp.broadcast_to(jr.normal(k1, (E, C, 1)), (E, C, K))
    y = jr.normal(k2, (E, C, K))
    mask = jnp.arange(K)[None, :] < jnp.array([6, 4, 1, 0, 3])[:, None]
    return x, y, mask


@pytest.mark.usefixtures("inputs")
class TestNeighborAttention:
    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_weights_normalized_over_valid_neighbors(self, inputs, attention_type):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(1), E, attention_type, attention_dim=8)
        w = att(x, y, mask)
        assert w.shape == (C, K)
        np.testing.assert_array_equal(np.asarray(w)[~np.asarray(mask)], 0.0)
        sums = np.asarray(w.sum(axis=-1))
        # rows with neighbors sum to one, the empty row is all zero
        np.testing.assert_allclose(sums, [1, 1, 1, 0, 1], atol=1e-6)
        assert (np.asarray(w) >= 0).all()

    def test_cosine_values(self, inputs):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(1), E, "cosine")
        w = np.asarray(att(x, y, mask))
        xn, yn, m = np.asarray(x), np.asarray(y), np.asarray(mask)
        for i in range(C):
            logits = np.array(
                [
                    xn[:, i, j]
                    @ yn[:, i, j]
                    / np.linalg.norm(xn[:, i, j])
                    / np.linalg.norm(yn[:, i, j])
                    for j in range(K)
                ]
            )[m[i]]
            if logits.size == 0:
                continue
            expected = np.exp(logits - logits.max())
            np.testing.assert_allclose(w[i][m[i]], expected / expected.sum(), atol=1e-5)

    def test_dot_product_values(self, inputs):
        x, y, mask = inputs
        dim = 8
        att = NeighborAttention(jr.key(1), E, "dot_product", attention_dim=dim)
        w = np.asarray(att(x, y, mask))
        wq, bq = (
            np.asarray(att.query_proj.weights[0]),
            np.asarray(att.query_proj.biases[0]),
        )
        wk, bk = np.asarray(att.key_proj.weights[0]), np.asarray(att.key_proj.biases[0])
        xn, yn, m = np.asarray(x), np.asarray(y), np.asarray(mask)
        for i in range(C):
            logits = np.array(
                [
                    ((wq @ xn[:, i, j] + bq) @ (wk @ yn[:, i, j] + bk)) / np.sqrt(dim)
                    for j in range(K)
                ]
            )[m[i]]
            if logits.size == 0:
                continue
            expected = np.exp(logits - logits.max())
            np.testing.assert_allclose(w[i][m[i]], expected / expected.sum(), atol=1e-5)

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_padding_values_are_ignored(self, inputs, attention_type):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(1), E, attention_type)
        w = att(x, y, mask)
        # change coordinates of padded neighbors
        garbage = jr.normal(jr.key(7), y.shape) * 100.0
        w_garbage = att(x, jnp.where(mask[None], y, garbage), mask)
        np.testing.assert_allclose(w, w_garbage, atol=1e-6)

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_gradients_are_finite(self, inputs, attention_type):
        x, y, mask = inputs
        # points at the origin and an empty neighborhood
        x = x.at[:, 0].set(0.0)
        y = y.at[:, 1, 0].set(0.0)
        att = NeighborAttention(jr.key(1), E, attention_type)

        def loss(att, x, y):
            return jnp.sum(att(x, y, mask) * jnp.arange(K))

        grads = jax.grad(loss, argnums=(0, 1, 2))(att, x, y)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert np.isfinite(np.asarray(leaf)).all()

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_jittable(self, inputs, attention_type):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(1), E, attention_type)
        assert_filter_jittable(att, x, y, mask)

    def test_channel_mismatch_raises(self, inputs):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(1), E + 1)
        with pytest.raises(ValueError, match="channels"):
            att(x, y, mask)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"in_channels": 0},
            {"in_channels": 2, "attention_dim": 0},
            {"in_channels": 2, "attention_type": "invalid"},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises((ValueError, TypeCheckError)):
            NeighborAttention(jr.key(0), **kwargs)

    @pytest.mark.parametrize(
        "attention_type,reduced_dtype",
        (
            ("cosine", jnp.float16),
            ("cosine", jnp.bfloat16),
            ("dot_product", jnp.bfloat16),
            ("dot_product", jnp.float16),
        ),
    )
    def test_mixed_precision(self, inputs, attention_type, reduced_dtype):
        x, y, mask = inputs
        att = NeighborAttention(jr.key(0), E, attention_type)
        assert_reduced_precision_close_to_full(
            att,
            filter_kwarg_names=["mask"],
            reduced_dtype=reduced_dtype,
            x=x,
            y=y,
            mask=mask,
        )
