# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from neojax.nn.attention import MultiHeadAttention
from neojax.nn.positional_embedding import RotaryEmbedding
from neojax.tests.conftest import assert_filter_jittable

C, T = 16, 11


def manual_attention(attn, x, rope=False):
    """Plain numpy multi-head (grouped query) attention."""
    x = np.asarray(x)
    h, g, d = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = (np.asarray(attn.q_proj.weights[0]) @ x).reshape(h, d, T)
    k = (np.asarray(attn.k_proj.weights[0]) @ x).reshape(g, d, T)
    v = (np.asarray(attn.v_proj.weights[0]) @ x).reshape(g, d, T)
    if rope:
        q = np.asarray(attn.rope(jnp.asarray(q)))
        k = np.asarray(attn.rope(jnp.asarray(k)))
    out = np.zeros((h, d, T))
    for head in range(h):
        kv = head // (h // g)
        scores = q[head].T @ k[kv] / np.sqrt(d)  # (t_q, t_k)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        out[head] = (weights @ v[kv].T).T
    return np.asarray(attn.o_proj.weights[0]) @ out.reshape(C, T)


class TestMultiHeadAttention:
    @pytest.mark.parametrize("n_heads, n_kv_heads", [(4, 4), (4, 2), (4, 1), (2, None)])
    @pytest.mark.parametrize("use_rope", [False, True])
    def test_matches_manual(self, n_heads, n_kv_heads, use_rope):
        attn = MultiHeadAttention(jr.key(0), C, n_heads, n_kv_heads, use_rope=use_rope)
        x = jr.normal(jr.key(1), (C, T))
        out = attn(x)
        assert out.shape == (C, T)
        np.testing.assert_allclose(
            out, manual_attention(attn, x, rope=use_rope), rtol=1e-4, atol=1e-5
        )
        assert_filter_jittable(attn, x)

    def test_no_bias_parameters(self):
        attn = MultiHeadAttention(jr.key(0), C, 4)
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]:
            assert all(b is None for b in proj.biases)
        assert attn.k_proj.weights[0].shape == (C, C)
        gqa = MultiHeadAttention(jr.key(0), C, 4, 2)
        assert gqa.k_proj.weights[0].shape == (C // 2, C)

    def test_permutation_equivariant_without_rope(self):
        attn = MultiHeadAttention(jr.key(0), C, 4)
        x = jr.normal(jr.key(1), (C, T))
        perm = jr.permutation(jr.key(2), T)
        np.testing.assert_allclose(attn(x)[:, perm], attn(x[:, perm]), atol=1e-5)

    def test_rope_breaks_permutation_symmetry(self):
        attn = MultiHeadAttention(jr.key(0), C, 4, use_rope=True)
        x = jr.normal(jr.key(1), (C, T))
        perm = jr.permutation(jr.key(2), T)
        assert not np.allclose(attn(x)[:, perm], attn(x[:, perm]), atol=1e-4)
        assert isinstance(attn.rope, RotaryEmbedding)

    def test_dropout(self):
        attn = MultiHeadAttention(jr.key(0), C, 4, 2, dropout=0.5)
        x = jr.normal(jr.key(1), (C, T))
        clean = attn(x, inference=True)
        # inference ignores the key; without key and without inference it fails
        np.testing.assert_array_equal(clean, attn(x, key=jr.key(3), inference=True))
        with pytest.raises(ValueError, match="PRNG key"):
            attn(x)
        dropped = attn(x, key=jr.key(3))
        assert not np.allclose(clean, dropped)
        np.testing.assert_array_equal(dropped, attn(x, key=jr.key(3)))
        assert not np.allclose(dropped, attn(x, key=jr.key(4)))

    def test_dropout_path_without_dropout_matches_fused_path(self):
        attn = MultiHeadAttention(jr.key(0), C, 4, 2, use_rope=True)
        x = jr.normal(jr.key(1), (C, T))
        q = attn.rope(attn.q_proj(x).reshape(4, 4, T))
        k = attn.rope(attn.k_proj(x).reshape(2, 4, T))
        v = attn.v_proj(x).reshape(2, 4, T)
        explicit = attn.o_proj(
            attn._attention_with_dropout(q, k, v, 1e-12, jr.key(0)).reshape(C, T)
        )
        np.testing.assert_allclose(explicit, attn(x), rtol=1e-4, atol=1e-5)

    def test_vmap_and_grad(self):
        attn = MultiHeadAttention(jr.key(0), C, 4, 2, dropout=0.1, use_rope=True)
        xs = jr.normal(jr.key(1), (3, C, T))
        keys = jr.split(jr.key(2), 3)
        out = jax.vmap(lambda x, k: attn(x, key=k))(xs, keys)
        assert out.shape == (3, C, T)
        grads = eqx.filter_grad(lambda m: jnp.sum(m(xs[0], key=keys[0]) ** 2))(attn)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert bool(jnp.all(jnp.isfinite(leaf)))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"channels": 0},
            {"n_heads": 0},
            {"n_heads": 3},  # 16 % 3 != 0
            {"n_heads": 4, "n_kv_heads": 3},
            {"n_heads": 4, "dropout": 1.0},
            {"n_heads": 16, "use_rope": True},  # odd head dimension (1)
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        full = {"channels": C} | kwargs
        with pytest.raises(ValueError):
            MultiHeadAttention(jr.key(0), **full)
