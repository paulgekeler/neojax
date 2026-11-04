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

from neojax.nn.uvit import UViT
from neojax.tests.conftest import assert_filter_jittable

T = 9


def make(**kwargs):
    defaults = dict(
        key=jr.key(0),
        in_channels=8,
        out_channels=8,
        hidden_channels=16,
        n_layers=3,
        n_heads=4,
    )
    return UViT(**(defaults | kwargs))


class TestUViT:
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4, 5])
    def test_block_layout(self, n_layers):
        model = make(n_layers=n_layers)
        assert len(model.encoder_blocks) == n_layers // 2
        assert len(model.decoder_blocks) == n_layers // 2
        assert (model.middle_block is not None) == (n_layers % 2 == 1)
        assert all(b.skip_proj is None for b in model.encoder_blocks)
        assert all(b.skip_proj is not None for b in model.decoder_blocks)
        x = jr.normal(jr.key(1), (8, T))
        assert model(x).shape == (8, T)

    @pytest.mark.parametrize(
        "in_c, out_c, hidden", [(8, 8, 16), (16, 16, 16), (16, 5, 16), (4, 7, 16)]
    )
    def test_projections_only_when_widths_differ(self, in_c, out_c, hidden):
        model = make(in_channels=in_c, out_channels=out_c, hidden_channels=hidden)
        assert (model.input_proj is None) == (in_c == hidden)
        assert (model.output_proj is None) == (out_c == hidden)
        x = jr.normal(jr.key(1), (in_c, T))
        assert model(x).shape == (out_c, T)
        assert_filter_jittable(model, x)

    def test_forward_matches_manual_u_structure(self):
        model = make(n_layers=5)
        x = jr.normal(jr.key(1), (8, T))
        h = model.input_proj(x)
        skips = []
        for block in model.encoder_blocks:
            h = block(h)
            skips.append(h)
        h = model.middle_block(h)
        for block in model.decoder_blocks:
            h = block(h, skips.pop())
        expected = model.output_proj(h)
        np.testing.assert_allclose(model(x), expected, rtol=1e-5, atol=1e-6)

    def test_without_long_range_skip(self):
        model = make(n_layers=4, use_long_range_skip=False)
        assert all(b.skip_proj is None for b in model.decoder_blocks)
        x = jr.normal(jr.key(1), (8, T))
        h = model.input_proj(x)
        for block in (*model.encoder_blocks, *model.decoder_blocks):
            h = block(h)
        np.testing.assert_allclose(model(x), model.output_proj(h), rtol=1e-5, atol=1e-6)
        # skip connections change the function
        skipped = make(n_layers=4)
        assert not np.allclose(model(x), skipped(x))

    def test_dropout_keys(self):
        model = make(n_layers=3, dropout=0.4, use_rope=True)
        x = jr.normal(jr.key(1), (8, T))
        with pytest.raises(ValueError, match="PRNG key"):
            model(x)
        clean = model(x, inference=True)
        np.testing.assert_array_equal(clean, model(x, key=jr.key(2), inference=True))
        dropped = model(x, key=jr.key(2))
        assert not np.allclose(clean, dropped)
        np.testing.assert_array_equal(dropped, model(x, key=jr.key(2)))

    def test_vmap_and_grad(self):
        model = make()
        xs = jr.normal(jr.key(1), (3, 8, T))
        assert jax.vmap(model)(xs).shape == (3, 8, T)
        grads = eqx.filter_grad(lambda m: jnp.sum(m(xs[0]) ** 2))(model)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert bool(jnp.all(jnp.isfinite(leaf)))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"in_channels": 0},
            {"out_channels": 0},
            {"hidden_channels": 0},
            {"n_layers": 0},
            {"n_heads": 3},
        ],
    )
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            make(**kwargs)
