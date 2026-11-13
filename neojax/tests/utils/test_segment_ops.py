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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.jet import jet

import neojax.utils._jet_rules  # noqa: F401
from neojax.tests.conftest import assert_filter_jittable
from neojax.utils.segment_ops import segment_max, segment_mean, segment_softmax

# segments: 0 has three edges, 1 is empty, 2 has one edge, the rest is padding
IDS = jnp.array([0, 0, 0, 2, 3, 3])
NUM = 3
VALUES = jnp.array([1.0, 3.0, 2.0, 5.0, 100.0, -100.0])


@pytest.mark.parametrize("is_sorted", [False, True])
class TestSegmentOps:
    def test_mean(self, is_sorted):
        out = segment_mean(VALUES, IDS, NUM, indices_are_sorted=is_sorted)
        np.testing.assert_allclose(out, [2.0, 0.0, 5.0])

    def test_max(self, is_sorted):
        out = segment_max(VALUES, IDS, NUM, indices_are_sorted=is_sorted)
        np.testing.assert_allclose(out, [3.0, 0.0, 5.0])

    def test_max_of_negative_values(self, is_sorted):
        out = segment_max(-VALUES, IDS, NUM, indices_are_sorted=is_sorted)
        np.testing.assert_allclose(out, [-1.0, 0.0, -5.0])

    def test_softmax(self, is_sorted):
        out = segment_softmax(VALUES, IDS, NUM, indices_are_sorted=is_sorted)
        expected = np.exp([1.0, 3.0, 2.0]) / np.exp([1.0, 3.0, 2.0]).sum()
        np.testing.assert_allclose(out[:3], expected, rtol=1e-6)
        np.testing.assert_allclose(out[3:], [1.0, 0.0, 0.0])

    def test_trailing_axes_are_elementwise(self, is_sorted):
        data = jnp.stack([VALUES, 2 * VALUES], axis=-1)
        kwargs = {"indices_are_sorted": is_sorted}
        for fn in (segment_mean, segment_max):
            out = fn(data, IDS, NUM, **kwargs)
            np.testing.assert_allclose(out[:, 1], 2 * fn(VALUES, IDS, NUM, **kwargs))
        out = segment_softmax(data, IDS, NUM, **kwargs)
        np.testing.assert_allclose(jnp.sum(out[:3], axis=0), [1.0, 1.0], rtol=1e-6)


class TestDifferentiation:
    def test_softmax_ignores_padding(self):
        grad = jax.grad(lambda v: segment_softmax(v, IDS, NUM)[0])(VALUES)
        assert np.all(np.isfinite(grad))
        np.testing.assert_array_equal(grad[3:], 0.0)

    def test_softmax_survives_extreme_values(self):
        out = segment_softmax(jnp.array([1e4, 0.0, 3.0]), jnp.array([0, 0, 1]), 2)
        np.testing.assert_allclose(out, [1.0, 0.0, 1.0])

    def test_softmax_keeps_dtype(self):
        logits = jnp.array([1.0, 2.0], dtype=jnp.float16)
        assert segment_softmax(logits, jnp.array([0, 0]), 1).dtype == jnp.float16

    def test_max_gradient_of_ties_matches_jnp_max(self):
        values = jnp.array([1.0, 3.0, 3.0])
        ids = jnp.zeros(3, jnp.int32)
        grad = jax.grad(lambda v: segment_max(v, ids, 1)[0])(values)
        np.testing.assert_allclose(grad, jax.grad(jnp.max)(values))

    @pytest.mark.parametrize(
        "fn",
        [segment_mean, segment_max, segment_softmax],
        ids=["mean", "max", "softmax"],
    )
    def test_jet_matches_nested_derivatives(self, fn):
        def f(t):
            return fn(VALUES * t**2, IDS, NUM, indices_are_sorted=True)

        t = jnp.array(1.3)
        primal, (first, second) = jet(f, (t,), ((jnp.ones(()), jnp.zeros(())),))
        np.testing.assert_allclose(primal, f(t), rtol=1e-5)
        np.testing.assert_allclose(first, jax.jacfwd(f)(t), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            second, jax.jacfwd(jax.jacfwd(f))(t), rtol=1e-3, atol=1e-4
        )


@pytest.mark.parametrize(
    "fn", [segment_mean, segment_max, segment_softmax], ids=["mean", "max", "softmax"]
)
def test_jittable(fn):
    assert_filter_jittable(fn, VALUES, IDS, NUM, indices_are_sorted=True)
