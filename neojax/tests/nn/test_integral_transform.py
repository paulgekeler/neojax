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

from neojax.nn.integral_transform import IntegralTransform
from neojax.nn.mollifier_functions import make_weighting_fn
from neojax.nn.neighbor_attention import NeighborAttention
from neojax.nn.neighbor_search import radius_neighbor_search
from neojax.nn.pointwise_mlp import PointwiseMLP
from neojax.tests.conftest import assert_filter_jittable

D, N, M, F_C, RADIUS, CAPACITY = 2, 40, 15, 3, 0.4, 600


def edge_sq_dist(y, x, neighbors):
    query = jnp.minimum(neighbors.query_index, x.shape[0] - 1)
    return jnp.sum((x[query] - y[neighbors.data_index]) ** 2, axis=-1)


@pytest.fixture
def setup():
    k1, k2, k3, k4 = jr.split(jr.key(0), 4)
    y = jr.uniform(k1, (N, D))
    x = jr.uniform(k2, (M, D))
    f_y = jr.normal(k3, (F_C, N))
    neighbors = radius_neighbor_search(y, x, RADIUS, CAPACITY)
    return {
        "y": y,
        "x": x,
        "f_y": f_y,
        "neighbors": neighbors,
        "sq_dist": edge_sq_dist(y, x, neighbors),
        "key": k4,
    }


@pytest.fixture
def make_transform():
    def f(key, transform_type, out_c=F_C, **kwargs):
        kin = 2 * D + (F_C if transform_type.startswith("nonlinear") else 0)
        mlp = PointwiseMLP(key=key, layers=(kin, 8, out_c))
        return IntegralTransform(mlp, transform_type=transform_type, **kwargs)

    return f


@pytest.fixture
def manual_transform():
    def f(
        it, y, x, f_y, neighbors, weighting_fn=None, mean=False, quadrature_weights=None
    ):
        y, x, f_y = map(
            np.asarray, (y, x, f_y if f_y is not None else np.zeros((0, N)))
        )
        query, data = (
            np.asarray(neighbors.query_index),
            np.asarray(neighbors.data_index),
        )
        nonlinear = it.transform_type.startswith("nonlinear")
        out = None
        for i in range(x.shape[0]):
            acc = 0.0
            count = 0
            for edge in np.nonzero(query == i)[0]:
                yj = data[edge]
                feats = [y[yj], x[i]] + ([f_y[:, yj]] if nonlinear else [])
                k = np.asarray(
                    it.channel_mlp(jnp.array(np.concatenate(feats))[:, None])
                )[:, 0]
                if it.transform_type in ("linear", "nonlinear"):
                    k = k * f_y[:, yj]
                if quadrature_weights is not None:
                    k = k * float(quadrature_weights[yj])
                if weighting_fn is not None:
                    sq_dist = np.sum((x[i] - y[yj]) ** 2)
                    k = k * np.asarray(weighting_fn(jnp.array(sq_dist)))
                acc = acc + k
                count += 1
            acc = np.zeros(it.channel_mlp.weights[-1].shape[0]) if count == 0 else acc
            if mean and count > 0:
                acc = acc / count
            out = (
                acc[:, None]
                if out is None
                else np.concatenate([out, acc[:, None]], axis=1)
            )
        return out

    return f


@pytest.mark.usefixtures("make_transform", "setup", "manual_transform")
class TestIntegralTransform:
    @pytest.mark.parametrize(
        "transform_type",
        ["linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"],
    )
    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_matches_manual(
        self, setup, transform_type, reduction, make_transform, manual_transform
    ):
        it = make_transform(setup["key"], transform_type, reduction=reduction)
        out = it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        assert out.shape == (F_C, M)
        expected = manual_transform(
            it,
            setup["y"],
            setup["x"],
            setup["f_y"],
            setup["neighbors"],
            mean=reduction == "mean",
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        assert_filter_jittable(
            it, setup["y"], setup["x"], setup["neighbors"], setup["f_y"]
        )

    def test_kernel_output_channels_free_for_kernelonly(self, setup, make_transform):
        it = make_transform(setup["key"], "nonlinear_kernelonly", out_c=5)
        out = it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        assert out.shape == (5, M)

    def test_linear_without_f_is_kernel_only(self, setup):
        mlp = PointwiseMLP(key=setup["key"], layers=(2 * D, 8, 4))
        it = IntegralTransform(mlp, transform_type="linear")
        out = it(setup["y"], setup["x"], setup["neighbors"], None)
        assert out.shape == (4, M)

    def test_weighting_fn_forces_sum(self, setup, make_transform, manual_transform):
        weighting_fn = make_weighting_fn("half_cos", sq_radius=RADIUS**2)
        it = make_transform(
            setup["key"], "linear", reduction="mean", weighting_fn=weighting_fn
        )
        assert it.reduction == "sum"
        out = it(
            setup["y"],
            setup["x"],
            setup["neighbors"],
            setup["f_y"],
            sq_dist=setup["sq_dist"],
        )
        expected = manual_transform(
            it,
            setup["y"],
            setup["x"],
            setup["f_y"],
            setup["neighbors"],
            weighting_fn=weighting_fn,
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    def test_weighting_fn_requires_sq_dist(self, setup, make_transform):
        weighting_fn = make_weighting_fn("half_cos", sq_radius=RADIUS**2)
        it = make_transform(setup["key"], "linear", weighting_fn=weighting_fn)
        with pytest.raises(ValueError, match="sq_dist"):
            it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])

    @pytest.mark.parametrize("transform_type", ["linear", "nonlinear_kernelonly"])
    def test_quadrature_weights(
        self, setup, transform_type, make_transform, manual_transform
    ):
        it = make_transform(setup["key"], transform_type)
        args = (setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        delta = jr.uniform(jr.key(11), (N,)) + 0.5

        out = it(*args, quadrature_weights=delta)
        expected = manual_transform(
            it,
            setup["y"],
            setup["x"],
            setup["f_y"],
            setup["neighbors"],
            quadrature_weights=np.asarray(delta),
        )
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

        # a constant quadrature weight is a constant factor of the sum
        np.testing.assert_allclose(
            it(*args, quadrature_weights=jnp.full((N,), 0.25)),
            0.25 * it(*args),
            rtol=1e-5,
            atol=1e-6,
        )
        assert_filter_jittable(lambda m: m(*args, quadrature_weights=delta), it)

    @pytest.mark.parametrize("reduction", ["sum", "mean"])
    def test_empty_neighborhood_is_zero(self, setup, reduction, make_transform):
        it = make_transform(setup["key"], "linear", reduction=reduction)
        x_far = jnp.array([[10.0, 10.0]])
        neighbors = radius_neighbor_search(setup["y"], x_far, RADIUS, 8)
        out = it(setup["y"], x_far, neighbors, setup["f_y"])
        np.testing.assert_array_equal(out, jnp.zeros((F_C, 1)))

    @pytest.mark.parametrize("chunk", [1, 7, 64, CAPACITY - 1, CAPACITY, 2 * CAPACITY])
    def test_chunked_equals_unchunked(self, setup, chunk, make_transform):
        full = make_transform(setup["key"], "nonlinear", reduction="mean")
        chunked = IntegralTransform(
            full.channel_mlp, "nonlinear", "mean", edge_chunk_size=chunk
        )
        args = (setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        np.testing.assert_allclose(full(*args), chunked(*args), rtol=1e-5, atol=1e-6)

        def loss(model):
            return jnp.sum(model(*args) ** 2)

        g_full = eqx.filter_grad(loss)(full)
        g_chunked = eqx.filter_grad(loss)(chunked)
        for a, b in zip(
            jax.tree_util.tree_leaves(g_full),
            jax.tree_util.tree_leaves(g_chunked),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_vmap_over_batch_with_shared_geometry(self, setup, make_transform):
        it = make_transform(setup["key"], "nonlinear")
        f_batch = jr.normal(jr.key(7), (3, F_C, N))
        batched = jax.vmap(it, in_axes=(None, None, None, 0))(
            setup["y"], setup["x"], setup["neighbors"], f_batch
        )
        assert batched.shape == (3, F_C, M)
        for b in range(3):
            np.testing.assert_allclose(
                batched[b],
                it(setup["y"], setup["x"], setup["neighbors"], f_batch[b]),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_grad_step(self, setup, make_transform):
        it = make_transform(
            setup["key"],
            "nonlinear",
            weighting_fn=make_weighting_fn("bump", RADIUS**2),
        )

        def loss(args):
            model, f_y = args
            return jnp.sum(
                model(
                    setup["y"],
                    setup["x"],
                    setup["neighbors"],
                    f_y,
                    sq_dist=setup["sq_dist"],
                )
            )

        g_model, g_f = eqx.filter_grad(loss)((it, setup["f_y"]))
        for leaf in jax.tree_util.tree_leaves(g_model):
            assert bool(jnp.all(jnp.isfinite(leaf)))
            assert float(jnp.abs(leaf).sum()) > 0
        assert bool(jnp.all(jnp.isfinite(g_f)))

    def test_invalid_arguments(self, setup):
        mlp = PointwiseMLP(key=setup["key"], layers=(2 * D, 4, F_C))
        with pytest.raises(ValueError):
            IntegralTransform(mlp, edge_chunk_size=0)

    def test_missing_f_for_nonlinear_raises(self, setup, make_transform):
        it = make_transform(setup["key"], "nonlinear")
        with pytest.raises(ValueError):
            it(setup["y"], setup["x"], setup["neighbors"], None)

    def test_channel_mismatch_for_linear_raises(self, setup, make_transform):
        it = make_transform(setup["key"], "linear", out_c=F_C + 1)
        with pytest.raises(ValueError):
            it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])

    @pytest.mark.parametrize("attention_type", ["cosine", "dot_product"])
    def test_attention_matches_manual(self, setup, attention_type, make_transform):
        attention = NeighborAttention(jr.key(21), D, attention_type, attention_dim=6)
        it = make_transform(
            setup["key"], "linear", reduction="mean", attention=attention
        )
        assert it.reduction == "sum"
        out = it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])

        y, x, f_y = (np.asarray(setup[n]) for n in ("y", "x", "f_y"))
        query = np.asarray(setup["neighbors"].query_index)
        data = np.asarray(setup["neighbors"].data_index)
        expected = np.zeros((F_C, M))
        for i in range(M):
            js = data[query == i]
            if js.size == 0:
                continue
            if attention_type == "cosine":
                logits = np.array(
                    [
                        x[i] @ y[j] / (np.linalg.norm(x[i]) * np.linalg.norm(y[j]))
                        for j in js
                    ]
                )
            else:
                wq, bq = (
                    np.asarray(a[0])
                    for a in (attention.query_proj.weights, attention.query_proj.biases)
                )
                wk, bk = (
                    np.asarray(a[0])
                    for a in (attention.key_proj.weights, attention.key_proj.biases)
                )
                logits = np.array(
                    [((wq @ x[i] + bq) @ (wk @ y[j] + bk)) / np.sqrt(6) for j in js]
                )
            alpha = np.exp(logits - logits.max())
            alpha /= alpha.sum()
            for a, j in zip(alpha, js, strict=True):
                feats = jnp.array(np.concatenate([y[j], x[i]]))[:, None]
                k = np.asarray(it.channel_mlp(feats))[:, 0]
                expected[:, i] += a * k * f_y[:, j]
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)
        assert_filter_jittable(
            it, setup["y"], setup["x"], setup["neighbors"], setup["f_y"]
        )

    def test_attention_with_quadrature_and_weighting_fn(self, setup, make_transform):
        attention = NeighborAttention(jr.key(21), D)
        weighting_fn = make_weighting_fn("half_cos", sq_radius=RADIUS**2)
        it = make_transform(
            setup["key"], "linear", attention=attention, weighting_fn=weighting_fn
        )
        plain = make_transform(setup["key"], "linear", weighting_fn=weighting_fn)
        args = (setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        delta = jr.uniform(jr.key(11), (N,)) + 0.5
        kwargs = {"quadrature_weights": delta, "sq_dist": setup["sq_dist"]}
        out = it(*args, **kwargs)
        assert out.shape == (F_C, M)
        assert bool(jnp.all(jnp.isfinite(out)))
        # attention weights sum to one, so attention changes the result
        assert not np.allclose(out, plain(*args, **kwargs))

    @pytest.mark.parametrize("chunk", [1, 7, 100])
    def test_attention_chunked_equals_unchunked(self, setup, chunk, make_transform):
        # the softmax of a query spans several chunks
        attention = NeighborAttention(jr.key(21), D, "dot_product")
        full = make_transform(setup["key"], "nonlinear", attention=attention)
        chunked = IntegralTransform(
            full.channel_mlp,
            "nonlinear",
            edge_chunk_size=chunk,
            attention=attention,
        )
        args = (setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
        np.testing.assert_allclose(full(*args), chunked(*args), rtol=1e-5, atol=1e-6)

    def test_attention_empty_neighborhood_is_zero(self, setup, make_transform):
        attention = NeighborAttention(jr.key(21), D)
        it = make_transform(setup["key"], "linear", attention=attention)
        x_far = jnp.array([[10.0, 10.0]])
        neighbors = radius_neighbor_search(setup["y"], x_far, RADIUS, 8)
        out = it(setup["y"], x_far, neighbors, setup["f_y"])
        np.testing.assert_array_equal(out, jnp.zeros((F_C, 1)))

    def test_attention_channel_mismatch_raises(self, setup, make_transform):
        attention = NeighborAttention(jr.key(21), D + 1)
        it = make_transform(setup["key"], "linear", attention=attention)
        with pytest.raises(ValueError, match="attention"):
            it(setup["y"], setup["x"], setup["neighbors"], setup["f_y"])
