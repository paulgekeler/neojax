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

"""Utility for registering custom jet rules.

This module adds all the outstanding primitive rules needed by neojax
to `jax.experimental.jet.jet_rules` which is accessible.
It acts as a temporary patch for current JAX versions.

The module is imported in `neojax.metrics.__init__`. No manual work has to be done.

Future JAX versions may resolve this
once PR https://github.com/jax-ml/jax/pull/41176
is merged. The respective issue is https://github.com/jax-ml/jax/issues/41041.

See https://github.com/paulgekeler/neojax/issues/23 and its
sub-issues for details.
"""

import functools as ft

import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import jet

# Zero propagation primitive
if lax.le_to_p not in jet.jet_rules:
    jet.defzero(lax.le_to_p)
if lax.lt_to_p not in jet.jet_rules:
    jet.defzero(lax.lt_to_p)
if lax.eq_to_p not in jet.jet_rules:
    jet.defzero(lax.eq_to_p)
if lax.argmin_p not in jet.jet_rules:
    jet.defzero(lax.argmin_p)
if lax.argmax_p not in jet.jet_rules:
    jet.defzero(lax.argmax_p)
if lax.iota_p not in jet.jet_rules:
    jet.defzero(lax.iota_p)
if lax.clz_p not in jet.jet_rules:
    jet.defzero(lax.clz_p)
if lax.population_count_p not in jet.jet_rules:
    jet.defzero(lax.population_count_p)
if hasattr(lax, "mulhi_p") and lax.mulhi_p not in jet.jet_rules:
    jet.defzero(lax.mulhi_p)
if lax.reduce_and_p not in jet.jet_rules:
    jet.defzero(lax.reduce_and_p)
if lax.reduce_or_p not in jet.jet_rules:
    jet.defzero(lax.reduce_or_p)
if lax.reduce_xor_p not in jet.jet_rules:
    jet.defzero(lax.reduce_xor_p)

# Linear propagation primitives
if lax.pmax_p not in jet.jet_rules:
    jet.deflinear(lax.pmax_p)
if lax.pmin_p not in jet.jet_rules:
    jet.deflinear(lax.pmin_p)
if lax.psum_p not in jet.jet_rules:
    jet.deflinear(lax.psum_p)
if lax.tile_p not in jet.jet_rules:
    jet.deflinear(lax.tile_p)
if lax.all_to_all_p not in jet.jet_rules:
    jet.deflinear(lax.all_to_all_p)
if lax.all_gather_p not in jet.jet_rules:
    jet.deflinear(lax.all_gather_p)
if lax.ppermute_p not in jet.jet_rules:
    jet.deflinear(lax.ppermute_p)
if lax.axis_index_p not in jet.jet_rules:
    jet.deflinear(lax.axis_index_p)
if lax.ragged_all_to_all_p not in jet.jet_rules:
    jet.deflinear(lax.ragged_all_to_all_p)
if lax.reduce_precision_p not in jet.jet_rules:
    jet.deflinear(lax.reduce_precision_p)

# Basic composition primitives
if lax.cbrt_p not in jet.jet_rules:
    jet.def_comp(lax.cbrt_p, lambda x: x ** (1.0 / 3.0))
if lax.acos_p not in jet.jet_rules:
    jet.def_comp(lax.acos_p, lambda x: lax.sub(jnp.pi / 2.0, lax.asin(x)))
if lax.atan_p not in jet.jet_rules:
    jet.def_comp(lax.atan_p, lambda x: lax.atan2(x, lax.full_like(x, 1.0)))
if lax.tan_p not in jet.jet_rules:
    jet.def_comp(lax.tan_p, lambda x: lax.div(lax.sin(x), lax.cos(x)))
if lax.asin_p not in jet.jet_rules:
    jet.def_comp(
        lax.asin_p, lambda x: lax.atan2(x, lax.sqrt(lax.sub(1.0, lax.square(x))))
    )


# Proper rules
def _scatter_rule(
    primals_in,
    series_in,
    *,
    update_jaxpr,
    update_consts,
    dimension_numbers,
    indices_are_sorted,
    unique_indices,
    mode,
):
    bind = ft.partial(
        lax.scatter_p.bind,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
    )
    operand, scatter_indices, updates = primals_in
    primal_out = bind(operand, scatter_indices, updates)
    series_out = [
        bind(d1, scatter_indices, d2) for d1, _, d2 in zip(*series_in, strict=True)
    ]
    return primal_out, series_out


if lax.scatter_p not in jet.jet_rules:
    jet.jet_rules[lax.scatter_p] = _scatter_rule


def _approx_top_k_rule(primals_in, series_in, **params):
    (primals_in,) = primals_in
    (series_in,) = series_in

    val_out, idxs_out = lax.approx_top_k_p.bind(primals_in, **params)
    if series_in is jet.zero_series:
        val_terms = jet.zero_series
    else:
        val_terms = [
            jnp.take_along_axis(t, idxs_out, axis=params.get("reduction_dimension", -1))
            if t is not jet.zero_term
            else jet.zero_term
            for t in series_in
        ]

    return (val_out, idxs_out), (val_terms, jet.zero_series)


if lax.approx_top_k_p not in jet.jet_rules:
    jet.jet_rules[lax.approx_top_k_p] = _approx_top_k_rule


def _top_k_rule(primals_in, series_in, **params):
    (primals_in,) = primals_in
    (series_in,) = series_in

    val_out, idxs_out = lax.top_k_p.bind(primals_in, **params)
    if series_in is jet.zero_series:
        val_terms = jet.zero_series
    else:
        val_terms = [
            jnp.take_along_axis(t, idxs_out, axis=params.get("axis", -1))
            if t is not jet.zero_term
            else jet.zero_term
            for t in series_in
        ]

    return (val_out, idxs_out), (val_terms, jet.zero_series)


if lax.top_k_p not in jet.jet_rules:
    jet.jet_rules[lax.top_k_p] = _top_k_rule


def _sort_rule(primals_in, series_in, *, dimension, is_stable, num_keys):
    shape = primals_in[0].shape
    iota = lax.broadcasted_iota(jnp.int32, shape, dimension)
    *primals_out, perm = lax.sort_p.bind(
        *primals_in,
        iota,
        dimension=dimension,
        is_stable=is_stable,
        num_keys=num_keys,
    )
    series_out = [
        [jnp.take_along_axis(t, perm, axis=dimension) for t in terms]
        for terms in series_in
    ]
    return tuple(primals_out), series_out


if lax.sort_p not in jet.jet_rules:
    jet.jet_rules[lax.sort_p] = _sort_rule


def _nextafter_rule(primals_in, series_in, **params):
    x, y = primals_in
    x_terms, _ = series_in
    primal_out = lax.nextafter_p.bind(x, y, **params)
    series_out = [jnp.broadcast_to(t, primal_out.shape) for t in x_terms]
    return primal_out, series_out


if lax.nextafter_p not in jet.jet_rules:
    jet.jet_rules[lax.nextafter_p] = _nextafter_rule
