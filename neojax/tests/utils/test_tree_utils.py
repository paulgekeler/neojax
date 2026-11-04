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

import jax.numpy as jnp
import pytest

from neojax.utils import tree_add_broadcast_to


@pytest.fixture
def pytree1():
    return {"one": jnp.ones((4, 2, 1, 3)), "two": jnp.ones((4, 2, 3)) * 2.0}


def get_expected_shape(shape, axis, size):
    ndim = len(shape)
    # axis exists and has the requested size -> unchanged
    if -ndim <= axis < ndim and shape[axis] == size:
        return shape
    elif -ndim <= axis < ndim and shape[axis] == 1:
        if axis == -1:
            return shape[:axis] + (size,)
        return shape[:axis] + (size,) + shape[axis + 1 :]
    # otherwise insert new axis of size
    axis = axis % (ndim + 1)
    return shape[:axis] + (size,) + shape[axis:]


@pytest.mark.parametrize(["axis", "size"], [(0, 1), (-1, 1), (2, 3), (3, 5)])
def test_tree_add_broadcast_to(pytree1, axis, size):
    out = tree_add_broadcast_to(pytree1, axis=axis, size=size)
    for key in pytree1:
        assert out[key].shape == get_expected_shape(pytree1[key].shape, axis, size)
    # values remain unchanged
    assert jnp.all(out["one"] == 1.0)
    assert jnp.all(out["two"] == 2.0)
