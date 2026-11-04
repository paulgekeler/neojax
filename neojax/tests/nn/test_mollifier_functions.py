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
from jaxtyping import TypeCheckError

from neojax.nn.mollifier_functions import make_weighting_fn
from neojax.tests.conftest import assert_filter_jittable


@pytest.mark.parametrize("name", ["bump", "half_cos", "quadr", "quartic", "octic"])
def test_mollifier_jittable(name):
    assert_filter_jittable(make_weighting_fn(name, 1.0), jnp.linspace(0.0, 2.0, 20))


def test_wrong_name_raises():
    with pytest.raises((ValueError, TypeCheckError)):
        _ = make_weighting_fn("invalid", sq_radius=1.0)
