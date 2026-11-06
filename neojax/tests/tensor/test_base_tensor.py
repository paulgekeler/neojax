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
import jax.random as jr
import pytest

from neojax.tensor import BaseTensor, DenseTensor


class TestRealPairs:
    def test_round_trip(self):
        key_re, key_im = jr.split(jr.key(0))
        z = jr.normal(key_re, (3, 4)) + 1j * jr.normal(key_im, (3, 4))
        pairs = BaseTensor.to_real(z)
        assert pairs.shape == (3, 4, 2)
        assert not jnp.iscomplexobj(pairs)
        assert jnp.array_equal(BaseTensor.to_complex(pairs), z)

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (jnp.float16, jnp.complex64),
            (jnp.bfloat16, jnp.complex64),
            (jnp.float32, jnp.complex64),
        ],
    )
    def test_to_complex_dtype(self, dtype, expected):
        pairs = jnp.array([[0.5, -2.0]], dtype=dtype)
        z = BaseTensor.to_complex(pairs)
        assert z.dtype == expected
        assert z[0] == 0.5 - 2.0j

    def test_inherited_by_children(self):
        tensor = DenseTensor(jr.key(0), 2, 2, (4, 4))
        assert tensor.to_complex(tensor.weights[0]).shape == (2, 2, 4, 4)
        assert DenseTensor.to_real(jnp.ones(3, dtype=jnp.complex64)).shape == (3, 2)
