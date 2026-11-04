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

from neojax.nn.skip_connections import Flattened1dConv, SoftGating


class TestSoftGating:
    def test_dimensions(self):
        in_channels = 3
        # 2D
        gating_2d = SoftGating(ndim=2, in_channels=in_channels, use_bias=True)
        x_2d = jnp.ones((in_channels, 16, 16))
        assert gating_2d(x_2d).shape == (in_channels, 16, 16)

        # 3D
        gating_3d = SoftGating(ndim=3, in_channels=in_channels)
        x_3d = jnp.ones((in_channels, 8, 8, 8))
        assert gating_3d(x_3d).shape == (in_channels, 8, 8, 8)

        # 4D
        gating_4d = SoftGating(ndim=4, in_channels=in_channels)
        x_4d = jnp.ones((in_channels, 4, 4, 4, 4))
        assert gating_4d(x_4d).shape == (in_channels, 4, 4, 4, 4)

    def test_broadcasting(self):
        in_channels = 3
        gating = SoftGating(ndim=2, in_channels=in_channels)
        x_batch = jnp.ones((2, in_channels, 16, 16))
        assert gating(x_batch).shape == (2, in_channels, 16, 16)

    def test_invalid_channels(self):
        with pytest.raises(ValueError):
            SoftGating(ndim=2, in_channels=3, out_channels=4)


class TestFlattened1dConv:
    def test_dimensions(self):
        key = jr.key(seed=1)
        in_channels, out_channels = 3, 5
        conv = Flattened1dConv(
            key=key, in_channels=in_channels, out_channels=out_channels, kernel_size=1
        )

        # 2D
        assert conv(jnp.ones((in_channels, 16, 16))).shape == (out_channels, 16, 16)
        # 3D
        assert conv(jnp.ones((in_channels, 4, 4, 4))).shape == (out_channels, 4, 4, 4)
        # 4D
        assert conv(jnp.ones((in_channels, 4, 4, 4, 4))).shape == (
            out_channels,
            4,
            4,
            4,
            4,
        )
