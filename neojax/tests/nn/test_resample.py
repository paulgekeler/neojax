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

import jax.random as jr
import numpy as np
import pytest

from neojax.nn.resample import Resampler


class TestResampler:
    def test_1d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 5))
        resampler = Resampler(input_shape=x.shape, res_scale=2, axes=1)
        out = resampler(x)
        assert out.shape == (5, 10)

    def test_2d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 5, 5))
        resampler = Resampler(input_shape=x.shape, res_scale=(2, 3), axes=(1, 2))
        out = resampler(x)
        assert out.shape == (5, 10, 15)

    def test_3d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 5, 5, 5))
        resampler = Resampler(input_shape=x.shape, res_scale=(2, 3, 2), axes=(1, 2, 3))
        out = resampler(x)
        assert out.shape == (5, 10, 15, 10)

    def test_4d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 5, 5, 5, 5))
        resampler = Resampler(
            input_shape=x.shape, res_scale=(2, 3, 2, 3), axes=(1, 2, 3, 4)
        )
        out = resampler(x)
        assert out.shape == (5, 10, 15, 10, 15)

    def test_params(self):
        with pytest.raises(ValueError):
            _ = Resampler(input_shape=(2, 10, 10), res_scale=(2, 2), axes=(1,))

        # verify output_shape doesnt change output
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 5, 5))

        resampler_w_out_shape = Resampler(
            input_shape=(2, 5, 5),
            res_scale=(2, 2),
            axes=(1, 2),
            output_shape=(2, 10, 10),
        )

        resampler = Resampler(
            input_shape=(2, 5, 5),
            res_scale=(2, 2),
            axes=(1, 2),
        )
        # test ignore res_scale
        resampler_no_rscale = Resampler(
            input_shape=(2, 5, 5),
            res_scale=(10, 10),
            axes=(1, 2),
            output_shape=(2, 10, 10),
        )

        np.testing.assert_allclose(resampler_w_out_shape(x), resampler(x))
        np.testing.assert_allclose(resampler_w_out_shape(x), resampler_no_rscale(x))
