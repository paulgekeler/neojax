import jax.random as jr
import numpy as np
import pytest

from neojax.nn.resample import Resampler


class TestResampler:
    def test_1d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 50))
        resampler = Resampler(input_shape=x.shape, res_scale=2, axes=1)
        out = resampler(x)
        assert out.shape == (5, 100)

    def test_2d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 50, 50))
        resampler = Resampler(input_shape=x.shape, res_scale=(2, 3), axes=(1, 2))
        out = resampler(x)
        assert out.shape == (5, 100, 150)

    def test_3d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 50, 50, 50))
        resampler = Resampler(input_shape=x.shape, res_scale=(2, 3, 2), axes=(1, 2, 3))
        out = resampler(x)
        assert out.shape == (5, 100, 150, 100)

    def test_4d(self):
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 50, 50, 50, 50))
        resampler = Resampler(
            input_shape=x.shape, res_scale=(2, 3, 2, 3), axes=(1, 2, 3, 4)
        )
        out = resampler(x)
        assert out.shape == (5, 100, 150, 100, 150)

    def test_params(self):
        with pytest.raises(ValueError):
            _ = Resampler(input_shape=(2, 10, 10), res_scale=(2, 2), axes=(1,))

        # verify output_shape doesnt change output
        key = jr.key(0)
        x = jr.normal(key=key, shape=(5, 50, 50))

        resampler_w_out_shape = Resampler(
            input_shape=(2, 50, 50),
            res_scale=(2, 2),
            axes=(1, 2),
            output_shape=(2, 100, 100),
        )

        resampler = Resampler(
            input_shape=(2, 50, 50),
            res_scale=(2, 2),
            axes=(1, 2),
        )
        # test ignore res_scale
        resampler_no_rscale = Resampler(
            input_shape=(2, 50, 50),
            res_scale=(10, 10),
            axes=(1, 2),
            output_shape=(2, 100, 100),
        )

        np.testing.assert_allclose(resampler_w_out_shape(x), resampler(x))
        np.testing.assert_allclose(resampler_w_out_shape(x), resampler_no_rscale(x))
