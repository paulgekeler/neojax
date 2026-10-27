import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from neojax.nn.filtered_lrelu import FilteredLReLU
from neojax.nn.kaiser_fir_filter import KaiserFIRFilter
from neojax.tests.conftest import assert_filter_jittable

# (up, down, out_size / in_size)
RESAMPLING = {
    "invariant": (2, 2, 1),
    "upsample": (4, 2, 2),
    "downsample": (2, 4, 0.5),
}


def design_filters(block_type, cutoff_den=2.0001, half_width_mult=0.8, filter_size=6):
    """Filters of the CNO reference, designed on a unit input grid."""
    rate_in = 1.0
    rate_out = {"invariant": 1.0, "upsample": 2.0, "downsample": 0.5}[block_type]
    rate_act = 2 * max(rate_in, rate_out)
    up, down = round(rate_act / rate_in), round(rate_act / rate_out)
    h_up = KaiserFIRFilter(
        rate_in / cutoff_den,
        filter_size,
        half_width_mult * rate_in - rate_in / cutoff_den,
        up,
        rate_in,
    )
    h_down = KaiserFIRFilter(
        rate_out / cutoff_den,
        filter_size,
        half_width_mult * rate_out - rate_out / cutoff_den,
        down,
        rate_out,
    )
    return h_up, h_down


def make_flrelu(ndim, block_type, channels=1, **kwargs):
    h_up, h_down = design_filters(block_type)
    return FilteredLReLU(ndim, channels, h_up, h_down, block_type, **kwargs)


def numpy_reference(x, flrelu, bias=None):
    """Direct implementation of the definition (zero padding, along each axis)."""
    up, down = flrelu.up, flrelu.down
    h_up, h_down = np.array(flrelu.h_up.filter), np.array(flrelu.h_down.filter)
    pad_total = len(h_up) + len(h_down) - down - 1
    pad_lo = (pad_total + up) // 2
    pad_hi = pad_total - pad_lo

    def up_axis(v):
        z = np.zeros(up * len(v))
        z[::up] = up * v
        return np.convolve(np.pad(z, (pad_lo, pad_hi)), h_up, mode="valid")

    def down_axis(v):
        return np.convolve(v, h_down, mode="valid")[::down]

    x = np.asarray(x, dtype=np.float64)
    if bias is not None:
        x = x + np.asarray(bias).reshape(-1, *([1] * (x.ndim - 1)))
    for ax in range(1, x.ndim):
        x = np.apply_along_axis(up_axis, ax, x)
    x = np.where(x >= 0, x, flrelu.negative_slope * x) * flrelu.gain
    for ax in range(1, x.ndim):
        x = np.apply_along_axis(down_axis, ax, x)
    return x


BLOCK_TYPES = ["invariant", "upsample", "downsample"]


class TestFilteredLReLU:
    def test_invalid_arguments_raise(self):
        h_up, h_down = design_filters("invariant")
        args = {
            "ndim": 2,
            "channels": 1,
            "h_up": h_up,
            "h_down": h_down,
            "block_type": "invariant",
        }
        for invalid in [
            {"pad_mode": "invalid"},
            {"block_type": "invalid"},
            {"negative_slope": 0.0},
            {"negative_slope": -1.0},
        ]:
            with pytest.raises((ValueError, TypeCheckError)):
                _ = FilteredLReLU(**args | invalid)
        # 6 taps are no multiple of the 4x upsampling factor of U-blocks
        six_taps = KaiserFIRFilter(0.4, 3, 0.2, 2, 1)
        with pytest.raises(ValueError):
            _ = FilteredLReLU(**args | {"block_type": "upsample", "h_up": six_taps})
        # 3 taps are no multiple of the 2x downsampling factor
        three_taps = KaiserFIRFilter(0.4, 3, 0.2, 1, 1)
        with pytest.raises(ValueError):
            _ = FilteredLReLU(**args | {"h_down": three_taps})
        # D-block halves the resolution, odd sizes are not possible
        with pytest.raises(ValueError):
            make_flrelu(2, "downsample")(jnp.ones((1, 15, 16)))

    @pytest.mark.parametrize("block_type", BLOCK_TYPES)
    def test_shapes(self, block_type):
        _, _, ratio = RESAMPLING[block_type]
        for pad_mode in ["wrap", "reflect", "constant"]:
            flrelu = make_flrelu(2, block_type, pad_mode=pad_mode)
            y = flrelu(jr.normal(jr.key(0), (1, 8, 8)))
            assert y.shape == (1, int(8 * ratio), int(8 * ratio))

    @pytest.mark.parametrize(
        ("ndim", "block_type"),
        [(1, "invariant"), (2, "upsample"), (2, "downsample")],
    )
    def test_matches_definition(self, ndim, block_type):
        flrelu = make_flrelu(ndim, block_type, channels=2, pad_mode="constant")
        bias = jr.normal(jr.key(1), (2,))
        flrelu = eqx.tree_at(lambda f: f.bias, flrelu, bias)
        x = jr.normal(jr.key(0), (2,) + (12,) * ndim)
        np.testing.assert_allclose(
            flrelu(x), numpy_reference(x, flrelu, bias), rtol=1e-4, atol=1e-4
        )

    def test_wrap_is_periodic_equivariant(self):
        # Shifting the input by 2 samples shifts the output by 2 * ratio samples
        x = jr.normal(jr.key(0), (1, 16))
        for block_type in BLOCK_TYPES:
            flrelu = make_flrelu(1, block_type, pad_mode="wrap")
            _, _, ratio = RESAMPLING[block_type]
            np.testing.assert_allclose(
                flrelu(jnp.roll(x, 2, axis=1)),
                jnp.roll(flrelu(x), int(2 * ratio), axis=1),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_bias(self):
        assert make_flrelu(2, "invariant", use_bias=False).bias is None
        flrelu = make_flrelu(2, "invariant", channels=2)
        x = jr.normal(jr.key(0), (2, 8, 8))
        grads = eqx.filter_grad(lambda f: jnp.sum(f(x) ** 2))(flrelu)
        assert jnp.any(grads.bias != 0)
        assert jnp.all(jnp.isfinite(grads.bias))

    def test_filters_are_compiled_constants_and_jittable(self):
        # Filter taps must be embedded in the graph as literals,
        # i.e. no array constants/inputs for the filters
        flrelu = make_flrelu(3, "downsample", use_bias=False)
        jaxpr = jax.make_jaxpr(flrelu)(jnp.ones((1, 8, 8, 8)))
        assert len(jaxpr.consts) == 0
        assert len(jaxpr.in_avals) == 1
        assert_filter_jittable(flrelu, jr.normal(jr.key(0), (1, 8, 8, 8)))
