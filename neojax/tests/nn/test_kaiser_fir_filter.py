import numpy as np
import pytest
import scipy.signal as signal

from neojax.nn.kaiser_fir_filter import KaiserFIRFilter


@pytest.fixture
def filter_args():
    return {
        "cutoff_freq": 30,
        "num_taps": 20,
        "half_band_width": 10,
        "sampling_fac": 1,
        "sampling_freq": 200.0,
    }


class TestKaiserFIRFilter:
    @pytest.mark.parametrize(
        "overrides", [{}, {"num_taps": 24}, {"cutoff_freq": 60.0}, {"sampling_fac": 2}]
    )
    def test_matches_scipy(self, filter_args, overrides):
        args = {**filter_args, **overrides}
        kaiser_filter = KaiserFIRFilter(**args)
        # Manually scale firwin input with sampling_fac
        scipy_output = signal.firwin(
            numtaps=args["num_taps"] * args["sampling_fac"],
            cutoff=args["cutoff_freq"],
            width=args["half_band_width"] * 2,
            fs=args["sampling_freq"] * args["sampling_fac"],
        )
        np.testing.assert_allclose(kaiser_filter.filter, scipy_output, rtol=1e-5)

    def test_matches_scipy_low_attenuation(self):
        # Attenuation below 21 dB (beta = 0, i.e. a rectangular window)
        kaiser_filter = KaiserFIRFilter(
            cutoff_freq=30,
            num_taps=3,
            half_band_width=40,
            sampling_fac=1,
            sampling_freq=200.0,
        )
        scipy_output = signal.firwin(numtaps=3, cutoff=30, width=80, fs=200.0)
        np.testing.assert_allclose(kaiser_filter.filter, scipy_output, rtol=1e-5)

    @pytest.mark.parametrize("cutoff_freq", [0, 100, 150])
    def test_wrong_cutoff_raises(self, filter_args, cutoff_freq):
        with pytest.raises(ValueError):
            args = filter_args | {"cutoff_freq": cutoff_freq}
            _ = KaiserFIRFilter(**args)

    def test_getitem(self, filter_args):
        kf = KaiserFIRFilter(**filter_args)
        assert kf[::2] == kf.filter[::2]
        assert kf[0] == kf.filter[0]
        assert kf[3:] == kf.filter[3:]
