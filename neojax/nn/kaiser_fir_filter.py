"""Implementation of the FIR filter (with Kaiser window) for the CNO."""

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jss
from jaxtyping import Array, Float


class KaiserFIRFilter(eqx.Module):
    r"""FIR Filter with Kaiser window.

    Computes

    $$
    h_K(x) = 2 f_c \cdot \text{sinc}(2 f_c x) \cdot w_K(x)
    $$

    where $w_K(x)$ is the window function, $f_c$ the cutoff frequency
    and $h_K(x)$ is the practical approximation of $\psi(x)$,
    the ideal continuous-time low-pass filter
    (as in appendix C of `StyleGAN3` and `scipy.signal.firwin`).

    Part of the filtered leaky ReLU of the `CNO`.

    Args:
        cutoff_freq: Cutoff frequency $f_c$ (same unit as `sampling_freq`).
            Has to be in the open interval `(0, sampling_freq / 2)`.
        num_taps: Discrete spatial locations, i.e. taps, where the filter
            is defined. Default is `12`, as in the `CNO` configuration.
        half_band_width: Half-width of the transition band, i.e. $f_h$. Same as
            `scipy.signal.firwin`'s `width / 2`. Default is `0.8`.
        sampling_fac: Up-/ and downsampling factor. Multiplication factor for
            `num_taps` and `sampling_freq`. Needed due to zero-padding or strided
            sampling. Default is `2`.
        sampling_freq: The sampling frequency or rate $s$/$f_s$. Default is 2.
        normalize: Whether to normalize the filter, i.e. enforce $\sum_i h_{K}\left[i\right] = 1$.
            Default is `True`.

    Raises:
        ValueError: If `cutoff_freq` is not in `(0, sampling_freq / 2)`.

    ??? info "Internal Attributes"
        These fields store the internal layer states.

        * **_filter** (`tuple[float, ...]`): The filter stencil.

    ??? cite
        [Alias-Free Generative Adversarial Networks](https://proceedings.
        neurips.cc/paper_files/paper/
        2021/file/076ccd93ad68be51f23707988e934906-Paper.pdf)

        ```bibtex
        @inproceedings{Karras2021,
            author = {Tero Karras and Miika Aittala and
            Samuli Laine and Erik H\"ark\"onen and
            Janne Hellsten and Jaakko Lehtinen and Timo Aila},
            title = {Alias-Free Generative Adversarial Networks},
            booktitle = {Proc. NeurIPS},
            year = {2021}
        }
        ```
    """

    _filter: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        cutoff_freq: float | int,
        num_taps: int = 12,
        half_band_width: float | int = 0.8,
        sampling_fac: int = 2,
        sampling_freq: float | int = 2.0,
        normalize: bool = True,
    ) -> None:
        if cutoff_freq <= 0 or cutoff_freq >= sampling_freq / 2:
            raise ValueError(
                "`cutoff_freq` has to be in the open interval `(0, sampling_freq / 2)`."
            )
        num_taps = num_taps * sampling_fac
        sampling_freq = sampling_freq * sampling_fac
        # x in the appendix are num_taps because the filter is only defined at those locations
        # they are symmetric and evenly spaced around zero
        # Define x as (i - (n - 1) / 2) / s for i in 1, ..., n - 1,
        # where s is the sampling rate and n the number of taps
        spatial_locs = jnp.array(
            [(i - (num_taps - 1) / 2) / sampling_freq for i in range(num_taps)]
        )
        spatial_extent = (num_taps - 1) / sampling_freq
        # Define Kaiser window
        band_width = (2 * half_band_width) / (sampling_freq / 2)
        # Attenuation
        max_atten = jnp.asarray(2.285 * (num_taps - 1) * jnp.pi * band_width + 7.95)
        beta = self._beta(max_atten)

        bessel_frac = jss.i0(
            beta * jnp.sqrt(1 - jnp.pow((2 * spatial_locs) / spatial_extent, 2))
        ) / jss.i0(beta)
        window = jnp.where(
            jnp.abs(spatial_locs) <= spatial_extent / 2, bessel_frac, 0.0
        )

        filter = (
            2 * cutoff_freq * jnp.sinc(2 * cutoff_freq * spatial_locs) * window
        ) / sampling_freq
        # Enforce sum_i(h_K[i]) = 1
        if normalize:
            filter = filter / jnp.sum(filter)

        self._filter = tuple(map(float, filter))

    def _beta(self, max_atten: Float[Array, ""]) -> Float[Array, ""]:
        """Compute the shape parameter beta.

        Args:
            max_atten: Maximum attenuation achievable.

        Returns:
            The shape parameter beta.
        """
        if max_atten > 50:
            return 0.1102 * (max_atten - 8.7)
        elif max_atten >= 21:
            return 0.5842 * jnp.pow(max_atten - 21, 0.4) + 0.07886 * (max_atten - 21)
        return jnp.zeros((), dtype=max_atten.dtype)

    def __getitem__(self, idx: int | slice) -> float | tuple[float, ...]:
        """Gets items from underlying filter."""
        return self._filter[idx]

    @property
    def filter(self) -> tuple[float, ...]:
        """Assembled static filter stencil tuple with shape (`num_taps`,)."""
        return self._filter
