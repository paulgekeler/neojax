"""Implementation fo the filtered Leaky ReLU of the CNO (classic version)."""

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact

from neojax.nn.kaiser_fir_filter import KaiserFIRFilter


class FilteredLReLU(eqx.Module):
    r"""Filtered leaky ReLU block (n-dimensional).

    Computes the alias-free activation layer fused with the
    preceding/subsequent resampling operator of the CNO (does not
    include the preceding convolution layer).

    Which activation layer and resampling operator to compute can
    be controlled via `block_type`:

    - `"invariant"` (I-block/R-block): $u = 2, d = 2$. `s -> 2s -> lReLU -> s`
    - `"upsample"` (U-block): $u = 4, d = 2$. `s -> 4s -> lReLU -> 2s`
    - `"downsample"` (D-block): $u = 2, d = 4$. `s -> 2s -> lReLU -> s/2`

    Implements the CUDA logic as in the reference (classic CNO variant),
    i.e. as in StyleGAN3 in JAX. The filter taps are compile-time constants.

    Args:
        ndim: Number of spatial dimensions.
        channels: Number of channels of the input, needed for the bias.
        h_up: Kaiser windowed FIR filter used for upsampling. It needs
            `up * taps` taps, with `up` being the upsampling factor
            of the `block_type`, and `taps` any integer.
        h_down: Kaiser windowed FIR filter used for downsampling. It needs
            `down * taps` taps, with `down` being the downsampling factor
            of the `block_type`, and `taps` any integer.
        block_type: The type of activation layer + resampling operator
            block to compute. Must be one of `"invariant"`, `"upsample"`
            or `"downsample"`. See class docstring for details.
        use_bias: Whether to add a learnable channel-wise bias
            (initialized to zero) before the activation. Default is `True`.
        pad_mode: Padding mode for the up- and downsampling padding. Use `"wrap"`
            for a periodic domain, otherwise `"reflect"` or `"constant"`
            (zero padding as in the reference). Default is `"wrap"`.
        negative_slope: Negative slope for the leaky ReLU
            (multiplication factor for `x < 0`). Default is `0.2`.
        gain: Factor multiplying the output of the leaky ReLU.
            Default is $\sqrt{2}$, as in the reference.

    Raises:
        ValueError: If `block_type` or `pad_mode` are invalid.
        ValueError: If `negative_slope` is less than or equal to zero.
        ValueError: If the number of taps of `h_up` (`h_down`) is not
            a multiple of the up- (down-)sampling factor, or if `h_down`
            has an odd number of taps.

    ??? info "Internal Attributes"
        These fields store the internal layer states and weights.

        * **bias** (`Float[Array, "c"] | None`): The learnable channel-wise bias.
        * **h_up** (`KaiserFIRFilter`): The upsampling filter.
        * **h_down** (`KaiserFIRFilter`): The downsampling filter.
        * **resampling_factors** (`dict[str, tuple[int, int]]`): Dictionary mapping
            `block_type` to up- and downsampling factors.
        * **block_type** (`str`): The type of block.
        * **pad_mode** (`str`): The padding mode.
        * **spatial_axes** (`tuple[int, ...]`): The (negative) spatial axes.
        * **negative_slope** (`float`): The slope of the leaky ReLU.
        * **gain** (`float`): The gain after the leaky ReLU.

    Examples:
        ```python
        import jax.numpy as jnp
        from neojax.nn import FilteredLReLU, KaiserFIRFilter

        # Filters of an I-block, designed on a unit grid
        h_up = KaiserFIRFilter(1 / 2.0001, 6, 0.8 - 1 / 2.0001, 2, 1)
        h_down = KaiserFIRFilter(1 / 2.0001, 6, 0.8 - 1 / 2.0001, 2, 1)
        act = FilteredLReLU(
            ndim=2,
            channels=3,
            h_up=h_up,
            h_down=h_down,
            block_type="invariant",
        )
        x = jnp.ones((3, 32, 32))
        # out.shape = (3, 32, 32)
        out = act(x)
        ```
    """

    bias: Float[Array, "c"] | None
    h_up: KaiserFIRFilter
    h_down: KaiserFIRFilter
    resampling_factors: dict[str, tuple[int, int]] = eqx.field(static=True)
    block_type: str = eqx.field(static=True)
    pad_mode: str = eqx.field(static=True)
    spatial_axes: tuple[int, ...] = eqx.field(static=True)
    negative_slope: float = eqx.field(static=True)
    gain: float = eqx.field(static=True)

    def __init__(
        self,
        ndim: int,
        channels: int,
        h_up: KaiserFIRFilter,
        h_down: KaiserFIRFilter,
        block_type: Literal["invariant", "upsample", "downsample"],
        use_bias: bool = True,
        pad_mode: Literal["wrap", "reflect", "constant"] = "wrap",
        negative_slope: float = 0.2,
        gain: float = math.sqrt(2),
    ) -> None:
        self.resampling_factors = {
            "invariant": (2, 2),
            "upsample": (4, 2),
            "downsample": (2, 4),
        }
        self.spatial_axes = tuple(-i for i in range(ndim, 0, -1))
        if block_type not in self.resampling_factors:
            raise ValueError(
                "Invalid block_type. Must be one of ['invariant', 'upsample', 'downsample']"
            )
        up, down = self.resampling_factors[block_type]
        self.block_type = block_type
        if pad_mode not in ["wrap", "reflect", "constant"]:
            raise ValueError(
                "Invalid pad_mode. Must be one of ['wrap', 'reflect', 'constant']."
            )
        self.pad_mode = pad_mode
        if negative_slope <= 0:
            raise ValueError("negative_slope must be strictly greater than zero.")
        self.negative_slope = negative_slope
        if len(h_up.filter) % up != 0:
            raise ValueError(
                f"`h_up` needs a multiple of {up} taps for block_type '{block_type}', "
                f"but has {len(h_up.filter)} taps."
            )
        if len(h_down.filter) % down != 0 or len(h_down.filter) % 2 != 0:
            raise ValueError(
                f"`h_down` needs an even multiple of {down} taps for block_type "
                f"'{block_type}', but has {len(h_down.filter)} taps."
            )
        self.h_up = h_up
        self.h_down = h_down
        self.gain = gain
        self.bias = jnp.zeros((channels,)) if use_bias else None

    @property
    def up(self) -> int:
        """Upsampling factor of the activation rate w.r.t the input."""
        return self.resampling_factors[self.block_type][0]

    @property
    def down(self) -> int:
        """Downsampling factor of the activation rate w.r.t. the output."""
        return self.resampling_factors[self.block_type][1]

    def _upsample(self, x: Inexact[Array, "..."], axis: int) -> Inexact[Array, "..."]:
        """Upsamples along one axis (zero insertion + FIR filter), polyphase.

        The result is not cropped to `up * size`, but extended by the
        samples the subsequent downsampling filter needs
        (`len(h_down) - down` more samples), such that the downsampling
        can be a plain strided "valid" convolution. This reproduces the padding
        of the reference.

        Args:
            x: Signal to upsample.
            axis: Axis along which to upsample the signal.

        Returns:
            The upsampled signal, with extent `up * size + len(h_down) - down`
            along the given axis. Includes the factor `up` of the reference.
        """
        up, down = self.up, self.down
        taps_up, taps_down = len(self.h_up.filter), len(self.h_down.filter)
        taps_per_phase = taps_up // up
        x = jnp.moveaxis(x, axis, -1)
        size = x.shape[-1]
        # Padding in the upsampled domain (as in the reference), i.e.
        # y[m] = sum_k h[k] * z[m + taps_up - 1 - pad_lo - k], z = zero-inserted x
        pad_total = taps_up + taps_down - down - 1
        pad_lo = (pad_total + up) // 2
        out_size = up * size + taps_down - down
        # Only z[up * n] = up * x[n] is nonzero. For s = m + offset, write
        # s = up * q + r. Then y[m] = up * sum_i h[r + up * i] * x[q - i]
        offset = taps_up - 1 - pad_lo
        # Compute whole groups of `up` phases, crop afterwards
        s_first = (offset // up) * up
        s_last = -((-(offset + out_size)) // up) * up
        q_first, q_last = s_first // up, s_last // up
        # Samples of x outside [0, size) are padding
        pad_before = max(0, taps_per_phase - 1 - q_first)
        pad_after = max(0, q_last - size)

        x_pad = jnp.pad(
            x, [(0, 0)] * (x.ndim - 1) + [(pad_before, pad_after)], self.pad_mode
        )
        phases = []
        for r in range(up):
            phase = sum(
                (up * self.h_up[r + up * i])
                * x_pad[..., q_first - i + pad_before : q_last - i + pad_before]
                for i in range(taps_per_phase)
            )
            phases.append(phase)
        y = jnp.stack(phases, axis=-1).reshape(*x.shape[:-1], up * (q_last - q_first))
        crop = offset - s_first

        return jnp.moveaxis(y[..., crop : crop + out_size], -1, axis)

    def _downsample(self, x: Inexact[Array, "..."], axis: int) -> Inexact[Array, "..."]:
        """Filters and downsamples along one axis ("valid" convolution).

        Args:
            x: Signal to downsample, extended as returned by `_upsample`.
            axis: Axis along which to downsample the signal.

        Returns:
            The downsampled signal, with extent `(extent - len(h_down)) // down + 1`.
        """
        down = self.down
        taps = len(self.h_down.filter)
        x = jnp.moveaxis(x, axis, -1)
        out_size = (x.shape[-1] - taps) // down + 1
        # The Kaiser filter is symmetric: group tap k and taps - 1 - k
        # to halve the multiplications
        y = sum(
            self.h_down[k]
            * (
                x[..., k : k + down * out_size : down]
                + x[..., taps - 1 - k : taps - 1 - k + down * out_size : down]
            )
            for k in range(taps // 2)
        )
        return jnp.moveaxis(y, -1, axis)

    def __call__(
        self,
        x: Inexact[Array, "c ..."],
    ) -> Inexact[Array, "c ..."]:
        """Forward pass through the fused sampling and LReLU block.

        Args:
            x: Input signal shaped (c, d1, ..., dN).

        Returns:
            The up- or downsampled signal with applied LReLU activation,
            shaped (c, d1', ..., dN') with `d_i' = d_i * up / down`.

        Raises:
            ValueError: If `up * d_i` is not divisible by `down` for a spatial size `d_i`.
        """
        for ax in self.spatial_axes:
            if (self.up * x.shape[ax]) % self.down != 0:
                raise ValueError(
                    f"Spatial size {x.shape[ax]} along axis {ax} is not compatible "
                    f"with block_type '{self.block_type}' "
                    f"({self.up}x up, {self.down}x down)."
                )

        if self.bias is not None:
            x = x + self.bias.reshape(-1, *([1] * len(self.spatial_axes)))

        for ax in self.spatial_axes:
            x = self._upsample(x, axis=ax)

        x = jax.nn.leaky_relu(x, negative_slope=self.negative_slope) * self.gain

        for ax in self.spatial_axes:
            x = self._downsample(x, axis=ax)
        return x
