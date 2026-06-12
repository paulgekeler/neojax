"""Implementation of the domain padding."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class DomainPadding(eqx.Module):
    """Applies domain padding to a signal.

    Domain Padding helps to mitigate boundary artifacts.
    In the context of FNO, it is often used
    to handle non-periodic boundary conditions
    by padding the spatial dimensions before
    the spectral convolution and cropping the result
    back to the original resolution.
    Padding is applied symmetrically to each dimension,
    i.e., a 20% (0.2) padding pads an input by 10%
    on each side along an axis.

    Args:
        padding: Percentage of padding to apply
            to each spatial dimension. Can be a single float in [0, 1]
            (applied to all dims) or a sequence of floats.
        mode: The type of padding to apply (e.g. "constant", "edge").
            Defaults to "constant" which pads with zeros.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html#jax.numpy.pad
            for all padding modes.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **padding** (`float | Sequence[float]`): The stored padding ratios for each dimension.
        * **mode** (`str`): The padding mode.

    !!! info
        Currently doesn't support resolution scaling.
    """

    padding: float | Sequence[float] = eqx.field(static=True)  # static -> jittable
    mode: str = eqx.field(static=True, default="constant")

    def __init__(
        self,
        padding: float | Sequence[float],
        mode: str = "constant",
    ) -> None:
        """Initializes the DomainPadding module."""
        self.padding = padding
        if mode not in ("constant", "edge", "wrap", "maximum", "minimum"):
            raise ValueError(
                "Padding mode unavailable."
                " See jax.numpy.pad for padding modes without kwargs."
            )
        self.mode = mode

    def _get_pad_widths(self, in_shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """Computes the symmetric padding margins per dim.

        Args:
            in_shape: Shape of the array to pad.

        Returns:
            Pad widths ((before_1, after_1), ..., (before_N, after_N))
            as expected by `jax.numpy.pad`.

        Raises:
            ValueError: If padding length doesn't match array ndims.
        """
        # loose channel dim
        dim_shape = in_shape[1:]
        ndim = len(dim_shape)
        if isinstance(self.padding, float):
            paddings = (self.padding,) * ndim
        elif len(self.padding) != ndim:
            raise ValueError(
                f"Mismatch of padding length: {len(self.padding)} and ndim: {ndim}"
            )
        else:
            paddings = self.padding
        pad_widths = [(0, 0)]
        for p, dim in zip(paddings, dim_shape, strict=True):
            pw = round((p / 2) * dim)
            pad_widths.append((pw, pw))
        return tuple(pad_widths)

    def pad(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Pads the input array based on the configured ratios.

        Args:
            x: The input array of shape `(channels, d1, ..., dn)`.

        Returns:
            The padded array.
        """
        pad_widths = self._get_pad_widths(x.shape)
        return jnp.pad(
            x,
            pad_width=pad_widths,
            mode=self.mode,
        )

    def unpad(
        self, x: Float[Array, "c ..."], original_shape: tuple[int, ...]
    ) -> Float[Array, "c ..."]:
        """Crops the padded array back to its original resolution.

        Args:
            x: The padded array.
            original_shape: The original array shape.

        Returns:
            The cropped array of original resolution.
        """
        pad_widths = self._get_pad_widths(original_shape)
        slice_indices = [slice(None)]
        # pad_widths has length ndim + 1
        # (including channel dim at index 0)
        # skip the first element of pad_widths
        # since we already handle channels with slice(None)
        for i, (pw, _) in enumerate(pad_widths[1:]):
            sl = slice(pw, original_shape[i + 1] + pw)
            slice_indices.append(sl)
        return x[tuple(slice_indices)]

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Pads the input array based on the configured ratios.

        Args:
            x: The input array of shape `(channels, d1, ..., dn)`.

        Returns:
            The padded array.
        """
        return self.pad(x)
