"""Implementation of the domain padding."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Inexact


class DomainPadding(eqx.Module):
    """Applies domain padding to a signal.

    Domain Padding helps to mitigate boundary artifacts.
    In the context of FNO, it is often used
    to handle non-periodic boundary conditions
    by padding the spatial dimensions before
    the spectral convolution and cropping the result
    back to the original resolution.
    Padding is applied symmetrically to each dimension,
    i.e., a 20% (0.2) padding pads an input by 20%
    on each side along an axis.

    Args:
        padding: Percentage of padding to apply
            to each spatial dimension on both sides. Can be a single float in [0, 1]
            (applied to all dims) or a sequence of floats.
        mode: The type of padding to apply (e.g. "constant", "edge").
            Defaults to "constant" which pads with zeros.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.pad.html#jax.numpy.pad
            for all padding modes.
        resolution_scaling_factor: Scaling factor(s) for layers called between `pad`
            and `unpad`. Used to strip the padding after the input has been scaled.
            Default is 1, i.e., no scaling. If `resolution_scaling_factor` is a sequence,
            each entry is the scaling factor along that spatial dimension.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **padding** (`float | tuple[float]`): The stored padding ratios for each dimension.
        * **mode** (`str`): The padding mode.
        * **resolution_scaling_factor** (`float | int | tuple[float | int, ...]`):
    """

    padding: float | tuple[float] = eqx.field(static=True)  # static -> jittable
    mode: str = eqx.field(static=True, default="constant")
    resolution_scaling_factor: float | int | tuple[float | int, ...] = eqx.field(
        static=True
    )
    is_scaled: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        padding: float | Sequence[float],
        mode: str = "constant",
        resolution_scaling_factor: float | int | Sequence[float | int] | None = 1,
    ) -> None:
        """Initializes the DomainPadding module."""
        if isinstance(padding, float):
            self.padding = padding
        elif isinstance(padding, Sequence):
            if not isinstance(padding[0], float):
                raise ValueError(
                    "Invalid padding type. Must be float or Sequence[float]."
                )
            self.padding = tuple(padding)
            if isinstance(resolution_scaling_factor, Sequence) and len(
                resolution_scaling_factor
            ) != len(padding):
                raise ValueError(
                    "'resolution_scaling_factor' and 'padding' have unequal lengths."
                )
        else:
            raise ValueError("Invalid padding type. Must be float or Sequence[float].")
        if mode not in ("constant", "edge", "wrap", "maximum", "minimum"):
            raise ValueError(
                "Padding mode unavailable."
                " See jax.numpy.pad for padding modes without kwargs."
            )
        self.mode = mode
        if isinstance(resolution_scaling_factor, (float, int)):
            if resolution_scaling_factor != 1.0:
                self.is_scaled = True
            self.resolution_scaling_factor = resolution_scaling_factor
        elif resolution_scaling_factor is None:
            self.resolution_scaling_factor = 1
        elif isinstance(resolution_scaling_factor, Sequence):
            if not all([rs == 1 for rs in resolution_scaling_factor]):
                self.is_scaled = True
            self.resolution_scaling_factor = tuple(resolution_scaling_factor)
        else:
            raise ValueError("Invalid 'resolution_scaling_factor'.")

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
            pw = round(p * dim)
            pad_widths.append((pw, pw))
        return tuple(pad_widths)

    def pad(self, x: Inexact[Array, "c ..."]) -> Inexact[Array, "c ..."]:
        """Pads the input array based on the configured ratios.

        Args:
            x: The input array of shape `(channels, d1, ..., dN)`.

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
        self, x: Inexact[Array, "c ..."], original_shape: tuple[int, ...]
    ) -> Inexact[Array, "c ..."]:
        """Crops the padded array back to its original resolution.

        Args:
            x: The padded array.
            original_shape: The original array shape.

        Returns:
            The cropped array of original resolution.
        """
        pad_widths = self._get_pad_widths(original_shape)
        if isinstance(self.resolution_scaling_factor, (float | int)):
            res_scale_factor = (self.resolution_scaling_factor,) * len(original_shape)
        else:
            # resolution_scaling is only defined for spatial dims
            res_scale_factor = (0,) + self.resolution_scaling_factor
        if self.is_scaled:
            # if not all scaling factors are 1, we strip scaled padding
            pad_widths = tuple(
                tuple(round(i * ji) for ji in j)
                for (i, j) in zip(res_scale_factor, pad_widths, strict=True)
            )

        slice_indices = [slice(None)]
        # pad_widths has length ndim + 1
        # (including channel dim at index 0)
        # skip the first element of pad_widths
        # since we already handle channels with slice(None)
        for i, ((pw, _), rs) in enumerate(
            zip(pad_widths[1:], res_scale_factor[1:], strict=True)
        ):
            if self.is_scaled:
                sl = slice(pw, round(original_shape[i + 1] * rs) + pw)
            else:
                sl = slice(pw, original_shape[i + 1] + pw)
            slice_indices.append(sl)
        return x[tuple(slice_indices)]

    def __call__(self, x: Inexact[Array, "c ..."]) -> Inexact[Array, "c ..."]:
        """Pads the input array based on the configured ratios.

        Args:
            x: The input array of shape `(channels, d1, ..., dN)`.

        Returns:
            The padded array.
        """
        return self.pad(x)
