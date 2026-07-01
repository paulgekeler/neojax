"""Implementation of resampling for n-dimensional interpolation."""

import itertools
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class Resampler(eqx.Module):
    """Resample functions via n-dimensional interpolation.

    Uses linear interpolation for 1D inputs and bicubic interpolation
    for 2D inputs.

    For 3D and higher-dimensional inputs, spectral interpolation is used as follows:

    1. Input is transformed to frequency domain using real n-dim FFT.
    2. Frequency components are resized.
    3. Inverse FFT to spatial domain, resulting in smooth, alias-free
        interpolation.

    Args:
        input_shape: Shape of input including channel dimension,
            namely (c, d1, ..., dN).
        res_scale: Scaling factor along each of the dimensions
            in `axes`. If `res_scale` is scalar, then isotropic
            scaling is performed.
        axes: Optional axes/spatial dimension along which interpolation
            will be performed. Channel dimension is treated as 0-th dimension.
        output_shape: Optional output shape. Default is None.
            If `output_shape` is passed, `res_scale` is ignored.

    ??? info "Internal Attributes"
        * **res_scale** (`tuple[int, ...]`): Scaling factor per spatial dimension.
        * **axes** (`tuple[int, ...]`): Axes along which to interpolate.
        * **output_shape** (`tuple[int, ...]`): Output shape including channel dimension.
        * **input_shape** (`tuple[int, ...]`): Shape of input including channel dimension.
        * **output_modes** (`tuple[int, ...]`): Output modes to resize frequency components with.

    Example:
        ```python
        import jax.random as jr
        from neojax.nn import Resampler

        key = jr.key(0)
        x = jr.normal(key=key, shape=(2, 10))
        resampler_1d = Resampler(input_shape=x.shape, res_scale=2, axes=1)

        # out.shape = (2, 20)
        out = resampler_1d(x)
        ```

    !!! info
        This class provides the same functionality as the `resample` function
        in the `neuraloperator` library.
        It imitates the `align_corners` behavior using `jax.image.scale_and_translate`
        for 2d inputs. See [here](https://docs.jax.dev/en/latest/_autosummary/
        jax.image.scale_and_translate.html#jax.image.scale_and_translate) for details.
    """

    res_scale: tuple[int, ...] = eqx.field(static=True)
    axes: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    input_shape: tuple[int, ...] = eqx.field(static=True)
    output_modes: tuple[int, ...] | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        input_shape: Sequence[int],
        res_scale: float | int | Sequence[int | float],
        axes: int | tuple[int, ...] | None,
        output_shape: Sequence[int] | None = None,
    ) -> None:
        self.input_shape = input_shape
        sdims = len(input_shape) - 1
        if isinstance(res_scale, (int, float)):
            if axes is None:
                # offset axes because of vmap
                self.axes = tuple(range(sdims - 1))
                self.res_scale = (res_scale,) * sdims
            elif isinstance(axes, int):
                # offset axes because of vmap
                self.axes = (axes - 1,)
                self.res_scale = (res_scale,)
            else:
                self.res_scale = (res_scale,) * sdims
                self.axes = tuple(ax - 1 for ax in axes)
        else:
            if len(res_scale) != len(axes) != sdims:
                raise ValueError(
                    "Lengths of res_scale and axes or n spatial dims don't match."
                )
            self.res_scale = tuple(res_scale)
            if isinstance(axes, int):
                self.axes = (axes - 1,)
            else:
                self.axes = tuple(ax - 1 for ax in axes)

        if output_shape is None:
            self.output_shape = (input_shape[0],) + tuple(
                int(round(r * s))
                for r, s in zip(self.res_scale, input_shape[1:], strict=True)
            )
        else:
            self.output_shape = output_shape
        # get new fft modes separately
        if len(input_shape[1:]) > 2:
            self._compute_fft_output_shape()

    def _compute_fft_output_shape(self) -> None:
        """Pre-computes output shape for fft-resampling."""
        output_shape = list(self.output_shape)
        # truncate last dim from 0 to modes[-1] (hermitian symmetry)
        output_shape[-1] = output_shape[-1] // 2 + 1
        self.output_modes = tuple(output_shape[1:])

    def _resample_1d(self, x: Float[Array, "c in_l"]) -> Float[Array, "c out_l"]:
        """Resamples 1D inputs.

        Args:
            x: Inputs shaped (c, d1).

        Results:
            Resampled outputs.
        """
        # treat original dim and new dim as linspaced [0, 1] range
        interp_vmap = jax.vmap(jnp.interp, in_axes=(None, None, 0, None, None))
        return interp_vmap(
            jnp.linspace(0, 1, self.output_shape[-1]),
            jnp.linspace(0, 1, self.input_shape[-1]),
            x,
            "extrapolate",
            "extrapolate",
        )

    def _resample_2d(
        self, x: Float[Array, "c in_h in_w"]
    ) -> Float[Array, "c out_h out_w"]:
        """Resamples 2D inputs.

        Args:
            x: Inputs shaped (c, d1, d2).

        Returns:
            Resampled outputs.
        """
        # m are the spatial input dims
        # n are the spatial output dims
        scale = (jnp.array(self.output_shape[1:]) - 1) / (
            jnp.array(self.input_shape[1:]) - 1
        )
        translation = 0.5 * (1 - scale)
        # pass all args as positional args
        # jax raises TypeError for method string if passed as keyword arg
        interp_vmap = jax.vmap(
            jax.image.scale_and_translate,
            in_axes=(0, None, None, None, None, None, None),
        )
        return interp_vmap(
            x,
            self.output_shape[1:],
            self.axes,
            scale,
            translation,
            "bicubic-pytorch",
            False,
        )

    def _resample_nd(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Resamples 3D or higher dim inputs.

        Args:
            x: Inputs shaped (c, d1, ..., dN).

        Returns:
            Resampled outputs.
        """
        out_fft = jnp.zeros(
            (self.input_shape[0], *self.output_modes), dtype=jnp.complex64
        )

        def single_resample(xs, out):

            x_fft = jnp.fft.rfftn(xs, norm="forward", axes=self.axes)
            modes = [
                min(i, j)
                for (i, j) in zip(
                    self.output_modes, x_fft.shape[-len(self.axes) :], strict=True
                )
            ]
            mode_indexing = [((None, m // 2), (-m // 2, None)) for m in modes[:-1]] + [
                ((None, modes[-1]),)
            ]

            for _, boundaries in enumerate(itertools.product(*mode_indexing)):
                idx_tuple = tuple(slice(*b) for b in boundaries)

                out = out.at[idx_tuple].set(x_fft[idx_tuple])

            return jnp.fft.irfftn(
                out, s=self.output_shape[1:], norm="forward", axes=self.axes
            )

        interp_vmap = jax.vmap(single_resample, in_axes=(0, 0))
        return interp_vmap(x, out_fft)

    def __call__(self, x: Float[Array, "c ..."]) -> Float[Array, "c ..."]:
        """Resamples n-dimensional input array.

        Args:
            x: Inputs shaped (c, d1, ..., dN).

        Returns:
            Resampled outputs.
        """
        if len(self.axes) == 1:
            return self._resample_1d(x)
        elif len(self.axes) == 2:
            return self._resample_2d(x)
        else:
            return self._resample_nd(x)
