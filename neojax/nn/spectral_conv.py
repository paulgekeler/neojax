"""Implementation of a general n-dimensional spectral convolution."""

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PRNGKeyArray

from neojax.tensor import BaseTensor, CPTensor, DenseTensor, TTTensor, TuckerTensor


class SpectralConvNd(eqx.Module):
    """General n-dimensional spectral convolution layer.

    This layer computes the real n-dimensional forward FFT,
    truncates the higher frequency modes
    according to the specified `modes`,
    multiplies the remaining modes with learnable complex weights,
    and transforms the result back to the spatial domain
    using the inverse FFT.

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
            Can be an integer or sequence of integers.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not, should be a sequence of ranks.
            Default is None.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels * out_channels)) ** 0.5.
        enforce_hermitian_symmetry: Whether to enforce
            hermitian symmetry on the outputs before calling `irfftn`.
            Default is True.  If set to False, cuFFT on GPU may cause line artifacts
            when calling irfftn.
        fft_norm: FFT normalization. Can be `None`, `"backward"`,
            `"ortho"` or `"forward"`. Default is `"forward"`.
        is_complex_data: Whether input data is complex valued.
            If True, uses full FFT. Default is False.
        resolution_scaling_factor: Factor by which to scale the domain
            resolution of the function. Default is None, no scaling.
        factorization: Tensor factorization type. Can be a `BaseTensor`
            instance, a string ("tucker", "cp", "tt"), or None.
            Default is None.
        implementation: Weight reconstruction mode.
            Can be "reconstructed" or "factorized".
            Default is "factorized".
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **weights** (`BaseTensor`): Learnable complex weights for the retained Fourier modes.
        * **in_channels** (`int`): Number of input channels.
        * **out_channels** (`int`): Number of output channels.
        * **modes** (`tuple[int, ...]`): Number of Fourier modes to retain across each spatial dim.
        * **fft_norm** (`str | None`): FFT normalization.
        * **is_complex_data** (`bool`): Whether input data is complex valued.
        * **enforce_hermitian_symmetry** (`bool`): Whether to enforce
            hermitian symmetry on the outputs.
        * **resolution_scaling_factor** (`float | int | None`): Factor to scale domain resolution. Default is None.
        * **implementation** (`str`): Weight reconstruction mode.
        * **separable** (`bool`): Whether the convolution is separable.
        * **num_corners** (`int`): Number of corners of the n-dimensional FFT hypercube.

    Examples:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import SpectralConvNd

        key = jr.key(0)

        # 1D Spectral Convolution
        conv1d = SpectralConvNd(key, 3, 16, modes=16)
        # or
        conv1d = SpectralConvNd(key, 3, 16, modes=(16,))

        # 2D Spectral Convolution
        conv2d = SpectralConvNd(key, 3, 16, modes=(16, 16))

        # 3D Spectral Convolution
        conv3d = SpectralConvNd(key, 3, 16, modes=(16, 16, 16))

        # Input shape: (channels, d1, ..., dN)
        x = jnp.ones((3, 32, 32))
        out = conv2d(x)
        ```

    !!! info "Implementation Details"
        Unlike the reference PyTorch `neuraloperator` library, this implementation
        does not shift the frequency spectrum (e.g., using `fftshift` / `ifftshift`).
        Instead, it directly indexes the positive and negative frequency modes from the
        boundaries of the unshifted FFT representation using a binary bitmask to select the
        corners of an n-dimensional hypercube. This avoids memory copy/transposition
        overhead at runtime and generalizes to arbitrary dimensions.

    !!! warning "Limitation: Half & Mixed Precision"
        Unlike PyTorch, JAX does not natively support complex half-precision types
        (like `complex32`). Therefore, setting `fno_block_precision` to `"half"` or
        `"mixed"` is not supported, and all spectral operations run in full
        precision (`complex64`).

    """

    weights: BaseTensor
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes: tuple[int, ...] = eqx.field(static=True)
    fft_norm: str | None = eqx.field(static=True)
    is_complex_data: bool = eqx.field(static=True)
    enforce_hermitian_symmetry: bool = eqx.field(static=True)
    resolution_scaling_factor: float | int | None = eqx.field(static=True, default=None)
    implementation: Literal["reconstructed", "factorized"] = eqx.field(static=True)
    separable: bool = eqx.field(static=True)
    num_corners: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        ranks: int | Sequence[int] | None = None,
        init_std: float | Literal["auto"] = "auto",
        enforce_hermitian_symmetry: bool = True,
        fft_norm: Literal["forward", "backward", "ortho"] | None = "forward",
        is_complex_data: bool = False,
        resolution_scaling_factor: float | int | None = None,
        factorization: BaseTensor | Literal["tucker", "cp", "tt"] | None = None,
        implementation: Literal["reconstructed", "factorized"] = "factorized",
        separable: bool = False,
    ):
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(out_channels, int) or out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")

        if not isinstance(separable, bool):
            raise ValueError("separable must be a boolean.")
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        if not isinstance(enforce_hermitian_symmetry, bool):
            raise ValueError("enforce_hermitian_symmetry must be a boolean.")
        if not isinstance(is_complex_data, bool):
            raise ValueError("is_complex_data must be a boolean.")

        if implementation not in ["reconstructed", "factorized"]:
            raise ValueError(
                "'implementation' must be one of ['reconstructed', 'factorized']."
            )
        self.implementation = implementation
        self.separable = separable

        if fft_norm not in ["forward", "backward", "ortho", None]:
            raise ValueError(
                "'fft_norm' must be one of ['forward', 'backward', 'ortho', None]."
            )
        self.fft_norm = fft_norm

        if resolution_scaling_factor is not None:
            if not isinstance(resolution_scaling_factor, (int, float)):
                raise ValueError(
                    "resolution_scaling_factor must be an int, float, or None."
                )
            if resolution_scaling_factor <= 0:
                raise ValueError("resolution_scaling_factor must be positive.")
        self.resolution_scaling_factor = resolution_scaling_factor

        if isinstance(modes, int):
            self.modes = (modes,)
        elif isinstance(modes, Sequence):
            self.modes = tuple(modes)
        else:
            raise ValueError("modes must be an int or a sequence of ints.")

        if len(self.modes) == 0:
            raise ValueError("modes sequence cannot be empty.")
        if not all(isinstance(m, int) and m > 0 for m in self.modes):
            raise ValueError("All modes must be positive integers.")

        if not isinstance(init_std, float) and init_std != "auto":
            raise ValueError("init_std must be a float or 'auto'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enforce_hermitian_symmetry = enforce_hermitian_symmetry
        self.is_complex_data = is_complex_data

        if isinstance(factorization, BaseTensor):
            if (
                hasattr(factorization, "separable")
                and factorization.separable != separable
            ):
                raise ValueError(
                    f"factorization.separable ({factorization.separable}) must match the separable argument ({separable})."
                )
            self.weights = factorization
        elif isinstance(factorization, str):
            if factorization not in ["tucker", "cp", "tt"]:
                raise ValueError("Passed 'factorization' string invalid.")
            if ranks is None:
                raise ValueError(
                    "ranks must be provided if factorization is specified as a string"
                )
            if isinstance(ranks, int):
                if ranks <= 0:
                    raise ValueError("ranks must be a positive integer.")
            elif isinstance(ranks, Sequence):
                if not all(isinstance(r, int) and r > 0 for r in ranks):
                    raise ValueError("All ranks must be positive integers.")
            else:
                raise ValueError("ranks must be an int, a sequence of ints, or None.")

            if factorization == "tucker":
                self.weights = TuckerTensor(
                    key=key,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=self.modes,
                    ranks=ranks,
                    init_std=init_std,
                    separable=separable,
                )
            elif factorization == "tt":
                self.weights = TTTensor(
                    key=key,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=self.modes,
                    ranks=ranks,
                    init_std=init_std,
                    separable=separable,
                )
            elif factorization == "cp":
                self.weights = CPTensor(
                    key=key,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    modes=self.modes,
                    ranks=ranks,
                    init_std=init_std,
                    separable=separable,
                )
        elif factorization is None:
            self.weights = DenseTensor(
                key=key,
                in_channels=in_channels,
                out_channels=out_channels,
                modes=self.modes,
                init_std=init_std,
                separable=separable,
            )
        else:
            raise ValueError("Passed 'factorization' invalid.")

        self.num_corners = 2 ** (len(self.modes) - 1)

    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Perform n-dimensional spectral convolution.

        Args:
            x: Input signal.

        Returns:
            Output signal.
        """
        ndim = len(self.modes)
        if x.shape[0] != self.in_channels:
            raise ValueError(
                f"Input channel dimension {x.shape[0]} does not match expected in_channels {self.in_channels}."
            )
        if len(x.shape) != ndim + 1:
            raise ValueError(
                f"Input must have {ndim + 1} dimensions (1 channel dim + {ndim} spatial dims), got {len(x.shape)} dimensions."
            )

        spatial_shape = x.shape[1:]
        for d, mode in enumerate(self.modes):
            if d == ndim - 1 and not self.is_complex_data:
                max_available_modes = spatial_shape[d] // 2 + 1
            else:
                max_available_modes = spatial_shape[d]
            if max_available_modes < mode:
                raise ValueError(
                    f"Spatial dimension {d} of input has resolution {spatial_shape[d]}, "
                    f"which is too small to retain {mode} modes."
                )

        if self.is_complex_data:
            x_ft = jnp.fft.fftn(x, axes=tuple(range(1, ndim + 1)), norm=self.fft_norm)
        else:
            # Truncate last dim to spatial_shape[-1] // 2 + 1
            x_ft = jnp.fft.rfftn(x, axes=tuple(range(1, ndim + 1)), norm=self.fft_norm)

        out_ft_shape = (self.out_channels,) + x_ft.shape[1:]
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)

        if self.implementation == "reconstructed":
            dense_weights = self.weights.to_dense()
            weights = DenseTensor.from_weights(dense_weights, separable=self.separable)
        else:
            weights = self.weights

        # Iterate through the corners using binary representation
        for corner_idx in range(self.num_corners):
            slices = [slice(None)]

            for d in range(ndim):
                if d == ndim - 1 and not self.is_complex_data:
                    # Truncate last dim from 0 to modes[-1] (hermitian symmetry)
                    slices.append(slice(0, self.modes[d]))
                else:
                    # Other dims use the positive or negative freq edge
                    is_negative_edge = (corner_idx >> d) & 1
                    if is_negative_edge:
                        slices.append(slice(-self.modes[d], None))
                    else:
                        slices.append(slice(0, self.modes[d]))

            # Channel-wise matrix multiplication using einsum
            grid_slice = tuple(slices)
            out_ft = out_ft.at[grid_slice].set(weights(corner_idx, x_ft[grid_slice]))

        # Apply resolution scaling
        if self.resolution_scaling_factor is not None:
            spatial_shape = tuple(
                round(self.resolution_scaling_factor * s) for s in spatial_shape
            )

        if self.is_complex_data:
            return jnp.fft.ifftn(
                out_ft,
                s=spatial_shape,
                axes=tuple(range(1, ndim + 1)),
                norm=self.fft_norm,
            )
        else:
            if self.enforce_hermitian_symmetry:
                # Enforce that 0-th frequency and Nyquist frequency of final dim are real valued
                out_ft = out_ft.at[..., 0].set(jnp.real(out_ft[..., 0]))
                if spatial_shape[-1] % 2 == 0:
                    out_ft = out_ft.at[..., -1].set(jnp.real(out_ft[..., -1]))
            return jnp.fft.irfftn(
                out_ft,
                s=spatial_shape,
                axes=tuple(range(1, ndim + 1)),
                norm=self.fft_norm,
            )
