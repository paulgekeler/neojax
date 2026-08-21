"""Implementation of N-dimensional continuous spectral convolution for meshes."""

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact, Int, PRNGKeyArray

from neojax.nn.spectral_conv import SpectralConvNd
from neojax.tensor import BaseTensor, DenseTensor


class GeoSpectralConvNd(eqx.Module):
    """N-dimensional geometry-aware continuous spectral convolution.

    Wraps a standard `SpectralConvNd` and extends it to support continuous
    Fourier transforms on irregular meshes via direct DFT summation. When
    no coordinate meshes are provided at call time, it falls back to the
    standard fast grid-based spectral convolution.

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to retain.
        ranks: Number of factorization ranks.
        init_std: Weight initialization standard deviation.
        enforce_hermitian_symmetry: Enforce Hermitian symmetry for real outputs.
        fft_norm: FFT normalization.
        is_complex_data: If True, uses full complex FFT.
        resolution_scaling_factor: Scale factor for grid resolution.
        factorization: Tensor factorization scheme.
        implementation: Weight reconstruction mode.
        separable: If True, uses separable contraction.

    ??? info "Internal Attributes"
        * **conv** (`SpectralConvNd`): The underlying grid spectral convolution module
            whose weights and modes are reused for the continuous transform.
    """

    conv: SpectralConvNd

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
        self.conv = SpectralConvNd(
            key=key,
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            ranks=ranks,
            init_std=init_std,
            enforce_hermitian_symmetry=enforce_hermitian_symmetry,
            fft_norm=fft_norm,
            is_complex_data=is_complex_data,
            resolution_scaling_factor=resolution_scaling_factor,
            factorization=factorization,
            implementation=implementation,
            separable=separable,
        )

    def __call__(
        self,
        u: Inexact[Array, "in_c ..."],
        x_in: Float[Array, "N d"] | None = None,
        x_out: Float[Array, "M d"] | None = None,
    ) -> Inexact[Array, "out_c ..."]:
        """Perform continuous or discrete spectral convolution.

        Args:
            u: Input features on the grid or mesh.
            x_in: Input spatial coordinates. If None, grid-based conv is used.
            x_out: Output spatial coordinates. Defaults to `x_in`.

        Returns:
            Output features after spectral convolution.
        """
        if x_in is None:
            return self.conv(u)

        if x_out is None:
            x_out = x_in

        N = x_in.shape[0]

        # Accumulate outputs across all corners of the Fourier space
        u_out = jnp.zeros(
            (self.conv.out_channels, x_out.shape[0]),
            dtype=jnp.complex64 if self.conv.is_complex_data else jnp.float32,
        )

        if self.conv.implementation == "reconstructed":
            dense_weights = self.conv.weights.to_dense()
            weights = DenseTensor.from_weights(dense_weights, separable=self.conv.separable)
        else:
            weights = self.conv.weights

        for corner_idx in range(self.conv.num_corners):
            K_corner = self.get_corner_wavenumbers(corner_idx)

            # Basis for continuous forward DFT: exp(-2pi i K x)
            basis_in = jnp.exp(-2j * jnp.pi * jnp.dot(K_corner, x_in.T))
            u_ft_corner = jnp.dot(u, basis_in.T) / N

            # Reshape to match FNO weight contraction shapes
            u_ft_corner = u_ft_corner.reshape((self.conv.in_channels,) + self.conv.modes)

            # Contract weight with Fourier coefficients
            out_ft_corner = weights(corner_idx, u_ft_corner)

            # Reshape back to flat wavenumbers
            modes_prod = K_corner.shape[0]
            out_ft_corner = out_ft_corner.reshape(self.conv.out_channels, modes_prod)

            # Basis for continuous inverse DFT: exp(2pi i K x)
            basis_out = jnp.exp(2j * jnp.pi * jnp.dot(K_corner, x_out.T))

            # Multiply coefficients by basis function values
            val = out_ft_corner[..., None] * basis_out[None, ...]

            if self.conv.is_complex_data:
                u_out_corner = jnp.sum(val, axis=1)
            else:
                # Hermitian symmetry optimization for real signals:
                # Summing positive frequencies is equal to 2 * Re(z) for positive modes
                factor = jnp.where(K_corner[:, -1:] > 0, 2.0, 1.0)
                val = val * factor.reshape(1, -1, 1)
                u_out_corner = jnp.real(jnp.sum(val, axis=1))

            u_out += u_out_corner

        return u_out

    def get_corner_wavenumbers(self, corner_idx: int) -> Int[Array, "modes_prod ndim"]:
        """Generates coordinate-aligned wavenumbers for a given corner slice of the Fourier space.

        Args:
            corner_idx: Integer index encoding which corner of the Fourier cube to use.
                The $j$-th bit of ``corner_idx`` controls the sign of the $j$-th frequency
                axis: `0` means non-negative frequencies, `1` means negative frequencies.

        Returns:
            Integer wavenumber matrix of shape `(modes_prod, ndim)` where
            `modes_prod = prod(modes)`.
        """
        modes = self.conv.modes
        ndim = len(modes)
        wavenumbers_list = []
        for j in range(ndim):
            m = modes[j]
            if j == ndim - 1 and not self.conv.is_complex_data:
                w = jnp.arange(0, m)
            else:
                is_negative_edge = (corner_idx >> j) & 1
                if is_negative_edge:
                    w = jnp.arange(-m, 0)
                else:
                    w = jnp.arange(0, m)
            wavenumbers_list.append(w)

        grids = jnp.meshgrid(*wavenumbers_list, indexing="ij")
        K_corner = jnp.stack([g.ravel() for g in grids], axis=-1)
        return K_corner
