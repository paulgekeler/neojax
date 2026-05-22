"""Implementation of a general n-dimensional spectral convolution."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class SpectralConvNd(eqx.Module):
    """General n-dimensional spectral convolution layer.

    This layer computes the real n-dimensional forward FFT,
    truncates the higher frequency modes
    according to the specified `modes`,
    multiplies the remaining modes with learnable complex weights,
    and transforms the result back to the spatial domain
    using the inverse FFT.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
        key: PRNG key for weight initialization.

    Attributes:
        weights: Learnable complex weights
            for the retained Fourier modes.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.

    Example:
        ```python
        import jax.random as jr
        from neojax.nn import SpectralConvNd

        key = jr.PRNGKey(0)

        # 1D Spectral Convolution
        conv1d = SpectralConvNd(3, 16, modes=(16,), key=key)

        # 2D Spectral Convolution
        conv2d = SpectralConvNd(3, 16, modes=(16, 16), key=key)

        # 3D Spectral Convolution
        conv3d = SpectralConvNd(3, 16, modes=(16, 16, 16), key=key)
        ```

    Notes:
        The current implementation doesn't support the following
        features yet:
            - Tucker factorization
            - Complex inputs (always truncates final dim currently)
            - Super resolution outputs, i.e., rescaling
        This is planned for an upcoming release.
    """

    weights: tuple[Complex[Array, "out_c in_c ..."], ...]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        num_corners = 2 ** (len(modes) - 1)
        scale = 1.0 / (in_channels * out_channels)
        keys = jr.split(key, num_corners)

        weights_list = []
        weight_shape = (out_channels, in_channels) + modes

        for k in keys:
            w_real = jr.normal(k, weight_shape)
            w_imag = jr.normal(jr.split(k)[0], weight_shape)
            weights_list.append(scale * (w_real + 1j * w_imag))

        self.weights = tuple(weights_list)

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Perform n-dimensional spectral convolution.

        Args:
            x: Input signal.

        Returns:
            Output signal.
        """
        spatial_shape = x.shape[1:]
        ndim = len(self.modes)
        # truncate last dim to spatial_shape[-1] // 2 + 1
        x_ft = jnp.fft.rfftn(x, axes=tuple(range(1, ndim + 1)))

        out_ft_shape = (self.out_channels,) + x_ft.shape[1:]
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)

        # iterate through the corners using binary representation
        for corner_idx, weight_tensor in enumerate(self.weights):
            slices = [slice(None)]

            for d in range(ndim):
                if d == ndim - 1:
                    # truncate last dim from 0 to modes[-1]
                    slices.append(slice(0, self.modes[d]))
                else:
                    # other dims use the positive or negative freq edge
                    is_negative_edge = (corner_idx >> d) & 1
                    if is_negative_edge:
                        slices.append(slice(-self.modes[d], None))
                    else:
                        slices.append(slice(0, self.modes[d]))

            # channel-wise matrix multiplication using einsum
            grid_slice = tuple(slices)
            out_ft = out_ft.at[grid_slice].set(
                jnp.einsum("oi...,i...->o...", weight_tensor, x_ft[grid_slice])
            )

        return jnp.fft.irfftn(out_ft, s=spatial_shape, axes=tuple(range(1, ndim + 1)))
