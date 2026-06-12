"""General n-dimensional Tucker factorized Spectral Convolution."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class TuckerSpectralConvNd(eqx.Module):
    """General n-dimensional factorized spectral convolution layer.

    This layer computes the real n-dimensional forward FFT,
    truncates the higher frequency modes
    according to the specified `modes`,
    multiplies the remaining modes with the Tucker factorization
    of the learnable complex weight tensors
    and transforms the result back to the spatial domain
    using the inverse FFT.

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an Integer, the same number is used
            for all ranks. If not,
            should be num_spatial_dims + num_channel_dims ranks,
            e.g. in 2D 4 ranks (out_channel, in_channel, 2 spatial ranks).
        share_factor_matrices: Whether to share the factor matrices
            across the weight tensors. This further decreases the number
            of weight tensors. Only the core tensor is not shared then.
            Default is True.

    Example:
        ```python
        import jax.numpy as jnp
        import jax.random as jr
        from neojax.nn import TuckerSpectralConvNd

        key = jr.key(0)
        x = jnp.ones((3, 32, 32))
        tconv2d = TuckerSpectralConvNd(
            key, 3, 16, modes=(16, 16), ranks=4
        )
        out = tconv2d(x)
        ```

    !!! info "Upcoming Features"
        The current implementation doesn't support complex inputs
        (always truncates final dim currently).
        The addition is planned for an upcoming release.

    !!! info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **core_tensors** (`tuple[Complex[Array, ...], ...]`): Learnable complex core weight tensors.
        * **factor_matrices** (`tuple[Complex[Array, ...], ...]`): Learnable factor matrices for each dimension.
        * **einsum_str** (`str`): Static contraction string passed directly to `jnp.einsum`.
        * **ndim** (`int`): Total dimensions of the tensor structure (channels + spatial).

    ??? cite

        Useful overview on Tensor Operations.
        [Tensor Decompositions and Applications](https://epubs.siam.org/doi/epdf/10.1137/07070111X)

        ```bibtex
        @article{kolda2009tensor,
            title={Tensor decompositions and applications},
            author={Kolda, Tamara G and Bader, Brett W},
            journal={SIAM review},
            volume={51},
            number={3},
            pages={455--500},
            year={2009},
            publisher={SIAM}
        }
        ```
    """

    core_tensors: tuple[Complex[Array, "..."], ...]
    factor_matrices: tuple[Complex[Array, "..."], ...]
    einsum_str: str = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    modes: tuple[int, ...] = eqx.field(static=True)
    ranks: tuple[int, ...] = eqx.field(static=True)
    ndim: int = eqx.field(static=True)
    share_factor_matrices: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: int | Sequence[int],
        ranks: int | Sequence[int],
        share_factor_matrices: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.share_factor_matrices = share_factor_matrices

        if isinstance(modes, int):
            self.modes = (modes,)
        else:
            self.modes = modes

        if isinstance(ranks, int):
            self.ranks = (ranks,) * (2 + len(self.modes))
        else:
            if len(ranks) != 2 + len(self.modes):
                raise ValueError(
                    "Incorrect number of ranks passed."
                    " Need (2 + num_spatial_dims) ranks when passing a sequence."
                )
            self.ranks = ranks

        self.ndim = len(self.ranks)
        num_corners = 2 ** (len(modes) - 1)
        scale = 1.0 / (in_channels * out_channels)

        core_tensors = []

        for _ in range(num_corners):
            rkey, ikey, key = jr.split(key, 3)
            ct_real = jr.normal(rkey, self.ranks)
            ct_imag = jr.normal(ikey, self.ranks)
            core_tensors.append(scale * (ct_real + 1j * ct_imag))
        self.core_tensors = tuple(core_tensors)

        tensor_dims = [out_channels, in_channels, *self.modes]
        if share_factor_matrices:
            factor_matrices = []
            for r, d in zip(self.ranks, tensor_dims, strict=True):
                rkey, ikey, key = jr.split(key, 3)
                u_real = jr.normal(rkey, (r, d))
                u_imag = jr.normal(ikey, (r, d))
                factor_matrices.append(scale * (u_real + 1j * u_imag))
        else:
            factor_matrices = []
            for _ in range(num_corners):
                for r, d in zip(self.ranks, tensor_dims, strict=True):
                    rkey, ikey, key = jr.split(key, 3)
                    u_real = jr.normal(rkey, (r, d))
                    u_imag = jr.normal(ikey, (r, d))
                    factor_matrices.append(scale * (u_real + 1j * u_imag))

        self.factor_matrices = tuple(factor_matrices)
        # determine einsum indices here
        self.einsum_str = self._assemble_einsum_str()

    def _assemble_einsum_str(self) -> str:
        """Assembles the einsum contraction notation for arbitrary dimensions.

        The resulting einsum strings expects the factor matrices to be
        passed first, then the core tensor and then
        the input slice, e.g.:
        ```python
        jnp.einsum(
            "<einsum_string>",
            *factor_matrices,
            core_tensor,
            x[grid_slice]
        )
        ```

        Returns:
            The einsum contraction string (explicit).

        Raises:
            ValueError: If total number of indices is greater than 26.
        """
        if self.ndim > 26:
            raise ValueError("Ran out of indices. Too many input dims.")
        # input x has shape (in_channels, d1, ..., dN)
        # core_tensor has shape(r1, ..., rD)
        # factor matrices have shapes:
        # (r1, out_channels), (r2, in_channels), (r3, d1), ..., (rD, dN)
        # need D rank indices, N spatial dim indices and 2 channel indices
        indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2))])
        oc_i = indices[0]
        ic_i = indices[1]
        rank_i = indices[2 : 2 + self.ndim]
        spatial_i = indices[-(self.ndim - 2) :]
        # out- and in- channel factor matrices
        ch_f_str = f"{rank_i[0]} {oc_i}, {rank_i[1]} {ic_i}"
        # spatial factor matrices
        s_f_str = ", ".join(
            [f"{r} {d}" for r, d in zip(rank_i[2:], spatial_i, strict=True)]
        )
        # core tensor
        core_str = " ".join(list(rank_i))
        # input signal
        input_str = " ".join([ic_i] + list(spatial_i))
        # output
        output_str = " ".join([oc_i] + list(spatial_i))
        einsum_str = f" {ch_f_str}, {s_f_str}, {core_str}, {input_str} -> {output_str}"
        return einsum_str

    def __call__(self, x: Float[Array, "in_c ..."]) -> Float[Array, "out_c ..."]:
        """Perform n-dimensional factorized spectral convolution.

        Args:
            x: Input signal.

        Returns:
            Output signal.
        """
        spatial_shape = x.shape[1:]
        sdims = len(self.modes)
        x_ft = jnp.fft.rfftn(x, axes=tuple(range(1, sdims + 1)), norm="ortho")

        out_ft_shape = (self.out_channels,) + x_ft.shape[1:]
        out_ft = jnp.zeros(out_ft_shape, dtype=jnp.complex64)

        for corner_idx, core_tensor in enumerate(self.core_tensors):
            slices = [slice(None)]
            for d in range(sdims):
                if d == sdims - 1:
                    slices.append(slice(0, self.modes[d]))
                else:
                    is_negative_edge = (corner_idx >> d) & 1
                    if is_negative_edge:
                        slices.append(slice(-self.modes[d], None))
                    else:
                        slices.append(slice(0, self.modes[d]))

            grid_slice = tuple(slices)
            if self.share_factor_matrices:
                out_ft = out_ft.at[grid_slice].set(
                    jnp.einsum(
                        self.einsum_str,
                        *self.factor_matrices,
                        core_tensor,
                        x_ft[grid_slice],
                    )
                )
            else:
                out_ft = out_ft.at[grid_slice].set(
                    jnp.einsum(
                        self.einsum_str,
                        *self.factor_matrices[
                            corner_idx * self.ndim : corner_idx * self.ndim + self.ndim
                        ],
                        core_tensor,
                        x_ft[grid_slice],
                    )
                )

        return jnp.fft.irfftn(
            out_ft, s=spatial_shape, axes=tuple(range(1, sdims + 1)), norm="ortho"
        )
