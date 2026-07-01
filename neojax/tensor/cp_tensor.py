"""Implementation of a Canonical-Polyadic-decomposed spectral tensor."""

from collections.abc import Sequence
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, PRNGKeyArray

from neojax.tensor.base_tensor import BaseTensor


@final
class CPTensor(BaseTensor):
    r"""Canonical-Polyadic-decomposed Tensor.

    Decomposes the spectral weight tensor $W$ into a sum of $R$ rank-1 tensors (where $R$ is the CP rank), represented by a core vector $\lambda$ and factor matrices $U^{(j)}$ for each mode.

    For the standard case (`separable=False`), the weight tensor $W \in \mathbb{C}^{C_{out} \times C_{in} \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c_{out}, c_{in}, x_1, \dots, x_d} = \sum_{r=1}^R \lambda_r U^{(1)}_{c_{out}, r} U^{(2)}_{c_{in}, r} \prod_{j=1}^d U^{(j+2)}_{x_j, r}
    $$

    For the separable case (`separable=True`), the weight tensor $W \in \mathbb{C}^{C \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c, x_1, \dots, x_d} = \sum_{r=1}^R \lambda_r U^{(1)}_{c, r} \prod_{j=1}^d U^{(j+1)}_{x_j, r}
    $$

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
        ranks: Single rank for CP tensor. In a CP tensor,
            all modes share the same rank.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels + out_channels)) ** 0.5.
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        * **core_tensors** (`tuple[Complex[Array, "r"], ...]`): Learnable complex core weight tensors.
        * **factor_matrices** (`tuple[Complex[Array, "..."], ...]`): Learnable factor matrices for each dimension.
        * **einsum_str** (`str`): Contraction string passed directly to `jnp.einsum`.
        * **to_dense_einsum_str** (`str`): Contraction string passed to `jnp.einsum` in `to_dense`.
        * **ndim** (`int`): Spatial dimensions of the tensor structure.
        * **separable** (bool): Whether to use separable implementation of contraction.
    """

    core_tensors: tuple[Complex[Array, "r"], ...]
    factor_matrices: tuple[Complex[Array, "..."], ...]
    einsum_str: str = eqx.field(static=True)
    to_dense_einsum_str: str = eqx.field(static=True)
    ndim: int = eqx.field(static=True)
    separable: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        in_channels: int,
        out_channels: int,
        modes: Sequence[int],
        ranks: int,
        init_std: float | Literal["auto"] = "auto",
        separable: bool = False,
    ) -> None:
        self.separable = separable
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        if isinstance(init_std, str) and init_std == "auto":
            if separable:
                init_std = (2.0 / in_channels) ** 0.5
            else:
                init_std = (2.0 / (in_channels * out_channels)) ** 0.5
        if not isinstance(init_std, float):
            raise ValueError("'init_std' must be float or 'auto'.")

        # use modes instead of ranks
        self.ndim = len(modes)
        num_corners = 2 ** (len(modes) - 1)

        if separable:
            tensor_dims = [in_channels, *modes]
        else:
            tensor_dims = [out_channels, in_channels, *modes]

        # calculate variance-preserving scaling factor s
        # p is the number of factor matrices: len(tensor_dims)
        # total terms multiplied: p + 1 (including core)
        p = len(tensor_dims) + 1
        s = (init_std / (ranks ** 0.5)) ** (1.0 / p)
        scale = s / (2.0 ** 0.5)

        core_tensors = []
        for _ in range(num_corners):
            rkey, ikey, key = jr.split(key, 3)
            ct_real = jr.normal(rkey, (ranks,))
            ct_imag = jr.normal(ikey, (ranks,))
            core_tensors.append(scale * (ct_real + 1j * ct_imag))
        self.core_tensors = tuple(core_tensors)

        factor_matrices = []
        for _ in range(num_corners):
            for d in tensor_dims:
                rkey, ikey, key = jr.split(key, 3)
                u_real = jr.normal(rkey, (d, ranks))
                u_imag = jr.normal(ikey, (d, ranks))
                factor_matrices.append(scale * (u_real + 1j * u_imag))

        self.factor_matrices = tuple(factor_matrices)

        self.einsum_str = self._assemble_einsum_str()
        self.to_dense_einsum_str = self._assemble_dense_einsum_str()

    def _assemble_einsum_str(self) -> str:
        """Assembles the n-dim einsum contraction notation for weight-input contraction.

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

        if self.separable:
            # see below but without output channel for details
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim + 2))])
            ch_i = indices[0]
            rank_i = indices[1]
            spatial_i = indices[2:]

            # channel factor matrix
            ch_f_str = ch_i + rank_i
            # spatial factor matrices
            s_f_str = ",".join([d + rank_i for d in spatial_i])
            # core tensor
            core_str = rank_i
            # input signal
            input_str = ch_i + spatial_i
            # output
            output_str = ch_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str},{input_str}->{output_str}"
            return einsum_str
        else:
            # input x has shape (in_channels, d1, ..., dN)
            # core_tensor has shape (r,)
            # factor matrices have shapes:
            # (out_channels, r), (in_channels, r), (d1, r), ..., (dN, r)
            # need 1 rank index, N spatial dim indices and 2 channel indices
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim + 3))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2]
            spatial_i = indices[3:]
            # out- and in- channel factor matrices
            ch_f_str = f"{oc_i + rank_i},{ic_i + rank_i}"
            # spatial factor matrices
            s_f_str = ",".join([d + rank_i for d in spatial_i])
            # core tensor
            core_str = rank_i
            # input signal
            input_str = ic_i + spatial_i
            # output
            output_str = oc_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str},{input_str}->{output_str}"
            return einsum_str

    def _assemble_dense_einsum_str(self) -> str:
        """Assembles the n-dim einsum contraction notation for dense contraction.

        The resulting einsum string expects the factor matrices
        to be passed first, followed by the core tensors.

        ```python
        jnp.einsum(
            "<einsum_string>",
            *factor_matrices,
            core_tensor
        )
        ```

        Returns:
            The einsum contraction string (explicit).

        Raises:
            ValueError: If total number of indices is greater than 26.
        """
        if self.ndim > 26:
            raise ValueError("Ran out of indices. Too many input dims.")

        if self.separable:
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim + 2))])
            ch_i = indices[0]
            rank_i = indices[1]
            spatial_i = indices[2:]

            # channel factor matrix
            ch_f_str = f"{ch_i}{rank_i}"
            # spatial factor matrices
            s_f_str = ",".join([d + rank_i for d in spatial_i])
            # core tensor
            core_str = rank_i
            out_str = ch_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str}->{out_str}"
            return einsum_str
        else:
            # core_tensor has shape (r,)
            # factor matrices have shapes:
            # (out_channels, r), (in_channels, r), (d1, r), ..., (dN, r)
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim + 3))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2]
            spatial_i = indices[3:]
            # out- and in- channel factor matrices
            ch_f_str = f"{oc_i + rank_i},{ic_i + rank_i}"
            # spatial factor matrices
            s_f_str = ",".join(d + rank_i for d in spatial_i)
            # core tensor
            core_str = rank_i
            out_str = oc_i + ic_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str}->{out_str}"
            return einsum_str

    def to_dense(self) -> Complex[Array, "n_corners ..."]:
        """Reconstruct factorized tensors into single dense tensor.

        Useful, e.g. if the `reconstructed` flag is `True`.

        Returns:
            Reconstructed dense tensor stacked along first axis.
        """
        stride = self.ndim + 1 if self.separable else self.ndim + 2
        stacked_core_tensors = jnp.stack(self.core_tensors, axis=0)
        sorted_factors = [
            jnp.stack([t for t in self.factor_matrices[i::stride]], axis=0)
            for i in range(stride)
        ]
        # vectorize over n_corners
        return jax.vmap(jnp.einsum, in_axes=(None,) + (0,) * (stride + 1))(
            self.to_dense_einsum_str, *sorted_factors, stacked_core_tensors
        )

    def __call__(
        self, corner_idx: int, x_ft_slice: Complex[Array, "in_c ..."]
    ) -> Complex[Array, "..."]:
        """Performs cp tensor contraction.

        Args:
            corner_idx: Corner index of the n-dim FFT hypercube.
            x_ft_slice: Mode slice of fourier transformed input.

        Returns:
            Tensor contraction of respective weight and input.
        """
        stride = self.ndim + 1 if self.separable else self.ndim + 2
        return jnp.einsum(
            self.einsum_str,
            *self.factor_matrices[corner_idx * stride : corner_idx * stride + stride],
            self.core_tensors[corner_idx],
            x_ft_slice,
        )
