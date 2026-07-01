"""Implementation of Tucker-factorized spectral tensor."""

from collections.abc import Sequence
from typing import Literal, final
from math import prod
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, PRNGKeyArray

from neojax.tensor.base_tensor import BaseTensor


@final
class TuckerTensor(BaseTensor):
    r"""Tucker-factorized spectral tensor.

    Decomposes the spectral weight tensor $W$ into a core tensor $G$ and factor matrices $U^{(j)}$ for each dimension.

    For the standard case (`separable=False`), the weight tensor $W \in \mathbb{C}^{C_{out} \times C_{in} \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c_{out}, c_{in}, x_1, \dots, x_d} = \sum_{r_1, \dots, r_{d+2}} G_{r_1, \dots, r_{d+2}} U^{(1)}_{c_{out}, r_1} U^{(2)}_{c_{in}, r_2} \prod_{j=1}^d U^{(j+2)}_{x_j, r_{j+2}}
    $$

    For the separable case (`separable=True`), the weight tensor $W \in \mathbb{C}^{C \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c, x_1, \dots, x_d} = \sum_{r_1, \dots, r_{d+1}} G_{r_1, \dots, r_{d+1}} U^{(1)}_{c, r_1} \prod_{j=1}^d U^{(j+1)}_{x_j, r_{j+1}}
    $$

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
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels + out_channels)) ** 0.5.
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        * **core_tensors** (`tuple[Complex[Array, ...], ...]`): Learnable complex core weight tensors.
        * **factor_matrices** (`tuple[Complex[Array, ...], ...]`): Learnable factor matrices for each dimension.
        * **einsum_str** (`str`): Contraction string passed directly to `jnp.einsum`.
        * **to_dense_einsum_str** (`str`): Contraction string passed to `jnp.einsum` in `to_dense`.
        * **ndim** (`int`): Total dimensions of the tensor structure (channels + spatial).
        * **separable** (bool): Whether to use separable implementation of contraction.
    """

    core_tensors: tuple[Complex[Array, "..."], ...]
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
        ranks: int | Sequence[int],
        init_std: float | Literal["auto"] = "auto",
        separable: bool = False,
    ) -> None:
        self.separable = separable
        if separable and in_channels != out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) must equal out_channels ({out_channels}) when separable is True."
            )

        expected_len = 1 + len(modes) if separable else 2 + len(modes)
        if isinstance(ranks, int):
            ranks = [ranks] * expected_len
        elif len(ranks) != expected_len:
            raise ValueError(
                f"ranks must be of length {expected_len} when separable is {separable}"
            )

        if isinstance(init_std, str) and init_std == "auto":
            if separable:
                init_std = (2.0 / in_channels) ** 0.5
            else:
                init_std = (2.0 / (in_channels * out_channels)) ** 0.5
        if not isinstance(init_std, float):
            raise ValueError("'init_std' must be float or 'auto'.")

        self.ndim = len(ranks)
        num_corners = 2 ** (len(modes) - 1)

        if separable:
            tensor_dims = [in_channels, *modes]
        else:
            tensor_dims = [out_channels, in_channels, *modes]

        # calculate ranks product
        ranks_prod = prod(ranks)

        # p is the number of factor matrices: len(tensor_dims)
        # total terms multiplied: p + 1 (including core)
        p = len(tensor_dims) + 1
        s = (init_std / (ranks_prod ** 0.5)) ** (1.0 / p)
        scale = s / (2.0 ** 0.5)

        core_tensors = []
        for _ in range(num_corners):
            rkey, ikey, key = jr.split(key, 3)
            ct_real = jr.normal(rkey, ranks)
            ct_imag = jr.normal(ikey, ranks)
            core_tensors.append(scale * (ct_real + 1j * ct_imag))
        self.core_tensors = tuple(core_tensors)

        factor_matrices = []
        for _ in range(num_corners):
            for r, d in zip(ranks, tensor_dims, strict=True):
                rkey, ikey, key = jr.split(key, 3)
                u_real = jr.normal(rkey, (d, r))
                u_imag = jr.normal(ikey, (d, r))
                factor_matrices.append(scale * (u_real + 1j * u_imag))

        self.factor_matrices = tuple(factor_matrices)
        # determine einsum indices here
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
            # same as below but without out_channels
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2))])
            ch_i = indices[0]
            rank_i = indices[1 : 1 + self.ndim]
            spatial_i = indices[-(self.ndim - 1) :]
            ch_f_str = ch_i + rank_i[0]
            s_f_str = ",".join(
                [d + r for r, d in zip(rank_i[1:], spatial_i, strict=True)]
            )
            core_str = rank_i
            input_str = ch_i + spatial_i
            output_str = ch_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str},{input_str}->{output_str}"
            return einsum_str
        else:
            # input x has shape (in_channels, d1, ..., dN)
            # core_tensor has shape (r1, ..., rD)
            # factor matrices have shapes:
            # (out_channels, r1), (in_channels, r2), (d1, r3), ..., (dN, rD)
            # need D rank indices, N spatial dim indices and 2 channel indices
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2 : 2 + self.ndim]
            spatial_i = indices[-(self.ndim - 2) :]
            ch_f_str = f"{oc_i + rank_i[0]},{ic_i + rank_i[1]}"
            s_f_str = ",".join(
                [d + r for r, d in zip(rank_i[2:], spatial_i, strict=True)]
            )
            core_str = rank_i
            input_str = ic_i + spatial_i
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
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2))])
            ch_i = indices[0]
            rank_i = indices[1 : 1 + self.ndim]
            spatial_i = indices[-(self.ndim - 1) :]

            ch_f_str = ch_i + rank_i[0]
            s_f_str = ",".join(
                [d + r for r, d in zip(rank_i[1:], spatial_i, strict=True)]
            )
            core_str = rank_i
            out_str = ch_i + spatial_i
            einsum_str = f"{ch_f_str},{s_f_str},{core_str}->{out_str}"
            return einsum_str
        else:
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2 : 2 + self.ndim]
            spatial_i = indices[-(self.ndim - 2) :]
            # out- and in- channel factor matrices
            ch_f_str = f"{oc_i + rank_i[0]},{ic_i + rank_i[1]}"
            # spatial factor matrices
            s_f_str = ",".join(
                [d + r for r, d in zip(rank_i[2:], spatial_i, strict=True)]
            )
            # core tensor
            core_str = rank_i
            out_str = f"{oc_i}{ic_i}{spatial_i}"
            einsum_str = f"{ch_f_str},{s_f_str},{core_str}->{out_str}"
            return einsum_str

    def to_dense(self) -> Complex[Array, "n_corners ..."]:
        """Reconstruct factorized tensors into single dense tensor.

        Useful, e.g. if the `reconstructed` flag is `True`.

        Returns:
            Reconstructed dense tensors stacked along first axis.
        """
        stacked_core_tensors = jnp.stack(self.core_tensors, axis=0)
        sorted_factors = [
            jnp.stack([t for t in self.factor_matrices[i :: self.ndim]], axis=0)
            for i in range(self.ndim)
        ]
        return jax.vmap(jnp.einsum, in_axes=(None,) + (0,) * (self.ndim + 1))(
            self.to_dense_einsum_str, *sorted_factors, stacked_core_tensors
        )

    def __call__(
        self, corner_idx: int, x_ft_slice: Complex[Array, "in_c ..."]
    ) -> Complex[Array, "..."]:
        """Performs tucker tensor contraction.

        Args:
            corner_idx: Corner index of the n-dim FFT hypercube.
            x_ft_slice: Mode slice of fourier transformed input.

        Returns:
            Tensor contraction of respective weight and input.
        """
        return jnp.einsum(
            self.einsum_str,
            *self.factor_matrices[
                corner_idx * self.ndim : corner_idx * self.ndim + self.ndim
            ],
            self.core_tensors[corner_idx],
            x_ft_slice,
        )
