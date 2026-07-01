"""Implementation of Tensor Train Tensor."""

from collections.abc import Sequence
from typing import Literal
from math import prod
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Complex, PRNGKeyArray

from neojax.tensor.base_tensor import BaseTensor


class TTTensor(BaseTensor):
    r"""Tensor Train Tensor.

    Decomposes the spectral weight tensor $W$ into a train/chain of low-dimensional tensors $U^{(j)}$ connected back-to-back.

    For the standard case (`separable=False`), the weight tensor $W \in \mathbb{C}^{C_{out} \times C_{in} \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c_{out}, c_{in}, x_1, \dots, x_d} = \sum_{r_1, \dots, r_{d+1}} U^{(1)}_{c_{out}, r_1} U^{(2)}_{r_1, c_{in}, r_2} U^{(3)}_{r_2, x_1, r_3} \dots U^{(d+2)}_{r_{d+1}, x_d}
    $$

    For the separable case (`separable=True`), the weight tensor $W \in \mathbb{C}^{C \times m_1 \times \dots \times m_d}$ is reconstructed as:

    $$
    W_{c, x_1, \dots, x_d} = \sum_{r_1, \dots, r_d} U^{(1)}_{c, r_1} U^{(2)}_{r_1, x_1, r_2} \dots U^{(d+1)}_{r_d, x_d}
    $$

    Args:
        key: PRNG key for weight initialization.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes
            to retain across each spatial dim.
        ranks: Number of ranks to contract the spectral tensors to.
            If `ranks` is an integer, the same number is used
            for all ranks. If not,
            should be (num_spatial_dims + num_channel_dims) - 1 ranks,
            e.g. in 2D 3 ranks (out_channel, in_channel, 2 spatial ranks) - 1.
        init_std: Standard deviation to use for weight initialization,
            by default 'auto'. If 'auto',
            uses (2 / (in_channels + out_channels)) ** 0.5.
        separable: Whether to use separable implementation of contraction.
            If True, contracts factors of factorized tensor weight individually.
            Default is False.

    ??? info "Internal Attributes"
        * **lr_tensors** (`tuple[Complex[Array, "..."], ...]`): Learnable complex low-rank tensors.
        * **einsum_str** (`str`): Contraction string passed directly to `jnp.einsum`.
        * **to_dense_einsum_str** (`str`): Contraction string passed to `jnp.einsum` in `to_dense`.
        * **ndim** (`int`): Total dimensions of the tensor structure (channels + spatial).
        * **separable** (bool): Whether to use separable implementation of contraction.
    """

    lr_tensors: tuple[Complex[Array, "..."], ...]
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

        expected_len = len(modes) if separable else 1 + len(modes)
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

        self.ndim = len(ranks) + 1
        num_corners = 2 ** (len(modes) - 1)

        if separable:
            tensor_dims = [in_channels, *modes]
        else:
            tensor_dims = [out_channels, in_channels, *modes]

        # calculate ranks product
        ranks_prod = prod(ranks)

        # n is the number of factor tensors in the chain: len(tensor_dims)
        n = len(tensor_dims)
        s = (init_std / (ranks_prod ** 0.5)) ** (1.0 / n)
        scale = s / (2.0 ** 0.5)

        lr_tensors = []

        for _ in range(num_corners):
            for j, d in enumerate(tensor_dims):
                # len(ranks) = len(dims) - 1
                rkey, ikey, key = jr.split(key, 3)
                if j == 0:
                    # first tensor
                    lr_real = jr.normal(rkey, (d, ranks[0]))
                    lr_imag = jr.normal(ikey, (d, ranks[0]))
                    lr_tensors.append(scale * (lr_real + 1j * lr_imag))
                elif j == len(tensor_dims) - 1:
                    # last tensor
                    lr_real = jr.normal(rkey, (ranks[-1], d))
                    lr_imag = jr.normal(ikey, (ranks[-1], d))
                    lr_tensors.append(scale * (lr_real + 1j * lr_imag))
                else:
                    # intermediate tensors
                    lr_real = jr.normal(rkey, (ranks[j - 1], d, ranks[j]))
                    lr_imag = jr.normal(ikey, (ranks[j - 1], d, ranks[j]))
                    lr_tensors.append(scale * (lr_real + 1j * lr_imag))

        self.lr_tensors = tuple(lr_tensors)

        self.einsum_str = self._assemble_einsum_str()
        self.to_dense_einsum_str = self._assemble_dense_einsum_str()

    def _assemble_einsum_str(self) -> str:
        """Assembles the n-dim einsum contraction notation for weight-input contraction.

        The resulting einsum strings expects the low-rank tensors to be passed first
        and then the input slice, e.g.:

        ```python
        jnp.einsum(
            "<einsum_string>",
            *lr_tensors,
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
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2 - 1))])
            ch_i = indices[0]
            rank_i = indices[1 : self.ndim]
            spatial_i = indices[self.ndim :]

            # channel lr tensor
            ch_lr_str = ch_i + rank_i[0]
            # last spatial lr tensor
            ls_lr_str = rank_i[-1] + spatial_i[-1]
            # spatial lr tensors (except last)
            incs_lr_str = ",".join(
                [rank_i[i] + d + rank_i[i + 1] for i, d in enumerate(spatial_i[:-1])]
            )
            # input signal
            input_str = ch_i + spatial_i
            # output
            output_str = ch_i + spatial_i
            einsum_str = (
                f"{ch_lr_str},{incs_lr_str},{ls_lr_str},{input_str}->{output_str}"
            )
            return einsum_str
        else:
            # input x has shape (in_channels, d1, ..., dN)
            # lr_tensors have shapes:
            # (out_channels, r1), (r1, in_channels, r2), (r2, d1, r3), ..., (rD, dN)
            # need D-1 rank indices, N spatial dim indices and 2 channel indices
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2 - 1))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2 : 2 + self.ndim - 1]
            spatial_i = indices[-(self.ndim - 2) :]
            # out-channel lr tensor
            ch_lr_str = oc_i + rank_i[0]
            # last spatial lr tensor
            ls_lr_str = rank_i[-1] + spatial_i[-1]
            # in-channel and spatial lr tensors (except last)
            incs_lr_str = ",".join(
                [
                    rank_i[i] + d + rank_i[i + 1]
                    for i, d in enumerate(ic_i + spatial_i[:-1])
                ]
            )
            # input signal
            input_str = ic_i + spatial_i
            # output
            output_str = oc_i + spatial_i
            einsum_str = (
                f"{ch_lr_str},{incs_lr_str},{ls_lr_str},{input_str}->{output_str}"
            )
            return einsum_str

    def _assemble_dense_einsum_str(self) -> str:
        """Assembles the n-dim einsum contraction notation for dense contraction.

        The resulting einsum string expects the lr_tensors to be passed
        in order of creation.

        ```python
        jnp.einsum(
            "<einsum_string>",
            *lr_tensors
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
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2 - 1))])
            ch_i = indices[0]
            rank_i = indices[1 : self.ndim]
            spatial_i = indices[self.ndim :]

            # channel lr tensor
            ch_lr_str = ch_i + rank_i[0]
            # last spatial lr tensor
            ls_lr_str = rank_i[-1] + spatial_i[-1]
            # spatial lr tensors (except last)
            incs_lr_str = ",".join(
                [rank_i[i] + d + rank_i[i + 1] for i, d in enumerate(spatial_i[:-1])]
            )
            out_str = ch_i + spatial_i
            einsum_str = f"{ch_lr_str},{incs_lr_str},{ls_lr_str}->{out_str}"
            return einsum_str
        else:
            indices = "".join([chr(i) for i in range(97, 97 + (self.ndim * 2 - 1))])
            oc_i = indices[0]
            ic_i = indices[1]
            rank_i = indices[2 : 2 + self.ndim - 1]
            spatial_i = indices[-(self.ndim - 2) :]
            # out-channel lr tensor
            ch_lr_str = oc_i + rank_i[0]
            # last spatial lr tensor
            ls_lr_str = rank_i[-1] + spatial_i[-1]
            # in-channel and spatial lr tensors (except last)
            incs_lr_str = ",".join(
                [
                    rank_i[i] + d + rank_i[i + 1]
                    for i, d in enumerate(ic_i + spatial_i[:-1])
                ]
            )
            out_str = oc_i + ic_i + spatial_i
            einsum_str = f"{ch_lr_str},{incs_lr_str},{ls_lr_str}->{out_str}"
            return einsum_str

    def to_dense(self) -> Complex[Array, "n_corners ..."]:
        """Reconstruct factorized tensors into single dense tensor.

        Useful, e.g. if the `reconstructed` flag is `True`.

        Returns:
            Reconstructed dense tensor stacked along first axis.
        """
        sorted = [
            jnp.stack([t for t in self.lr_tensors[i :: self.ndim]], axis=0)
            for i in range(self.ndim)
        ]
        return jax.vmap(jnp.einsum, in_axes=(None,) + (0,) * self.ndim)(
            self.to_dense_einsum_str, *sorted
        )

    def __call__(
        self, corner_idx: int, x_ft_slice: Complex[Array, "in_c ..."]
    ) -> Complex[Array, "..."]:
        """Performs tensor train tensor contraction.

        Args:
            corner_idx: Corner index of the n-dim FFT hypercube.
            x_ft_slice: Mode slice of fourier transformed input.

        Returns:
            Tensor contraction of respective weight and input.
        """
        return jnp.einsum(
            self.einsum_str,
            *self.lr_tensors[
                corner_idx * self.ndim : corner_idx * self.ndim + self.ndim
            ],
            x_ft_slice,
        )
