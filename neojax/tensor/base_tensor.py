"""Implementation on an abstract base tensor."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Complex


class BaseTensor(eqx.Module):
    """Abstract base tensor.

    All neojax tensors subclass `BaseTensor`.
    Custom tensor classes should inherit from `BaseTensor`.
    """

    @abstractmethod
    def to_dense(self) -> Complex[Array, "n_corners ..."]:
        """Converts tensors to a dense tensor representation.

        Returns:
            Dense tensors stacked along first axis.
        """
        ...

    @abstractmethod
    def __call__(
        self, corner_idx: int, x_ft_slice: Complex[Array, "in_c ..."]
    ) -> Complex[Array, "out_c ..."]:
        """Contracts the given input slice with the tensor weights for the given corner.

        Args:
            corner_idx: Corner index of the n-dim FFT hypercube.
            x_ft_slice: Mode slice of fourier transformed input.

        Returns:
            Tensor contraction of weight and input.
        """
        ...
