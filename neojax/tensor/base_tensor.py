# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation on an abstract base tensor."""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float


class BaseTensor(eqx.Module):
    """Abstract base tensor.

    All neojax tensors subclass `BaseTensor`.
    Custom tensor classes should inherit from `BaseTensor`.

    !!! warning "Store learnable weights as real arrays"
        For complex parameters, `jax.grad` returns the conjugate of the descent
        direction, so passing such gradients to an optax optimizer moves the imaginary
        parts in the wrong direction. All neojax tensors therefore store
        real `(real, imag)` pairs along a last axis of size 2 (see `to_real`)
        and assemble the complex weights when they are used (see `to_complex`).
        Custom tensors should do the same.
    """

    @staticmethod
    def to_real(z: Complex[Array, "..."]) -> Float[Array, "... 2"]:
        """Decomposes a complex tensor into real array of stacked `(real, imag)` pairs.

        Stacks real parts along the last axis.

        Args:
            z: Complex weight tensor.

        Returns:
            The stacked real and imaginary parts.
        """
        return jnp.stack([z.real, z.imag], axis=-1)

    @staticmethod
    def to_complex(
        w: Float[Array, "... 2"],
    ) -> Complex[Array, "..."]:
        """Creates a complex tensor out of its real components.

        JAX has no half-precision complex type, so half-precision pairs
        (`float16`, `bfloat16`) are promoted and result in `complex64` array.

        Args:
            w: Real weight tensor with real and imaginary parts
                stacked along last axis.

        Returns:
            The complex tensor without last axis.
        """
        w = w.astype(jnp.promote_types(w.dtype, jnp.float32))
        return jax.lax.complex(w[..., 0], w[..., 1])

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
