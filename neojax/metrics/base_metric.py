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

"""Implementation of an abstract base metric class."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


class BaseMetric(eqx.Module):
    """Abstract base metric.

    New metrics and loss functions should inherit from BaseMetric.
    This class cannot be instantiated directly. Subclasses must be final
    following the "Abstract or Final" class design pattern.

    Every metric evaluates predictions against a ground-truth target. It can also
    act as a differentiable loss function during training. To support learnable
    weighting (e.g., in multi-task loss compositions), each metric carries a weight
    field that can optionally be flagged as learnable.

    Args:
        weight: (Learnable) scalar weight coefficient of the metric.
            The metric is computed as 'metric * weight'. Default is 1.0.
            If `learnable_weight` is True, weight has to be strictly positive and
            will be optimized via gradient descent.
        learnable_weight: Optional flag indicating whether `weight` is learnable.
            If `True`, this flag is used in the `is_learnable_metric_weight` filter function
            to indicate to `eqx.filter_...` or `eqx.partition` that `weight` should be
            adapted by the optimizer. Default is `False`.

    Raises:
        ValueError: If `learnable_weight` is `True` and `weight` is negative or zero.

    ??? info "Attributes"
        * **raw_weight** (`Float[Array, ""]`): Scalar weight coefficient for the metric.
            Used to balance multiple metrics/losses in compositions.
        * **learnable_weight** (`bool`): Static flag indicating whether the weight
            is learnable via gradient descent.

    !!! info "Why the raw weight is not learnable"
        A metric has a non-negative value. If the raw weight was learnable
        this would give the optimizer an easy way to drive the loss down:
        reduce the weight to zero or drive the weight negative
        which rewards the model to fit worse. To prevent such 'cheating', the `weight` is
        transformed by `jax.nn.softplus` which makes it strictly positive.
        An arbitrary fixed weight can still be set as long as `learnable_weight` is `False`.
    """

    raw_weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)

    def __init__(self, weight: float | int, learnable_weight: bool = False) -> None:
        self.learnable_weight = learnable_weight
        if learnable_weight:
            if weight <= 0:
                raise ValueError(
                    "weight must be strictly positive if learnable_weight is True."
                )
            self.raw_weight = jnp.asarray(jnp.log(jnp.expm1(weight)), dtype=jnp.float32)
        else:
            self.raw_weight = jnp.asarray(weight, dtype=jnp.float32)

    @property
    def weight(self) -> Float[Array, ""]:
        """The scalar weight. Strictly positive if `learnable_weight` is True."""
        if self.learnable_weight:
            return jax.nn.softplus(self.raw_weight)
        return self.raw_weight

    @staticmethod
    def batch_axes(x: PyTree[Any]) -> PyTree[int | None]:
        """Axes along which a batched model input is mapped (`jax.vmap`'s `in_axes`).

        Array leaves with at least one dimension are mapped over their leading batch
        dimension. `None`, Python scalars and zero-dimensional arrays (e.g. PRNG keys)
        are not mapped, so every sample receives them unchanged.

        Args:
            x: Batched model input.

        Returns:
            A pytree matching the structure of `x` with `0` or `None` as leaves.
        """
        return jax.tree.map(
            lambda leaf: 0 if eqx.is_array(leaf) and jnp.ndim(leaf) > 0 else None, x
        )

    @staticmethod
    def vmap_model(
        model: Callable[..., Any], x: PyTree[Any]
    ) -> PyTree[Float[Array, "b ..."]]:
        """Evaluates `model` on a batch of inputs.

        Follows the calling convention of the neojax evaluators. A `dict` is passed
        to the model as keyword arguments (`model(**x_i)`), a `tuple` or `list` as
        positional arguments (`model(*x_i)`) and anything else as the single
        positional argument (`model(x_i)`). The batch dimension is described by
        `batch_axes`.

        Args:
            model: The model, evaluated on single samples.
            x: Batched model input.

        Returns:
            The batched model output.
        """
        axes = BaseMetric.batch_axes(x)
        if isinstance(x, dict):
            return jax.vmap(lambda x_i: model(**x_i), in_axes=(axes,))(x)
        if isinstance(x, tuple | list):
            return jax.vmap(lambda x_i: model(*x_i), in_axes=(axes,))(x)
        return jax.vmap(model, in_axes=(axes,))(x)

    @abstractmethod
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: PyTree[Any] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes metric.

        Args:
            model: The neural operator model being evaluated. Default is None.
                Required for operator-dependent metrics (e.g., Sobolev metrics).
            target: Ground truth array of shape (batch, channels, *spatial).
            x: Input to the model. An array of shape (batch, in_channels, *spatial)
                or a pytree of batched inputs.
                Required for operator-dependent metrics.
            pred: Optional pre-computed model prediction array of shape
                (batch, channels, *spatial). If not provided, the metric will
                attempt to compute it by evaluating the model on `x`.
            **kwargs: Additional parameters for custom metrics.

        Returns:
            Scalar metric value.
        """
        ...
