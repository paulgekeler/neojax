"""Implementations of physical consistency metrics."""

from collections.abc import Callable
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing_extensions import override

from neojax.metrics.base_metric import BaseMetric


@final
class BoundaryConsistencyMetric(BaseMetric):
    r"""Boundary consistency metric.

    Evaluates the prediction alignment with the boundary condition values
    at the locations specified by the boundary condition mask.

    $$
    \text{BoundaryConsistency} = \left( \frac{\sum_{t, c, x} |(\hat{y}_{t,c,x} - y_{\text{bc}, t,c,x}) \cdot M_{c,x}|^p}{\sum_{t, c, x} M_{c,x} + \epsilon} \right)^{1/p}
    $$

    where $y_{\text{bc}}$ is the boundary values, $\hat{y}$ is the prediction,
    $M$ is the boundary mask, and $\epsilon = 10^{-7}$ is a small regularization constant.

    Args:
        p: Power of the error norm. Default is 2.0.
        weight: (Learnable) weight. Metric is computed as `weight` * `metric`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_metric_weight` utility function.
            Default is `False`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **p** (`float`): Power of the norm.
    """

    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)
    p: float = eqx.field(static=True)

    def __init__(
        self,
        p: float = 2.0,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.p = p
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."] | None = None,
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        bc_masks: Int[Array, "b c_bc ..."] | None = None,
        bc_values: Float[Array, "b t c_bc ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the boundary consistency metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Unused in boundary consistency metric.
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array shaped (batch, t, c, d1, ..., dN).
                If None, the metric computes `vmap(model)(x)`.
            bc_masks: Boundary mask array shaped (batch, c_bc, d1, ..., dN).
            bc_values: Boundary values array shaped (batch, t, c_bc, d1, ..., dN).
            **kwargs: Additional arguments.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If prediction is None and either model or x is None.
        """
        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "BoundaryConsistencyMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        if bc_masks is None or bc_values is None:
            # If no boundary conditions are defined, consistency is 0
            return jnp.array(0.0)

        def single_metric(p_i, mask_i, val_i):
            c_bc = mask_i.shape[0]
            pred_bc = p_i[:, :c_bc, ...]
            diff = (pred_bc - val_i) * mask_i[None, :, ...]
            t_steps = p_i.shape[0]
            mask_sum = jnp.sum(mask_i) * t_steps
            sum_err = jnp.sum(jnp.pow(jnp.abs(diff), self.p))
            return jnp.pow(sum_err / (mask_sum + 1e-7), 1 / self.p)

        batch_metric = jax.vmap(single_metric)(pred, bc_masks, bc_values)
        return self.weight * jnp.mean(batch_metric)


@final
class ConservationMetric(BaseMetric):
    r"""Conservation law metric.

    Evaluates the drift/deviation of a conserved physical quantity (like mass or energy)
    over a time trajectory relative to the initial state.

    $$
    \text{Drift} = \left( \frac{1}{T-1} \sum_{t=1}^{T-1} D_t^p \right)^{1/p}
    $$

    where the deviation at time step $t$ is:

    $$
    D_t = \begin{cases}
    |Q_t - Q_0|, & \text{if mode is absolute}, \\\\
    \frac{|Q_t - Q_0|}{|Q_0| + \epsilon}, & \text{if mode is relative}.
    \end{cases}
    $$

    and $Q_t = \text{conservation_fn}(y_t, x_{\text{coords}})$ is the quantity at time step $t$.

    Args:
        conservation_fn: Callable taking `(fields, coords)` and returning a scalar.
            `fields` is shaped `(c, *spatial)` and `coords` is shaped `(d, *spatial)`.
        mode: Type of drift deviation, either `"absolute"` or `"relative"`.
            Default is `"absolute"`.
        p: Power of the norm. Default is 2.0.
        weight: (Learnable) weight. Metric is computed as `weight` * `metric`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_metric_weight` utility function.
            Default is `False`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **conservation_fn** (`Callable`): Conserved quantity evaluator.
        * **mode** (`Literal["absolute", "relative"]`): Drift deviation mode.
        * **p** (`float`): Power of the norm.

    !!! info
        May be used to ensure the model doesn't violate conservation of energy or mass.
    """

    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)
    conservation_fn: Callable[
        [Float[Array, "c ..."], Float[Array, "d ..."]], Float[Array, ""]
    ] = eqx.field(static=True)
    mode: Literal["absolute", "relative"] = eqx.field(static=True)
    p: float = eqx.field(static=True)

    def __init__(
        self,
        conservation_fn: Callable[
            [Float[Array, "c ..."], Float[Array, "d ..."]], Float[Array, ""]
        ],
        mode: Literal["absolute", "relative"] = "absolute",
        p: float = 2.0,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.conservation_fn = conservation_fn
        if mode not in ("absolute", "relative"):
            raise ValueError("mode must be 'absolute' or 'relative'")
        self.mode = mode
        self.p = p
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."] | None = None,
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        coords: Float[Array, "d ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the conservation law metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Unused in conservation metric.
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array shaped (batch, t, c, d1, ..., dN).
                If None, the metric computes `vmap(model)(x)`.
            coords: Physical coordinates shaped (d, d1, ..., dN).
            **kwargs: Additional arguments.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `coords` is None.
            ValueError: If prediction is None and either model or x is None.
        """
        if coords is None:
            raise ValueError("ConservationMetric requires physical 'coords' in kwargs.")

        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "ConservationMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_metric(p_i, coords_val):
            # q shape: [t]
            q = jax.vmap(lambda f_t: self.conservation_fn(f_t, coords_val))(p_i)
            q_0 = q[0]
            q_t = q[1:]

            if self.mode == "relative":
                deviation = jnp.abs(q_t - q_0) / (jnp.abs(q_0) + 1e-7)
            else:
                deviation = jnp.abs(q_t - q_0)

            return jnp.pow(jnp.mean(jnp.pow(deviation, self.p)), 1 / self.p)

        batch_metric = jax.vmap(single_metric, in_axes=(0, None))(pred, coords)
        return self.weight * jnp.mean(batch_metric)


@final
class ResidualMetric(BaseMetric):
    r"""PDE residual metric.

    Evaluates the norm of the PDE residual using the user-provided residual function.

    $$
    \text{Residual} = \left( \frac{1}{N_{\text{res}}} \sum_{t, c, x} |R(\hat{y}, x_{\text{coords}})|^p \right)^{1/p}
    $$

    where $R$ is the residual evaluator.

    Args:
        residual_fn: Callable taking `(fields, coords)` and returning residual fields.
            `fields` is prediction trajectory shaped `(t, c, *spatial)` and `coords`
            is shaped `(d, *spatial)`. Output must have shape `(t_res, c_res, *spatial_res)`.
        p: Power of the norm. Default is 2.0.
        weight: (Learnable) weight. Metric is computed as `weight` * `metric`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_metric_weight` utility function.
            Default is `False`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **residual_fn** (`Callable`): Residual field evaluator.
        * **p** (`float`): Power of the norm.
    """

    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)
    residual_fn: Callable[
        [Float[Array, "t c ..."], Float[Array, "d ..."]], Float[Array, "..."]
    ] = eqx.field(static=True)
    p: float = eqx.field(static=True)

    def __init__(
        self,
        residual_fn: Callable[
            [Float[Array, "t c ..."], Float[Array, "d ..."]], Float[Array, "..."]
        ],
        p: float = 2.0,
        weight: float = 1.0,
        learnable_weight: bool = False,
    ) -> None:
        self.residual_fn = residual_fn
        self.p = p
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."] | None = None,
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        coords: Float[Array, "d ..."] | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the PDE residual metric.

        Args:
            model: The model being trained. Optional if `pred` is given.
            target: Unused in residual metric.
            x: Model input array shaped (batch, in_c, d1, ..., dN).
            pred: Model prediction array shaped (batch, t, c, d1, ..., dN).
                If None, the metric computes `vmap(model)(x)`.
            coords: Physical coordinates shaped (d, d1, ..., dN).
            **kwargs: Additional arguments.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If `coords` is None.
            ValueError: If prediction is None and either model or x is None.
        """
        if coords is None:
            raise ValueError("ResidualMetric requires physical 'coords' in kwargs.")

        if pred is None:
            if model is None or x is None:
                raise ValueError(
                    "ResidualMetric requires either 'pred' or both 'model' and 'x'."
                )
            pred = jax.vmap(model)(x)

        def single_metric(p_i, coords_val):
            residual = self.residual_fn(p_i, coords_val)
            return jnp.pow(jnp.mean(jnp.pow(jnp.abs(residual), self.p)), 1 / self.p)

        batch_metric = jax.vmap(single_metric, in_axes=(0, None))(pred, coords)
        return self.weight * jnp.mean(batch_metric)
