"""Implementation of a n-dimensional Recurrent Neural Operator (RNO)."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, Inexact, PRNGKeyArray, PyTree
from typing_extensions import override

from neojax.models.baseno import BaseNO


def concat_channels(
    x: Inexact[Array, "c ..."],
    f_t: PyTree[Inexact[Array, "_ ..."]] | None,
) -> Inexact[Array, "..."]:
    """Concatenates the state and the per-step forcing along the channel axis.

    This is the default `combine_fn` of the `RNO`.
    It is for every model that takes a single multichannel input array
    (FNO, TFNO, UNO, MgNO, ...).

    Args:
        x: The current state of shape `(c, d1, ..., dN)`.
        f_t: Pytree of per-step forcing arrays, each of shape
            `(f_c, d1, ..., dN)`, or `None`.

    Returns:
        The state with all forcing leaves appended along the channel axis,
        that is shape `(c + sum(f_c), d1, ..., dN)`. If there is no forcing,
        `x` is returned unchanged.
    """
    leaves = jtu.tree_leaves(f_t)
    if not leaves:
        return x
    return jnp.concatenate([x, *leaves], axis=0)


@final
class RNO(BaseNO):
    r"""General Recurrent Neural Operator architecture.

    Essentially wraps another Neural Operator and applies
    it recursively to its outputs for $t$ timesteps over
    a time interval $[0, \dots, T]$.

    The recurrent step is model-agnostic. It handles three kinds of input:

    - the state $\hat{u}_n$, which is the `lax.scan` carry,
    - the per-step `forcing` $f_{n+1}$, scanned over its leading time axis,
    - `static` inputs (`key`, `inference` and any extra keyword arguments
      of the wrapped model), constant across steps and not scanned.

    How state and forcing are turned into the wrapped model's input is
    handled by `combine_fn`. The default concatenates them along the channel
    axis, which fits FNO, TFNO, UNO and any other model taking a single
    multichannel array. The wrapped model then needs
    `in_channels = state_channels + forcing_channels` and
    `out_channels = state_channels`.

    Args:
        model: A suitable model instance to wrap.
        time_interval: Time interval `[start, end]` of trajetory.
        n_timesteps: Number of timesteps in the time trajectory.
        combine_fn: Function that determines how model inputs and
            additional optional per-step inputs are combined.
            For most models (e.g. FNO-style, MgNO), this should be
            a concatenation along the channel dimension:
            `lambda x, forcing: jnp.concat((x, forcing), axis=0)`.
            This is also the default.
        checkpoint_timesteps: Whether to recompute linearizations
            at each autoregressive timestep. Can decrease memory
            at the cost of increased computation time.
            Default is `False`.

    Raises:
        ValueError: If `n_timesteps` is smaller 1.
        ValueError: If `time_interval` is invalid.

    ??? info "Internal Attributes"
        These fields store the internal layers state (and weights).

        * **model** (`BaseNO`): Suitable model instance to wrap.
        * **n_timesteps** (`int`): Number of timesteps in the time trajectory.
        * **dt** (`float`): Per step time delta.
        * **combine_fn** (`Callable[
            [Inexact[Array, "c ..."], PyTree[Inexact[Array, "_ ..."]] | None],
            Inexact[Array, "..."],
            ]`): Function that determines
            how model inputs and additional optional per-step inputs are combined.
        * **checkpoint_timesteps** (`bool`): Whether to recompute linearizations
            at each autoregressive timestep. Default is `False`.

    ??? cite

        [Recurrent Neural Operators: Stable Long-Term PDE
        Prediction](https://arxiv.org/pdf/2505.20721)

        ```bibtex
        @article{ye2025recurrent,
            title={Recurrent neural operators: stable long-term PDE prediction},
            author={Ye, Zaijun and Zhang, Chen-Song and Wang, Wansheng},
            journal={arXiv preprint arXiv:2505.20721},
            year={2025}
        }
        ```

    !!! info "Time and Memory complexity"
        Computation time and memory consumption increase with the number
        of timesteps $t$. Even with `checkpoint_timesteps`, expect time per epoch
        and memory usage to be far greater than for a non-recurrent model.
    """

    model: BaseNO
    n_timesteps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    combine_fn: Callable[
        [Inexact[Array, "c ..."], PyTree[Inexact[Array, "_ ..."]] | None],
        Inexact[Array, "..."],
    ] = eqx.field(static=True)
    checkpoint_timesteps: bool = eqx.field(static=True)

    def __init__(
        self,
        model: BaseNO,
        time_interval: tuple[float | int, float | int],
        n_timesteps: int,
        combine_fn: Callable[
            [Inexact[Array, "c ..."], PyTree[Inexact[Array, "f ..."]] | None],
            Inexact[Array, "..."],
        ] = concat_channels,
        checkpoint_timesteps: bool = False,
    ) -> None:
        self.model = model
        if n_timesteps < 2:
            raise ValueError("Invalid `n_timesteps`. Must use 2 or more timesteps.")
        self.n_timesteps = n_timesteps
        self.checkpoint_timesteps = checkpoint_timesteps
        if time_interval[0] < 0 or time_interval[-1] <= time_interval[0]:
            raise ValueError(
                "Invalid `time_interval`. Must be positive, increasing interval."
            )
        self.dt = (time_interval[-1] - time_interval[0]) / n_timesteps
        self.combine_fn = combine_fn

    def step(
        self,
        x: Inexact[Array, "c ..."],
        f_t: PyTree[Inexact[Array, "_ ..."]] | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
        **static: Any,
    ) -> Inexact[Array, "c ..."]:
        r"""Computes a single timestep.

        Given initial $u_0$ and optional per-step forcing $f_{n+1}$,
        this computes the explicit Euler update

        $$
        \hat{u}_{n+1} = \hat{u}_{n} + \Delta t \, \mathcal{G}_{\theta}(\hat{u}_{n}, f_{n+1}),
         \; n = 0, 1, \dots, N - 1.
        $$

        Args:
            x: Current state of shape `(in_c, d1, ..., dN)`,
                where N are the number of spatial dimensions.
            f_t: Pytree of this step's forcing, passed to `combine_fn`
                together with `x`. Default is `None`.
            key: PRNG key used for the wrapped model.
                Default is `None`.
            inference: If `True`, training-only logic of the wrapped model
                is disabled. Default is `False`.
            static: Static extra keyword arguments passed unchanged to the
                wrapped model (e.g. `GeoFNO` coordinates).

        Returns:
            The state $\hat{u}_{n+1}$ of the same shape as `x`.

        Raises:
            ValueError: If model output's shape of step n doesn't match
                the model input's shape of step n + 1.
        """
        wrapped_model = self.model(
            self.combine_fn(x, f_t), key=key, inference=inference, **static
        )
        if wrapped_model.shape != x.shape:
            raise ValueError(
                f"Model output shape {wrapped_model.shape} doesn't match "
                f"model input shape {x.shape} of next step."
            )
        return x + self.dt * wrapped_model

    @override
    def __call__(
        self,
        x: Inexact[Array, "c ..."],
        forcing: PyTree[Inexact[Array, "t _ ..."]] | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = False,
        **static: Any,
    ) -> Inexact[Array, "t c ..."]:
        r"""Forward pass of the RNO.

        Applies `step` recursively with given initial $\hat{u}_0$
        and optional per-step parameters.
        The outputs across all timesteps are stacked
        into a single output with leading time axis.

        Args:
            x: Initial state of shape `(c, d1, ..., dN)`,
                where N are the number of spatial dimensions.
            forcing: Pytree of per-step forcing arrays, each of shape
                `(n_timesteps, f_c, d1, ..., dN)`. Slice `t` is used at step `t`.
                Time-constant fields that should be concatenated to the input
                need to be broadcast along the time axis, e.g. with
                `neojax.utils.tree_add_broadcast_to`.
                Default is `None`, no forcing.
            key: PRNG key used for the wrapped model. Split into one key per timestep.
                Default is `None`.
            inference: If `True`, training-only logic of the wrapped model
                is disabled. Default is `False`.
            static: Static extra keyword arguments passed unchanged to
                the wrapped model at every step (e.g. `GeoFNO` coordinates).
                They are not scanned over.

        Returns:
            The predicted field of shape `(n_timesteps, out_channels, d1, ..., dN)`.

        Raises:
            ValueError: If any leaf of the `forcing` tree doesn't have a leading
                time axis of size `n_timesteps`.
        """
        for leaf in jtu.tree_leaves(forcing):
            if leaf.shape[0] != self.n_timesteps:
                raise ValueError(
                    "Every `forcing` leaf needs a leading time axis of size "
                    f"{self.n_timesteps}, but got shape {leaf.shape}."
                )
        keys = None if key is None else jr.split(key, self.n_timesteps)

        def scan_step(carry, xs):
            f_t, key_t = xs
            x_next = self.step(carry, f_t, key=key_t, inference=inference, **static)
            return x_next, x_next

        if self.checkpoint_timesteps:
            scan_step = jax.checkpoint(scan_step)

        _, traj = lax.scan(scan_step, x, (forcing, keys), length=self.n_timesteps)
        return traj
