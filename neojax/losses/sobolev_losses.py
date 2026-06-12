"""Implementations of various Sobolev losses."""

from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import jet
from jaxtyping import Array, Float, PRNGKeyArray

from neojax.losses.base_loss import BaseLoss


@final
class SobolevLoss(BaseLoss):
    r"""General (Sobolev) $W^{k,p}$-loss.

    Computes the loss in the $W^{k,p}$ Sobolev space with the associated norm:
    $$
    \begin{aligned}
    \Vert (y - \hat{y}) \Vert_{W^{k,p}(\Omega)} &=
    \begin{cases}
    \left(\sum_{\vert \alpha \vert \le k} \Vert D^\alpha (y - \hat{y}) \Vert_{L^p(\Omega)}^p\right)^{1/p},
        & \text{if } p < \infty, \\\\
    \max_{\vert \alpha \vert \le k} \Vert D^\alpha (y - \hat{y}) \Vert_{L^\infty(\Omega)},
        & \text{if } p = \infty.
    \end{cases}
    \end{aligned}
    $$
    where $D^{\alpha}(y - \hat{y})$ are all combinations
    of partial derivatives up to the $k$-th order and
    $y$ is the ground truth and $\hat{y}$ is the model prediction.

    Default is the $W^{1,2}$-loss (known as $H^1$-loss).

    Args:
        k: Order of partial derivatives. Default is 1.
        p: Power of the $L^{p}$-norm. Default is 2.0.
        method: Method for computing higher order derivatives.
            See documentation for details. Default is `auto`.
        diff_mode: Inner AD mode. See documentation for details.
            Default is `auto`.
        weight: (Learnable) weight. Loss is computed as `weight` * `loss`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_loss_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.
        n_random_samples: Number of random vectors to use in
            `"stochastic"` method. Ignored in other methods.
            Default is 10.
        random_type: Type of distribution to sample from
            in `"stochastic"` method.
            Default is standard Normal distribution.

    !!! info "Internal Attributes"
        These fields store the internal state of the loss.

        * **k** (`int`): Order of partial derivatives.
        * **p** (`float`): Power of the $L^{p}$-norm.
        * **method** (`Literal["exact", "interpolate", "stochastic", "auto"]`): Method for computing higher order derivatives. See documentation for details.
        * **diff_mode** (`Literal["fwd", "bwd", "auto"]`): Inner AD mode. See documentation for details.
        * **weight** (`Float[Array, ""]`): Learnable loss weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **n_random_samples** (`int`): Number of random vectors to sample.
        * **random_type** (`Literal["rademacher", "normal"]`): Type of distribution to sample from in `"stochastic"` method. Default is standard Normal distribution.

    !!! info
        Computing higher order derivates is expensive. Given a function
        $f: \mathbb{R}^m \rightarrow \mathbb{R}^d $,
        computing the derivatives up to order k, the XLA graph
        scales with $\mathcal{O}(m \cdot d^k)$.
    """

    k: int = eqx.field(static=True)
    p: float = eqx.field(static=True)
    method: Literal["exact", "interpolate", "stochastic", "auto"] = eqx.field(
        static=True
    )
    diff_mode: Literal["fwd", "bwd", "auto"] = eqx.field(static=True)
    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)
    n_random_samples: int = eqx.field(static=True)
    random_type: Literal["rademacher", "normal"] = eqx.field(static=True)

    def __init__(
        self,
        k: int = 1,
        p: float = 2.0,
        method: Literal["exact", "interpolate", "stochastic", "auto"] = "auto",
        diff_mode: Literal["fwd", "bwd", "auto"] = "auto",
        weight: float = 1.0,
        learnable_weight: bool = False,
        n_random_samples: int = 10,
        random_type: Literal["rademacher", "normal"] = "normal",
    ) -> None:
        self.k = k
        self.p = p
        methods = ["exact", "interpolate", "stochastic", "auto"]
        diff_modes = ["fwd", "bwd", "auto"]
        if method not in methods:
            raise ValueError(f"Invalid method. Must be one of {methods}.")
        self.method = method
        if diff_mode not in diff_modes:
            raise ValueError(
                f"Invalid differentiation mode. Must be one of {diff_modes}."
            )
        self.diff_mode = diff_mode
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight
        self.n_random_samples = n_random_samples
        random_types = ["rademacher", "normal"]
        if random_type not in random_types:
            raise ValueError(
                f"Invalid distribution type. Must be one of {random_types}."
            )
        self.random_type = random_type

    def exact_sobolev(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, ""]:
        """Computes the Sobolev loss exactly for a single sample.

        Uses either `eqx.filter_jacrev` or `eqx.filter_jacfwd`
        to compute the exact derivative w.r.t. `x`,
        depending on `diff_mode`.

        If `diff_mode` is `auto`, uses forward-mode AD if
        all output dimensions are larger or the same as
        all input dimensions, else backward-mode AD.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).

        Returns:
            Scalar Sobolev loss.
        """
        if self.diff_mode == "bwd":
            jac = eqx.filter_jacrev(model)(x)
        elif self.diff_mode == "fwd":
            jac = eqx.filter_jacfwd(model)(x)
        else:  # diff_mode == "auto"
            num_in = x.size
            num_out = target.size
            if num_in <= num_out:
                # output dims larger or same as input dims
                jac = eqx.filter_jacfwd(model)(x)
            else:
                # output dims smaller than input dims
                jac = eqx.filter_jacrev(model)(x)

        fun_loss = jnp.mean(jnp.pow(jnp.abs(target - model(x)), self.p))
        deriv_loss = jnp.mean(jnp.pow(jnp.abs(jac), self.p))
        return jnp.pow(fun_loss + deriv_loss, 1 / self.p)

    def _sobolev_from_directions(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
        directions: Float[Array, "d in_c ..."],
    ) -> Float[Array, ""]:
        """Computes the Sobolev loss for a single sample in several directions.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).
            directions: Directions in which to compute derivatives.
                Shape: (num_directions, in_c, d1, ..., dN).

        Returns:
            Scalar Sobolev loss.
        """
        shared_zeros = [jnp.zeros_like(x) for _ in range(self.k - 1)]

        def directional_jet(direction, zeros):
            series_list = [direction] + list(zeros)

            primals_out, series_out = jet.jet(
                model, (x,), (series_list,), factorial_scaled=False
            )
            return target - primals_out, series_out

        # vmap over directions to get derivatives
        primals, series = jax.vmap(directional_jet, in_axes=(0, None))(
            directions, shared_zeros
        )
        series = jnp.stack(series, axis=1)
        # primals are function values, series are Taylor coefficients
        # primals shape: (N_total, m), series shape: (N_total, self.k, m)
        func_term = jnp.mean(jnp.pow(jnp.abs(primals), self.p))
        # mean over each individual derivative, then sum over derivatives
        mean_axes = (0,) + tuple(range(2, series.ndim))
        deriv_term = jnp.sum(jnp.mean(jnp.pow(jnp.abs(series), self.p), axis=mean_axes))
        lp_mean_sum = func_term + deriv_term
        return jnp.pow(lp_mean_sum, 1 / self.p)

    def interpolated_sobolev(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, ""]:
        """Compute Sobolev loss for a single sample by propagating a higher order Taylor polynomial.

        Uses `jax.experimental.jet`, which forwards a higher order
        Taylor polynomial. This is more efficient than the exact method
        for higher order derivatives.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).

        Returns:
            Scalar Sobolev loss.
        """
        total_size = x.size
        flat_id = jnp.eye(total_size, dtype=x.dtype)
        # basis tensors shaped (n_total, *pred_shape)
        basis_tensors = jnp.reshape(flat_id, shape=(total_size,) + x.shape)
        return self._sobolev_from_directions(model, x=x, target=target, directions=basis_tensors)

    def stochastic_sobolev(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Computes the stochastic Sobolev loss.

        Uses the STDE approach, making `jet` more
        efficient by sampling random vectors instead
        of using all basis tensors.

        ??? cite
            [Stochastic Taylor Derivative Estimator: Efficient
            amortization for arbitrary differential operators](https://proceedings.neurips.cc/paper_files/paper/2024/file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf)

            ```bibtex
            @article{shi2024stochastic,
                title={Stochastic taylor derivative estimator:
                Efficient amortization for arbitrary differential operators},
                author={Shi, Zekun and Hu, Zheyuan and Lin, Min and Kawaguchi, Kenji},
                journal={Advances in Neural Information Processing Systems},
                volume={37},
                pages={122316--122353},
                year={2024}
            }
            ```

        Args:
            model: The model being trained.
            x: Model input array shaped (c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).
            key: Random key.

        Returns:
            Scalar Sobolev loss.
        """
        if self.random_type == "rademacher":
            random_vectors = jr.rademacher(
                key=key, shape=(self.n_random_samples,) + x.shape, dtype=x.dtype
            )
        else:
            random_vectors = jr.normal(
                key=key, shape=(self.n_random_samples,) + x.shape, dtype=x.dtype
            )
        return self._sobolev_from_directions(model, x=x, target=target, directions=random_vectors)

    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ) -> Float[Array, ""]:
        """Computes the general Sobolev loss.

        Args:
            model: The model being trained.
            target: Ground truth array shaped (b, c, d1, ..., dN).
            x: Model input array shaped (b, in_c, d1, ..., dN)
            pred: Unused in Sobolev loss.
            key: Optional random key needed for stochastic Sobolev.
                Default is `None`.
            **kwargs: Additional arguments for specific losses.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If method is `"stochastic"` but no random key
                was passed. Or if `model` or `x` are not provided.

        !!! info
            Higher order (partial) derivatives are expensive.
            We therefore use both exact derivatives for lower order
            losses and inexact stochastic Sobolev losses for higher order k.
        """
        if model is None or x is None:
            raise ValueError(
                "SobolevLoss requires both 'model' and 'x' (inputs) "
                "to compute derivatives."
            )

        strategy = self.method
        if strategy == "auto":
            if self.k == 1:
                strategy = "exact"
            elif self.k >= 2 and not any(p > 10 for p in x.shape[2:]):
                strategy = "interpolate"
            else:
                strategy = "stochastic"
        if strategy == "stochastic" and key is None:
            raise ValueError("Stochastic Sobolev loss requires a random key.")

        def single_loss(x_i, target_i, key_i):
            if strategy == "exact":
                return self.exact_sobolev(model, x=x_i, target=target_i)
            elif strategy == "interpolate":
                return self.interpolated_sobolev(model, x=x_i, target=target_i)
            else:
                return self.stochastic_sobolev(model, x=x_i, target=target_i, key=key_i)

        batch_size = x.shape[0]
        if strategy == "stochastic":
            keys = jr.split(key, batch_size)
            batch_losses = jax.vmap(single_loss)(x, target, keys)
        else:

            def single_loss_no_key(x_i, target_i):
                return single_loss(x_i, target_i, None)

            batch_losses = jax.vmap(single_loss_no_key)(x, target)

        return self.weight * jnp.mean(batch_losses)
