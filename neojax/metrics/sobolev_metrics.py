"""Implementations of various Sobolev metrics."""

import math
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import jet
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import override

from neojax.metrics.base_metric import BaseMetric
from neojax.models.fno import FNO
from neojax.models.geo_fno import GeoFNO
from neojax.models.tfno import TFNO
from neojax.models.uno import UNO


@final
class SobolevMetric(BaseMetric):
    r"""General (Sobolev) $W^{k,p}$-metric.

    Computes the metric in the $W^{k,p}$ Sobolev space with the associated norm:

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

    Default is the $W^{1,2}$-metric (known as $H^1$-metric).

    Args:
        k: Order of partial derivatives. Default is 1.
        p: Power of the $L^{p}$-norm. Default is 2.0.
        method: Method for computing derivatives.
            See documentation for details. Default is `auto`.
        diff_mode: Inner AD mode for first order derivates.
            See documentation for details. Default is `auto`.
        weight: (Learnable) weight. Metric is computed as `weight` * `metric`.
            Default is 1.0.
        learnable_weight: Whether `weight` is learnable.
            Used to filter trainable parameters using
            `is_learnable_metric_weight` utility function
            with `equinox.filter_...` or `equinox.partition`.
            Default is `False`.
        n_random_samples: Number of random vectors to sample (without replacement) in
            `"stochastic"` method. Ignored in other methods.
            Default is 10.
        direction_threshold: Maximum number of basis tensors for derivative seeding
            until "stochastic" method is used. Default is 32.
            Only change if needed.

    Raises:
        ValueError: If the order of derivatives is negative or zero.
        ValueError: If the `method` or `diff_mode` is invalid.
        ValueError: If the `direction_threshold` is negative or zero.
        ValueError: If k > 1 and the method is `"exact"`.

    ??? info "Internal Attributes"
        These fields store the internal state of the metric.

        * **k** (`int`): Order of partial derivatives.
        * **p** (`float`): Power of the $L^{p}$-norm.
        * **method** (`Literal["exact", "interpolate", "stochastic", "auto"]`): Method for computing
            higher order derivatives. See documentation for details.
        * **diff_mode** (`Literal["fwd", "bwd", "auto"]`): Inner AD mode. See documentation for details.
        * **weight** (`Float[Array, ""]`): Learnable metric weight. Filter during training to prevent updates.
        * **learnable_weight** (`bool`): Flag indicating whether `weight` is learnable.
        * **n_random_samples** (`int`): Number of random vectors to sample.
        * **direction_threshold** (`int`): Maximum number of basis tensors for derivative seeding
            until "stochastic" method is used. Default is 1024.

    !!! warning
        Computing higher-order derivatives is expensive. Given a function
        $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ the k-th derivative has
        $m \cdot n^k$ components, so the size of the derivative representation can
        grow as $\mathcal{O}(m\cdot n^k)$. The actual computational and memory cost
        also depends on the model and differentiation strategy. Only change
        `method` if you know what you are doing; otherwise, it's a fast way to
        run out of memory.

    !!! warning "Unsupported primitive rules for jet"
        `jax.experimental.jet` doesn't yet implement rules for all jax primitives.
        If a model uses a primitive without corresponding jet rule
        (e.g. `neojax.models.GeoFNO` uses `jax.lax.scatter`), the methods
        for computing higher order partial derivatives with `jax.experimental.jet`
        (`interpolated_sobolev` and `stochastic_sobolev_stde`) are not available for that model.
        This condition raises a `KeyError` containing the name of the primitive if encountered.
        This is a limitation of `jax.experimental.jet`'s current implementation.
    """

    k: int = eqx.field(static=True)
    p: float | int = eqx.field(static=True)
    method: Literal["exact", "interpolate", "stochastic", "auto"] = eqx.field(
        static=True
    )
    diff_mode: Literal["fwd", "bwd", "auto"] = eqx.field(static=True)
    weight: Float[Array, ""]
    learnable_weight: bool = eqx.field(static=True)
    n_random_samples: int = eqx.field(static=True)
    direction_threshold: int = eqx.field(static=True)

    def __init__(
        self,
        k: int = 1,
        p: float | int = 2.0,
        method: Literal["exact", "interpolate", "stochastic", "auto"] = "auto",
        diff_mode: Literal["fwd", "bwd", "auto"] = "auto",
        weight: float = 1.0,
        learnable_weight: bool = False,
        n_random_samples: int = 10,
        direction_threshold: int = 1024,
    ) -> None:
        if k <= 0:
            raise ValueError("Order of derivatives cannot be negative or zero.")
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
        if k > 1 and method == "exact":
            raise ValueError(
                "The 'exact' method derivative is not available for higher order derivatives."
            )

        self.diff_mode = diff_mode
        self.weight = jnp.array(weight)
        self.learnable_weight = learnable_weight
        self.n_random_samples = n_random_samples
        if direction_threshold <= 0:
            raise ValueError("'direction_threshold' must be larger than zero.")
        self.direction_threshold = direction_threshold

    def exact_sobolev(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
    ) -> Float[Array, ""]:
        """Computes the first order Sobolev metric exactly.

        Uses either `eqx.filter_jacrev` or `eqx.filter_jacfwd`
        to compute the exact derivative w.r.t. `x`,
        depending on `diff_mode`.

        If `diff_mode` is `auto`, uses forward-mode AD
        if the output size is larger
        or the same as the input size, else backward-mode AD.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).

        Returns:
            Scalar Sobolev metric.
        """
        if self.diff_mode == "bwd":
            jac = eqx.filter_jacrev(model)(x)
        elif self.diff_mode == "fwd":
            jac = eqx.filter_jacfwd(model)(x)
        else:  # diff_mode == "auto"
            if x.size <= target.size:
                # Output dims larger or same as input dims
                jac = eqx.filter_jacfwd(model)(x)
            else:
                # Output dims smaller than input dims
                jac = eqx.filter_jacrev(model)(x)

        fun_metric = jnp.mean(jnp.pow(jnp.abs(target - model(x)), self.p))
        deriv_metric = jnp.mean(jnp.pow(jnp.abs(jac), self.p))
        return jnp.pow(fun_metric + deriv_metric, 1 / self.p)

    def _sobolev_jet_from_directions(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
        directions: Float[Array, "d in_c ..."],
    ) -> Float[Array, ""]:
        """Computes the Sobolev metric using jax.experimental.jet for a single sample in several directions.

        Used by `stochastic_sobolev_stde` and `interpolated_sobolev`
        with random or full basis vectors/tensors respectively.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).
            directions: Directions along which to compute derivatives.
                Shape: (num_directions, in_c, d1, ..., dN).

        Returns:
            Scalar Sobolev metric.
        """
        shared_zeros = [jnp.zeros_like(x) for _ in range(self.k - 1)]

        def directional_jet(direction):
            # jet's j-th series output is the coefficient of t^(j+1) in the
            # Taylor expansion of model along the input curve, i.e.
            # D^(j+1)model(x)[v,...,v] / (j+1)!, not the raw (j+1)-th order
            # Frechet derivative itself. Rescale by the factorial to recover it,
            # regardless of the factorial_scaled argument.
            order_factorials = [math.factorial(j + 1) for j in range(self.k)]

            # series_list: [e_j, 0, 0, ...] isolates d^k u / dx_j^k
            series_list = [direction] + shared_zeros

            primals_out, series_out = jet.jet(
                fun=model, primals=(x,), series=(series_list,), factorial_scaled=False
            )
            series_out = [
                term * fact
                for term, fact in zip(series_out, order_factorials, strict=True)
            ]
            return target - primals_out, series_out

        # vmap over directions to get derivatives
        primals, series = jax.vmap(directional_jet)(directions)
        series = jnp.stack(series, axis=1)
        # Primals are function values, series are Taylor coefficients
        # primals shape: (N_total, m), series shape: (N_total, self.k, m)
        func_term = jnp.mean(jnp.pow(jnp.abs(primals), self.p))
        # Mean over each individual derivative, then sum over derivatives
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
        """Computes Sobolev metric for higher orders by propagating a higher order Taylor polynomial.

        Uses `jax.experimental.jet`, which forwards a higher order
        Taylor polynomial. This is more efficient than the exact method
        for higher order derivatives.

        Args:
            model: The model being trained.
            x: Model input array shaped (in_c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).

        Returns:
            Scalar Sobolev metric.
        """
        total_size = x.size
        flat_id = jnp.eye(total_size, dtype=x.dtype)
        # Basis tensors shaped (n_total, *pred_shape)
        basis_tensors = jnp.reshape(flat_id, shape=(total_size,) + x.shape)

        return self._sobolev_jet_from_directions(
            model, x=x, target=target, directions=basis_tensors
        )

    def stochastic_sobolev_first_order(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Computes the stochastic Sobolev metric for first order derivatives.

        Uses a stochastic estimation approach inspired by sparse STDE in
        `stochastic_sobolev_stde` by seeding the JVP/VJP
        with a random sparse subset of basis vectors. This yields
        an exact subset of exact columns/rows of the Jacobian and converges
        to the exact Jacobian.
        It is an unbiased estimator of the Sobolev norm
        for k=1; useful for large input/output grids.

        Args:
            model: The model being trained.
            x: Model input array shaped (c, d1, ..., dN).
            target: Ground truth array shaped (c, d1, ..., dN).
            key: Random key.

        Returns:
            Scalar Sobolev metric.
        """
        # We use vjp in backward mode or if output dim < input dim and auto mode
        # else jvp
        if self.diff_mode == "bwd":
            diff_op = "vjp"
            seeding_size = target.size
            seeding_shape = target.shape
        elif self.diff_mode == "fwd":
            diff_op = "jvp"
            seeding_size = x.size
            seeding_shape = x.shape
        else:  # diff_mode == "auto"
            if x.size <= target.size:
                diff_op = "jvp"
                seeding_size = x.size
                seeding_shape = x.shape
            else:
                diff_op = "vjp"
                seeding_size = target.size
                seeding_shape = target.shape

        # Choose random subset of basis vectors without replacement (lower variance)
        n_samples = min(self.n_random_samples, seeding_size)
        sampled_indices = jr.choice(
            key, seeding_size, shape=(n_samples,), replace=False
        )

        flat_random_basis_vectors = jnp.eye(seeding_size, dtype=x.dtype)[
            sampled_indices
        ]
        random_basis_vectors = jnp.reshape(
            flat_random_basis_vectors, (n_samples,) + seeding_shape
        )

        if diff_op == "vjp":  # backward/reverse mode -> seed from codomain
            primals, f_vjp = eqx.filter_vjp(model, x)
            # vjp returns a tuple of matching array as inputs
            jacs = jax.vmap(f_vjp)(random_basis_vectors)[0]
        else:  # forward mode -> seed from domain
            primals, jacs = jax.vmap(eqx.filter_jvp, in_axes=(None, None, 0))(
                model, (x,), (random_basis_vectors,)
            )

        fun_term = jnp.mean(jnp.pow(jnp.abs(target - primals), self.p))
        deriv_term = jnp.mean(jnp.pow(jnp.abs(jacs), self.p))
        return jnp.pow(fun_term + deriv_term, 1 / self.p)

    def stochastic_sobolev_stde(
        self,
        model: eqx.Module,
        *,
        x: Float[Array, "in_c ..."],
        target: Float[Array, "c ..."],
        key: PRNGKeyArray,
    ) -> Float[Array, ""]:
        """Computes the stochastic Sobolev metric for higher order derivatives.

        Uses `jax.experimental.jet`, which forwards a higher order
        Taylor polynomial which is more efficient than nested first order methods.

        Uses the sparse STDE approach, making `jet` more
        efficient by sampling random basis tensors instead
        of using all basis tensors. This yields random subsamples
        of exact and pure partial derivatives.
        The idea is the same as in `stochastic_sobolev_first_order`:
        An unbiased estimator of the Sobolev norm for k>1
        and large grids.

        ??? cite
            [Stochastic Taylor Derivative Estimator: Efficient
            amortization for arbitrary differential operators](
            https://proceedings.neurips.cc/paper_files/paper/2024/
            file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf)

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
            Scalar Sobolev metric.
        """
        # Jet propagates forward only -> use input size
        d = x.size
        # Sample random basis tensors without replacement (lower variance)
        n_samples = min(self.n_random_samples, d)
        sampled_indices = jr.choice(key, d, shape=(n_samples,), replace=False)
        flat_basis_tensors = jnp.eye(d, dtype=x.dtype)[sampled_indices]
        basis_tensors = jnp.reshape(flat_basis_tensors, (n_samples,) + x.shape)

        return self._sobolev_jet_from_directions(
            model, x=x, target=target, directions=basis_tensors
        )

    @override
    def __call__(
        self,
        model: eqx.Module | None = None,
        *,
        target: Float[Array, "b c ..."],
        x: Float[Array, "b in_c ..."] | None = None,
        pred: Float[Array, "b c ..."] | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """Computes the general Sobolev metric.

        Args:
            model: The model being trained.
            target: Ground truth array shaped (b, c, d1, ..., dN).
            x: Model input array shaped (b, in_c, d1, ..., dN)
            pred: Unused in Sobolev metric.
            key: Optional random key needed for stochastic Sobolev.
                Default is `None`.
            **kwargs: Additional arguments for specific metrics.

        Returns:
            Scalar metric.

        Raises:
            ValueError: If method is `"stochastic"` but no random key
                was passed. Or if `model` or `x` are not provided.
            KeyError: If an outstanding jet primitive rule is encountered
                for higher order derivatives for some models.
        """
        if model is None or x is None:
            raise ValueError(
                "SobolevMetric requires both 'model' and 'x' (inputs) "
                "to compute derivatives."
            )
        # Raise KeyError directly if k > 1 for neojax models with known outstanding primitives
        if (self.k > 1 or self.method == "interpolate") and type(model) in (
            GeoFNO,
            FNO,
            TFNO,
            UNO,
        ):
            raise KeyError(
                f"Known unsupported jet primitive rule encountered for neojax model {type(model)}."
            )

        strategy = self.method
        # Determine method based on number of basis tensors used for seeding and
        # max derivative order
        # If k > 1, jax.experimental.jet always uses forward mode so we don't need to check target.size
        if strategy == "auto":
            # Strip batch dim in directions check
            num_directions = (
                min(x[0].size, target[0].size) if self.k == 1 else x[0].size
            )
            if num_directions > self.direction_threshold:
                strategy = "stochastic"
            elif self.k == 1:
                strategy = "exact"
            else:
                strategy = "interpolate"

        if strategy == "stochastic" and key is None:
            raise ValueError("Stochastic Sobolev metric requires a random key.")

        def single_metric(x_i, target_i, key_i):
            if strategy == "exact":
                return self.exact_sobolev(model, x=x_i, target=target_i)
            elif strategy == "interpolate":
                return self.interpolated_sobolev(model, x=x_i, target=target_i)
            else:
                if self.k == 1:
                    return self.stochastic_sobolev_first_order(
                        model, x=x_i, target=target_i, key=key_i
                    )
                else:
                    return self.stochastic_sobolev_stde(
                        model, x=x_i, target=target_i, key=key_i
                    )

        batch_size = x.shape[0]
        if strategy == "stochastic":
            keys = jr.split(key, batch_size)
            batch_metrics = jax.vmap(single_metric)(x, target, keys)
        else:

            def single_metric_no_key(x_i, target_i):
                return single_metric(x_i, target_i, None)

            batch_metrics = jax.vmap(single_metric_no_key)(x, target)

        return self.weight * jnp.mean(batch_metrics)
