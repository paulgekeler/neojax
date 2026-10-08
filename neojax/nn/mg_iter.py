"""Implementation of general multigrid smoothing iteration as used in MgNO."""

import equinox as eqx
from jaxtyping import Array, Inexact, PRNGKeyArray


class MgIter(eqx.Module):
    r"""Class representing one multigrid smoothing iteration.

    Computes the initial smoothing update $u = S(f)$ or the update
     $u \leftarrow u + S(f - A(u))$ on subsequent iterations,
    where $S$ is the learnable smoothing operator and $A$ is the
    approximated PDE operator.


    Args:
        smoothing_op: Learnable smoothing operator S.
        pde_op: Learnable operator approximating the PDE operator A.
            For the initial smoothing step, this may be `None`.
            Default is `None`.

    ??? cite "Internal Attributes"
        These fields store the internal attributes.

        * **** (`eqx.Module`): Learnable smoothing operator S.
        * **** (`eqx.Module | None`): Learnable PDE operator A.
    """

    smoothing_op: eqx.Module
    pde_op: eqx.Module | None

    def __init__(
        self, smoothing_op: eqx.Module, pde_op: eqx.Module | None = None
    ) -> None:
        self.smoothing_op = smoothing_op
        self.pde_op = pde_op

    def __call__(
        self,
        f: Inexact[Array, "..."],
        u: Inexact[Array, "..."] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Inexact[Array, "..."], Inexact[Array, "..."]]:
        """Applies one smoothing step to the solution.

        Args:
            f: The data field.
            u: The solution field.
            key: Ignored. Kept for compatibility with `eqx.nn.Sequential`.

        Returns:
            Input data field `f` unchanged and smoothed solution `u`.
        """
        if u is None:
            return f, self.smoothing_op(f)
        return f, u + self.smoothing_op(f - self.pde_op(u))
