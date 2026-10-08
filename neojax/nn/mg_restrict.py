"""Implementation of general multigrid restriction operator as used in MgNO."""

import equinox as eqx
from jaxtyping import Array, Inexact


class MgRestrict(eqx.Module):
    """Multigrid restriction operator.

    Projects `(u, f)` to a coarser grid.

    Optionally computes the residual $f - A(u)$ before restricting,
    which corresponds to the 'full approximation scheme' (FAS) if
    A is given.

    Args:
        sol_restrict_op: Learnable restriction operator for the solution u.
        data_restrict_op: Learnable restriction operator for the data f.
        pde_op: Learnable operator approximating the PDE operator A. Used to compute
            the residual if passed. Default is `None`.

    ??? cite "Internal Attributes"
        These fields store the internal attributes.

        * **sol_restrict_op** (`eqx.Module`): Restriction operator for the solution u.
        * **data_restrict_op** (`eqx.Module`): Restriction operator for the data f.
        * **pde_op** (`eqx.Module | None`): Learnable PDE operator.
    """

    sol_restrict_op: eqx.Module
    data_restrict_op: eqx.Module
    pde_op: eqx.Module | None

    def __init__(
        self,
        sol_restrict_op: eqx.Module,
        data_restrict_op: eqx.Module,
        pde_op: eqx.Module | None = None,
    ) -> None:
        self.sol_restrict_op = sol_restrict_op
        self.data_restrict_op = data_restrict_op
        self.pde_op = pde_op

    def __call__(
        self, f: Inexact[Array, "..."], u: Inexact[Array, "..."]
    ) -> tuple[Inexact[Array, "..."], Inexact[Array, "..."]]:
        """Restricts (prolongates) solution and data to coarser grid.

        Args:
            f: The data field.
            u: The solution field.

        Returns:
            The data field `f` and pde solution field `u` on the coarser grid.
        """
        if self.pde_op is None:
            f = self.data_restrict_op(f)
        else:
            f = self.data_restrict_op(f - self.pde_op(u))
        u = self.sol_restrict_op(u)
        return f, u
