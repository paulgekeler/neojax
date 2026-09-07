"""Implementation of a general data wrapper for arbitrary PDE problems."""

import equinox as eqx
from jaxtyping import Array, Inexact, Int, Real


class DataBundle(eqx.Module):
    r"""General data bundle/wrapper for arbitrary PDE data.

    Wraps both grid-like inputs and mesh-like inputs, e.g.

    FNO or 2D fluid simulation:

    - `coords` shape `[2, 64, 64]` ($x, y$ coordinates in $64 \times 64$ grid)
    - `fields` shape `[10, 1, 64, 64]` (10 time steps, 1 scalar vorticity channel)
    - `edge_indices` are `None` (model's schema implicity knows neighboring values from layout)

    Mesh-like (GNO):

    - `coords` shape `[2, 14500]` ($x, y$ coordinates for $14500$ mesh points)
    - `fields` shape `[10, 1, 14500]` (10 time steps, 1 scalar velocity channel per node)
    - `edge_indices` shape `[2, 58000]` (connectivity map how the vertices form triangles)

    !!! info "Attributes"
        * **coords** (`Real[Array, "#b d *spatial"]`): Grid-like or mesh-like coordinates shaped (dims, *spatial dims).
        * **fields** (`Inexact[Array, "#b t c *spatial"]`): Discretized function fields
            shaped (time dim, channels, *spatial dims).
        * **parameters** (`Inexact[Array, "#b ..."] | None`): Optional physical/material parameters of any shape.
            Default is `None`.
        * **bc_masks** (`Int[Array, "#b c_bc *spatial"] | None`): Optional boundary condition mask
            shaped (boundary cond channels, *spatial dims). Default is `None`.
        * **bc_values** (`Inexact[Array, "#b t c_bc *spatial"] | None`): Optional boundary condition fields
            shaped (time dim, boundary cond channels, *spatial dims). Default is `None`.
        * **edge_indices** (`Int[Array, "#b 2 e"] | None`): Optional edge connectivity map shaped (2, num edges).
            Default is `None`.

    !!! info
        Steady-state `fields` (and `bc_values`) should have `t=1`
        and the corresponding `Schema` should squeeze the time dimension
        before passing the inputs to the model.

        The batch dimension `*batch` is denoted as `#b` in the annotations,
        indicating that it is a broadcastable dimension of size 1 or more.
        During processing, `DataBundle` naturally supports passing batches of data.
    """

    # grid or mesh coords
    # grid: [b, d, x, y, ...] | mesh: [b, d, N]
    coords: Real[Array, "#b d *spatial"]
    # discretized function fields
    # grid: [b, t, c, x, y, ...] | mesh: [b, t, c, N]
    fields: Inexact[Array, "#b t c *spatial"]
    # physical/material parameters
    # may be fields, constants or other
    parameters: Inexact[Array, "#b ..."] | None = eqx.field(default=None)
    # optional boundary fields
    bc_masks: Int[Array, "#b c_bc *spatial"] | None = eqx.field(default=None)
    bc_values: Inexact[Array, "#b t c_bc *spatial"] | None = eqx.field(default=None)
    # optional mesh topology
    # cartesian domain = None otherwise e is number of edges connecting nodes
    edge_indices: Int[Array, "#b 2 e"] | None = eqx.field(default=None)
