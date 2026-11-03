"""Implementation of the patchification of n-dimensional regular grids."""

import math

from jaxtyping import Array, Inexact


def patchify(
    x: Inexact[Array, "c ..."], patch_size: int
) -> Inexact[Array, "p_c num_patches"]:
    """Splits a regular grid into non-overlapping square (cubic, ...) patches.

    Every patch of `patch_size` points per axis becomes a token whose features
    are the `channels * patch_size**ndim` values in the patch. The tokens are ordered
    like the grid, i.e. the first grid axis varies slowest.

    Args:
        x: Signal on a regular grid, shaped (channels, d1, ..., dN).
        patch_size: Number of grid points per axis of a patch.

    Returns:
        The tokens, shaped (channels * patch_size**N, d1/patch_size * ... * dN/patch_size).

    Raises:
        ValueError: If `patch_size` is not a positive integer.
        ValueError: If `x` has no spatial axis or an axis is not divisible by `patch_size`.
    """
    _check_patch_size(patch_size)
    channels, *grid = x.shape
    if len(grid) == 0:
        raise ValueError("x needs at least one spatial axis besides the channel axis.")
    if any(g % patch_size != 0 for g in grid):
        raise ValueError(
            f"The grid {tuple(grid)} is not divisible by patch_size {patch_size}."
        )
    ndim = len(grid)
    num = [g // patch_size for g in grid]

    split_shape = [channels]
    for n in num:
        split_shape += [n, patch_size]
    x = x.reshape(split_shape)  # (c, n1, p, n2, p, ...)
    # channel, then the axes within a patch, then the patch grid
    perm = [0] + [2 + 2 * i for i in range(ndim)] + [1 + 2 * i for i in range(ndim)]
    x = x.transpose(perm)
    return x.reshape(channels * patch_size**ndim, math.prod(num))


def unpatchify(
    tokens: Inexact[Array, "p_c num_patches"],
    patch_size: int,
    grid_shape: tuple[int, ...],
) -> Inexact[Array, "c ..."]:
    """Inverse of `patchify`.

    Takes square patches and restores the rectangular grid.

    Args:
        tokens: The tokens, shaped
            (channels * patch_size**N, d1/patch_size * ... * dN/patch_size).
        patch_size: Number of grid points per axis of a patch.
        grid_shape: Shape (d1, ..., dN) of the grid to restore.

    Returns:
        The signal on the grid, shaped (channels, d1, ..., dN).

    Raises:
        ValueError: If `patch_size` is not a positive integer.
        ValueError: If `grid_shape` is empty or not divisible by `patch_size`.
        ValueError: If the shape of `tokens` does not match `patch_size` and `grid_shape`.
    """
    _check_patch_size(patch_size)
    grid_shape = tuple(grid_shape)
    ndim = len(grid_shape)
    if ndim == 0:
        raise ValueError("grid_shape needs at least one axis.")
    if any(g % patch_size != 0 for g in grid_shape):
        raise ValueError(
            f"The grid {grid_shape} is not divisible by patch_size {patch_size}."
        )
    num = [g // patch_size for g in grid_shape]
    features, num_patches = tokens.shape
    if features % patch_size**ndim != 0 or num_patches != math.prod(num):
        raise ValueError(
            f"tokens of shape {tokens.shape} do not match the grid {grid_shape} "
            f"with patch_size {patch_size}."
        )
    channels = features // patch_size**ndim

    x = tokens.reshape(channels, *([patch_size] * ndim), *num)
    perm = [0]
    for i in range(ndim):
        perm += [1 + ndim + i, 1 + i]  # grid axis i, then patch axis i
    return x.transpose(perm).reshape(channels, *grid_shape)


def _check_patch_size(patch_size: int) -> None:
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer.")
