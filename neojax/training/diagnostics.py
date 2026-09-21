"""Gradient diagnostics for debugging neojax training runs."""

import fnmatch
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from neojax.training.trainer import Trainer, TrainState

_T = TypeVar("_T")


def select_by_path(report: dict[str, _T], pattern: str) -> dict[str, _T]:
    """Filters a path-keyed diagnostics dict by a glob-style pattern.

    Meant to narrow down `grad_norms`/`flag_anomalous_norms` outputs to one
    part of a large model, e.g. `select_by_path(grad_norms(grads),
    "*fno_blocks*")` for just the FNO blocks, or `"*.bias"` for every bias.

    Args:
        report: A dict as returned by `grad_norms`/`flag_anomalous_norms`,
            keyed by `jax.tree_util.keystr`-formatted paths.
        pattern: A glob-style pattern (`fnmatch` syntax: `*`, `?`, `[seq]`),
            matched against each key.

    Returns:
        The subset of `report` (a dict) whose keys match `pattern`.

    !!! warning "`[` and `]` are glob metacharacters, not literal brackets"
        `jax.tree_util.keystr` renders a sequence index as e.g. `[0]`, but
        `fnmatch` always parses `[...]` as a character class (matching
        exactly one character drawn from its contents).
        So a pattern like `"*channel_mlps[0]*"` will never match a key
        containing the literal text `[0]` or `[10]`. Don't try to write the
        brackets literally: use a plain `*` to span the index instead, e.g.
        `"*channel_mlps*.weight"` matches `channel_mlps[0].weight` and
        `channel_mlps[10].weight` alike. (Literal brackets are also matchable via
        `fnmatch`'s own escaping, `[[]` for `[` and `[]]` for `]`.)
    """
    return {k: v for k, v in report.items() if fnmatch.fnmatch(k, pattern)}


def grad_norms(grads: PyTree, ord: int | str = 2) -> dict[str, Float[Array, ""]]:
    """Computes each gradient leaf's norm, keyed by its path in the PyTree.

    Safe to call under `jit`/`vmap`: leaf paths are static (resolved at
    trace time into the returned dict's keys), and each norm is an ordinary
    traced JAX scalar, so the result is a valid PyTree that can be returned
    through a `Trainer(..., has_aux=True)` loss function's aux output to log
    live gradient norms without a second computation.

    Args:
        grads: A gradient PyTree, e.g. as returned by `eqx.filter_grad`.
            Leaves that are `None` (nondifferentiable, per
            `eqx.filter_grad`'s convention) contribute no entries.
        ord: Norm order forwarded to `jnp.linalg.norm`. Default is the L2 norm.

    Returns:
        A flat dict mapping each leaf's path string
        (e.g. `".fno_blocks.channel_mlps[0].weight"`) to its scalar norm.
    """
    return {
        jax.tree_util.keystr(path): jnp.linalg.norm(leaf.ravel(), ord=ord)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
        if leaf is not None
    }


def flag_anomalous_norms(
    grads: PyTree,
    *,
    min_norm: float | None = 1e-8,
    max_norm: float | None = None,
    ord: int | str = 2,
) -> dict[str, float]:
    """Flags gradient leaves whose norm falls outside `[min_norm, max_norm]`.

    A common first check for "did gradients actually reach this parameter"
    (near-zero norm, that is a dead leaf, e.g. from an accidental `static=True`
    or a faulty `filter_spec`) and "is training about to blow up" (a
    very large norm, e.g. from a noisy stochastic metric estimator).

    Args:
        grads: A gradient PyTree, as for `grad_norms`.
        min_norm: Leaves with norm strictly below this are flagged as
            (near-)vanishing. `None` disables the lower-bound check.
        max_norm: Leaves with norm strictly above this are flagged as
            exploding. `None` disables the upper-bound check.
        ord: Norm order forwarded to `jnp.linalg.norm`.

    Returns:
        A flat dict, keyed like `grad_norms`, containing only the leaves
        that violated a bound, mapped to their offending (Python float)
        norm or an empty dict if nothing was flagged.

    !!! warning "Call this outside `jit`"
        Which leaves end up in the returned dict depends on the gradients'
        actual (traced) values, so this cannot be staged inside a jitted
        function. Run it on materialized grads (e.g. from
        `compute_grad_report`, or from values pulled out of a
        `Trainer(has_aux=True)` step after it returns to plain Python).
    """
    norms = grad_norms(grads, ord=ord)
    return {
        path: float(norm)
        for path, norm in norms.items()
        if (min_norm is not None and float(norm) < min_norm)
        or (max_norm is not None and float(norm) > max_norm)
    }


def compute_grad_report(
    trainer: Trainer,
    state: TrainState,
    batch: Any,
    *,
    min_norm: float | None = 1e-8,
    max_norm: float | None = None,
    ord: int | str = 2,
) -> dict[str, float]:
    """Computes gradients for one batch and flags anomalous per-leaf norms.

    Mirrors `Trainer.train_step`'s own gradient computation (same
    `filter_spec`, same `loss_fn(..., training=True)` call), but stops after
    computing gradients. So it can be run standalone against a live `TrainState`
    (e.g. mid-run, loaded from a checkpoint) to inspect what a real training step's gradients
    look like, without disturbing training.

    Args:
        trainer: The `Trainer` whose `filter_spec`/`loss_fn`/`has_aux` to mirror.
        state: The `TrainState` to compute gradients at.
        batch: A single batch, exactly as passed to `Trainer.train_step`.
        min_norm: See `flag_anomalous_norms`.
        max_norm: See `flag_anomalous_norms`.
        ord: Norm order forwarded to `jnp.linalg.norm`.

    Returns:
        A flat dict of only the leaves whose gradient norm fell outside
        `[min_norm, max_norm]`, keyed by path, mapping to the offending
        norm or an empty dict if nothing was flagged.
    """
    filter_fn = (
        trainer.filter_spec if trainer.filter_spec is not None else eqx.is_inexact_array
    )
    diff, static = eqx.partition(state.model, filter_fn)

    def step_loss_fn(diff_parts):
        combined_model = eqx.combine(diff_parts, static)
        result = trainer.loss_fn(combined_model, batch, True)
        return result[0] if trainer.has_aux else result

    _, grads = eqx.filter_value_and_grad(step_loss_fn)(diff)
    return flag_anomalous_norms(grads, min_norm=min_norm, max_norm=max_norm, ord=ord)


__all__ = [
    "grad_norms",
    "flag_anomalous_norms",
    "compute_grad_report",
    "select_by_path",
]
