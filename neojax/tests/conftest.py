"""Globally available test suites and functions."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
import pytest
from jaxtyping import PyTree, install_import_hook

# setting up beartype with jaxtyping here
install_import_hook(
    modules=["neojax"],
    typechecker="beartype.beartype",
)


def assert_jittable(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Asserts a function is jittable.

    Asserts function is jittable and
    its outputs are unchanged under jit.

    Args:
        func: The function to test.
        args: Function arguments.
        kwargs: Function keyword arguments.
    """
    expected_output = func(*args, **kwargs)
    try:
        jit_func = jax.jit(func)
        # call once to check for correctness
        jit_fun_out1 = jit_func(*args, **kwargs)
        # call second time to check jax cache
        jit_fun_out2 = jit_func(*args, **kwargs)
    except Exception as e:
        pytest.fail(
            f"Function {func.__name__} failed to jit compile or execute. Exception {e}."
        )
    jtu.tree_map(
        lambda e, j1, j2: (
            np.testing.assert_allclose(j1, e, rtol=1e-5, atol=1e-5),
            np.testing.assert_allclose(j2, e, rtol=1e-5, atol=1e-5),
        ),
        expected_output,
        jit_fun_out1,
        jit_fun_out2,
    )


def assert_filter_jittable(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Asserts a function is filter_jittable using equinox.

    Useful for functions that take equinox modules as arguments.

    Args:
        func: The function to test.
        args: Function arguments.
        kwargs: Function keyword arguments.
    """
    expected_output = func(*args, **kwargs)
    try:
        jit_func = eqx.filter_jit(func)
        jit_fun_out1 = jit_func(*args, **kwargs)
        jit_fun_out2 = jit_func(*args, **kwargs)
    except Exception as e:
        pytest.fail(
            f"Function {func.__name__} failed to filter_jit execute. Exception {e}."
        )
    jtu.tree_map(
        lambda e, j1, j2: (
            np.testing.assert_allclose(j1, e, rtol=1e-4, atol=1e-4),
            np.testing.assert_allclose(j2, e, rtol=1e-4, atol=1e-4),
        ),
        expected_output,
        jit_fun_out1,
        jit_fun_out2,
    )


def assert_jittable_w_static(
    func: Callable[..., Any],
    *args: Any,
    static_argnames: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Asserts a function with (some) static args is jittable.

    Asserts function is jittable and
    its ouputs are unchanged under jit.

    Args:
        func: The function to test.
        args: Function arguments.
        static_argnames: Arguments declared static.
        kwargs: Function keyword arguments.
    """
    expected_output = func(*args, **kwargs)
    try:
        jit_func = jax.jit(func, static_argnames=static_argnames)
        # call once to check for correctness
        jit_fun_out1 = jit_func(*args, **kwargs)
        # call second time to check jax cache
        jit_fun_out2 = jit_func(*args, **kwargs)
    except Exception as e:
        pytest.fail(
            f"Function {func.__name__} failed to jit compile or execute w static args. Exception {e}."
        )
    jtu.tree_map(
        lambda e, j1, j2: (
            np.testing.assert_allclose(j1, e, rtol=1e-4, atol=1e-4),
            np.testing.assert_allclose(j2, e, rtol=1e-4, atol=1e-4),
        ),
        expected_output,
        jit_fun_out1,
        jit_fun_out2,
    )


def assert_tree_less(tree1: PyTree, tree2: PyTree):
    assert jtu.tree_all(jtu.tree_map(lambda a, b: a < b, tree1, tree2))


def assert_tree_greater(tree1: PyTree, tree2: PyTree):
    assert jtu.tree_all(jtu.tree_map(lambda a, b: a > b, tree1, tree2))


def assert_tree_equal(tree1: PyTree, tree2: PyTree):
    assert jtu.tree_all(jtu.tree_map(lambda a, b: a == b, tree1, tree2))


# The following functions are either stolen from the Equinox testing suite or adapted
def _assert_vars_equal(obj1, obj2, varnames):
    for varname in varnames:
        vars1 = getattr(obj1, varname)
        vars2 = getattr(obj2, varname)
        for a, b in zip(vars1, vars2, strict=True):
            assert a.aval.strip_weak_type() == b.aval.strip_weak_type()


def assert_jaxpr_equal(
    jaxpr1: jax.extend.core.ClosedJaxpr, jaxpr2: jax.extend.core.ClosedJaxpr
):
    assert len(jaxpr1.consts) == len(jaxpr2.consts)
    for c1, c2 in zip(jaxpr1.consts, jaxpr2.consts, strict=True):
        if isinstance(c1, jax.Array | np.ndarray) or isinstance(
            c2, jax.Array | np.ndarray
        ):
            np.testing.assert_array_equal(c1, c2)
        else:
            assert c1 == c2
    jaxpr1 = jaxpr1.jaxpr
    jaxpr2 = jaxpr2.jaxpr
    _assert_vars_equal(jaxpr1, jaxpr2, ("invars", "outvars", "constvars"))
    for eqn1, eqn2 in zip(jaxpr1.eqns, jaxpr2.eqns, strict=True):
        assert eqn1.primitive == eqn2.primitive
        assert eqn1.effects == eqn2.effects
        assert eqn1.params == eqn2.params
        _assert_vars_equal(eqn1, eqn2, ("invars", "outvars"))


def create_jaxpr_assert_equal(
    func1: Callable[[Any], Any],
    func2: Callable[[Any], Any],
    *args: Any,
    **kwargs: Any,
):
    """Creates and compares the jaxpressions of two functions with the same args for equality.

    Expressions are generated with `eqx.filter_make_jaxpr`.
    """
    jaxpr_fun1 = eqx.filter_make_jaxpr(func1)
    jaxpr_fun2 = eqx.filter_make_jaxpr(func2)
    closed_jaxpr1, shape_dtype_tree1, non_arr_tree1 = jaxpr_fun1(*args, **kwargs)
    closed_jaxpr2, shape_dtype_tree2, non_arr_tree2 = jaxpr_fun2(*args, **kwargs)
    assert_jaxpr_equal(closed_jaxpr1, closed_jaxpr2)
    assert_tree_equal(shape_dtype_tree1, shape_dtype_tree2)
    assert_tree_equal(non_arr_tree1, non_arr_tree2)


def create_assert_primitive_in_jaxpr(
    func: Callable[[Any], Any], prim: Any, *args: Any, **kwargs: Any
):
    """Creates jaxpression and verifies its equations contain given primitive."""
    jaxpr_fun = eqx.filter_make_jaxpr(func)
    closed_jaxpr, _, _ = jaxpr_fun(*args, **kwargs)
    prim_in_eqns = False
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive == prim:
            prim_in_eqns = True
            break
    assert prim_in_eqns, "Given primitive not in jaxpression"
