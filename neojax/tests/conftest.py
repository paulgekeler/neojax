"""Globally available test suites and functions."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest
from jaxtyping import install_import_hook

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
    jax.tree_util.tree_map(
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
    jax.tree_util.tree_map(
        lambda e, j1, j2: (
            np.testing.assert_allclose(j1, e, rtol=1e-5, atol=1e-5),
            np.testing.assert_allclose(j2, e, rtol=1e-5, atol=1e-5),
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
    jax.tree_util.tree_map(
        lambda e, j1, j2: (
            np.testing.assert_allclose(j1, e, rtol=1e-5, atol=1e-5),
            np.testing.assert_allclose(j2, e, rtol=1e-5, atol=1e-5),
        ),
        expected_output,
        jit_fun_out1,
        jit_fun_out2,
    )
