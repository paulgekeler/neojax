"""Implementation of the base Neural Operator."""

import json
import traceback
from abc import abstractmethod
from collections.abc import Callable
from os import PathLike
from typing import Any, BinaryIO

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, DTypeLike, Inexact, PRNGKeyArray


class BaseNO(eqx.Module):
    """Abstract base Neural Operator.

    Defines shared functionalities for serialization,
    dtype conversion, inspection and profiling.
    All neojax models subclass `BaseNO`.

    Custom Neural Operator classes should inherit from `BaseNO`.
    """

    def save_weights(self, file: str | PathLike | BinaryIO, **kwargs: Any) -> None:
        """Saves only the model weights to the given file.

        Args:
            file: File descriptor or binary IO stream.
            kwargs: Additional arguments to pass to `tree_serialise_leaves`.
        """
        eqx.tree_serialise_leaves(file, self, **kwargs)

    def load_weights(self, file: str | PathLike | BinaryIO, **kwargs: Any) -> "BaseNO":
        """Loads only the model weights from the given file.

        Args:
            file: File descriptor or binary IO stream.
            kwargs: Additional arguments to pass to `tree_deserialise_leaves`.
        """
        return eqx.tree_deserialise_leaves(file, self, **kwargs)

    def save(
        self,
        *,
        file: str | PathLike | BinaryIO,
        hyperparams: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Saves the model to the given file.

        Args:
            file: File descriptor or binary IO stream.
            hyperparams: The hyperparameters needed to replicate the model instance.
            kwargs: Additional arguments to pass to `tree_serialise_leaves`.
        """
        if hasattr(file, "write"):
            hyperparam_str = json.dumps(hyperparams)
            file.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(file, self, **kwargs)
        else:
            with open(file, "wb") as f:
                hyperparam_str = json.dumps(hyperparams)
                f.write((hyperparam_str + "\n").encode())
                eqx.tree_serialise_leaves(f, self, **kwargs)

    @classmethod
    def load(
        cls,
        *,
        file: str | PathLike | BinaryIO,
        make_fn: Callable[[Any], "BaseNO"],
        key: PRNGKeyArray | None = None,
        **kwargs: Any,
    ) -> "BaseNO":
        """Loads the model from the given file.

        Args:
            file: File descriptor or binary IO stream.
            make_fn: A function that takes all hyperparams saved to the model file
                and uses the necessary ones to construct the model
                and returns a model instance. See
                [Equinox docs](https://docs.kidger.site/equinox/examples/serialisation/)
                for details.
            key: Optional random key passed to `make_fn`. If `None` and the model creation
                requires a random key, make sure to pass via `functools.partial(make_fn, key=key)`.
                Default is `None`.
            kwargs: Additional arguments to pass to `tree_deserialise_leaves`.

        Returns:
            The loaded model instance.
        """
        if hasattr(file, "write"):
            hyperparams = json.loads(file.readline().decode())
            if key is not None:
                skeleton = make_fn(key=key, **hyperparams)
            else:
                skeleton = make_fn(**hyperparams)
            model = eqx.tree_deserialise_leaves(file, skeleton, **kwargs)
            return model
        else:
            with open(file, "rb") as f:
                hyperparams = json.loads(f.readline().decode())
                if key is not None:
                    skeleton = make_fn(key=key, **hyperparams)
                else:
                    skeleton = make_fn(**hyperparams)
                model = eqx.tree_deserialise_leaves(f, skeleton, **kwargs)
            return model

    def astype(self, dtype: DTypeLike) -> "BaseNO":
        """Casts all inexact model parameters to the given dtype.

        When casting to float, this method preserves the real/complex separation
        of complex parameters (by default,
        casting a complex to float in jax silently discards the imaginary part).

        E.g. `astype(jnp.float32)` casts complex parameters to `jnp.complex64`
        and real parameters to `jnp.float32`.

        All real parameters are cast to the desired type.

        Args:
            dtype: Valid `jaxtyping.DTypeLike` to cast parameters to.

        Returns:
            A model instance with parameters cast to `dtype`.

        Raises:
            ValueError: If an inadequate `dtype` is passed (e.g. any `int`).

        !!! warning "Half float casting for complex arrays"
            Some Neural Operators (FNO, ...) have complex spectral weights.
            JAX doesn't support `complex32`, meaning casting a model with complex weights
            to half precision floats (`jnp.float16`, `jnp.bfloat16`) leaves
            the complex weights as `jnp.complex64`.
        """
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise ValueError(
                "'model.astype()' only supports float or complex casting dtypes."
            )

        rc_dtype_mapping = {
            jnp.float32: jnp.complex64,
            jnp.float64: jnp.complex128,
            jnp.float16: jnp.complex64,  # fallback
            jnp.bfloat16: jnp.complex64,  # fallback
        }
        cr_dtype_mapping = {
            jnp.complex64: jnp.float32,
            jnp.complex128: jnp.float64,
        }

        def _cast_leaf(leaf):
            if not eqx.is_array(leaf):
                return leaf
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                target_dtype = rc_dtype_mapping.get(dtype, dtype)
                return leaf.astype(target_dtype)
            elif jnp.issubdtype(leaf.dtype, jnp.floating):
                target_dtype = cr_dtype_mapping.get(dtype, dtype)
                return leaf.astype(target_dtype)
            return leaf

        return jtu.tree_map(_cast_leaf, self)

    def profile_compile(
        self,
        dummy_input: Inexact[Array, "in_c ..."],
        return_lowering: bool = False,
        filter_jax_frames: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Profiles the XLA-compilation of the model.

        This method relies on jax AOT lowering and compilation.
        See [JAX docs](https://docs.jax.dev/en/latest/jax.stages.html) for details.

        Args:
            dummy_input: Dummy array representative of a training sample to pass
                to the jitted model.
            return_lowering: Whether to return a human-readable representation of
                the model lowering in the summary dict.
            filter_jax_frames: In case of exceptions, whether to filter the jax and equinox
                frames to only focus on model code (jax internal traces are long).

        Returns:
            `True` if the compilation succeeded else `False` and
            a summary dict of the compilation:

            {

                "lowering_cost_analysis": variable dict returned by `jax.stages.Lowered.cost_analysis()` or `None`,

                "compiled_cost_analysis": variable dict returned by `jax.stages.Compiled.cost_analysis()` or `None`,

                "exception": Possible exception string or `None`,

                "lowering": Text representation of model lowering if `return_lowering` is `True` else empty string
            }

        !!! warning "Analysis Outputs"
            The outputs of the cost analysis vary with the underlying compiler. They may be `None`
            if the compiler doesn't provide such information.
        """
        summary = {
            "lowering_cost_analysis": None,
            "compiled_cost_analysis": None,
            "exception": None,
            "lowering": "",
        }

        try:
            # filter_jit is a wrapper -> ensure we get the underlying object
            # then get jax.stages.Lowered from equinox.Lowered
            lowered = eqx.filter_jit(self).lower(dummy_input).lowered
            summary["lowering"] = lowered.as_text() if return_lowering else ""
            summary["lowering_cost_analysis"] = lowered.cost_analysis()
        except Exception as e:
            if filter_jax_frames:
                formatted_trace = traceback.format_exception(
                    type(e), e, e.__traceback__
                )
                # Filter frames containing 'jax/' or 'equinox/'
                user_friendly_trace = [
                    f for f in formatted_trace if "jax" not in f and "equinox" not in f
                ]
                summary["exception"] = "".join(user_friendly_trace)
            else:
                summary["exception"] = f"Lowering failed: {type(e).__name__} - {str(e)}"
            return False, summary

        try:
            compiled = lowered.compile()
            summary["compiled_cost_analysis"] = compiled.cost_analysis()
        except Exception as e:
            if filter_jax_frames:
                formatted_trace = traceback.format_exception(
                    type(e), e, e.__traceback__
                )
                # Filter frames containing 'jax/' or 'equinox/'
                user_friendly_trace = [
                    f for f in formatted_trace if "jax" not in f and "equinox" not in f
                ]
                summary["exception"] = "".join(user_friendly_trace)
            else:
                summary["exception"] = (
                    f"Compilation failed: {type(e).__name__} - {str(e)}"
                )
            return False, summary

        return True, summary

    def size(self, return_n_params: bool = False) -> float | tuple[float, int]:
        """Computes model size in MB.

        Args:
            return_n_params: Whether to return the number of parameters.
                Default is `False`.

        Returns:
            The model size in MB and number of parameters if `return_n_params` is `True`.
        """
        params = eqx.filter(self, eqx.is_array)
        n_params, bytesize = zip(
            *[(x.size, x.dtype.itemsize) for x in jtu.tree_leaves(params)], strict=True
        )
        model_size = (
            sum(np * bs for np, bs in zip(n_params, bytesize, strict=True)) / 10**6
        )
        if return_n_params:
            return model_size, sum(n_params)
        else:
            return model_size

    @abstractmethod
    def __call__(self, x: Inexact[Array, "in_c ..."]) -> Inexact[Array, "out_c ..."]:
        """Forward pass of the model.

        Args:
            x: Input array shaped (in_c, d1, ..., dN).

        Returns:
            Output array shaped (out_c, d1, ..., dN).
        """
        ...
