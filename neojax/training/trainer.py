"""Implementation of JAX-idiomatic training orchestrator (Trainer and TrainState)."""

import json
from collections.abc import Callable
from os import PathLike
from typing import Any, BinaryIO

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray


class TrainState(eqx.Module):
    """JAX PyTree holding the training state of an Equinox model.

    Enables functional optimization updates and checkpoint serialization.

    Args:
        model: The Equinox model to train.
        opt_state: The optax optimizer state.
        step: The current training step. Default is 0.
        metadata: Optional dictionary of starting training metadata.

    ??? info "Internal Attributes"
        These fields store the training state of the model.

        * **model** (`eqx.Module`): The Neural Operator or standard Equinox model.
        * **opt_state** (`optax.OptState`): The optax optimizer state.
        * **step** (`Int[Array, ""]`): The current training step/iteration.
        * **metadata** (`dict[str, Any]`): Static metadata dictionary (epochs, batch index, etc.).
    """

    model: eqx.Module
    opt_state: optax.OptState
    step: Int[Array, ""]
    metadata: dict[str, Any] = eqx.field(static=True)

    def __init__(
        self,
        model: eqx.Module,
        opt_state: optax.OptState,
        step: int | Int[Array, ""] = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.opt_state = opt_state
        if isinstance(step, int):
            self.step = jnp.asarray(step, dtype=jnp.int32)
        else:
            self.step = step
        self.metadata = metadata if metadata is not None else {}

    def save(
        self,
        file: str | PathLike | BinaryIO,
        model_hyperparams: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Saves the TrainState weights, optimizer state, and metadata to disk.

        Writes model hyperparameters and custom metadata as a JSON header line,
        followed by the binary leaves of the model and optimizer PyTree states.

        Args:
            file: File path or binary stream descriptor.
            model_hyperparams: Dictionary describing the model initialization configuration.
            kwargs: Additional arguments passed to `tree_serialise_leaves`.
        """
        # Cast step to int to be JSON serialisable
        checkpoint_dict = {
            "model_hyperparams": model_hyperparams,
            "metadata": {**self.metadata, "step": int(self.step)},
        }
        metadata_str = json.dumps(checkpoint_dict)

        if hasattr(file, "write"):
            file.write((metadata_str + "\n").encode())
            eqx.tree_serialise_leaves(file, self, **kwargs)
        else:
            with open(file, "wb") as f:
                f.write((metadata_str + "\n").encode())
                eqx.tree_serialise_leaves(f, self, **kwargs)

    @classmethod
    def load(
        cls,
        file: str | PathLike | BinaryIO,
        make_fn: Callable[..., eqx.Module],
        optimizer: optax.GradientTransformation,
        key: PRNGKeyArray | None = None,
        filter_spec: Any = None,
        **kwargs: Any,
    ) -> "TrainState":
        """Loads a TrainState and metadata from disk.

        Parses the JSON header, calls `make_fn` to reconstruct the model skeleton
        dynamically, initializes the opt_state skeleton, and deserializes all binary leaves.

        Args:
            file: File path or binary stream descriptor.
            make_fn: Function that constructs the model skeleton from loaded hyperparameters.
            optimizer: The optax GradientTransformation optimizer.
            key: Optional PRNGKey passed to `make_fn`.
            filter_spec: Optional PyTree filter spec (or callable) indicating which parameters
                are learnable. Defaults to `equinox.is_inexact_array`.
            kwargs: Additional arguments passed to `tree_deserialise_leaves`.

        Returns:
            The loaded TrainState instance with restored weights and metadata.
        """
        if hasattr(file, "read"):
            checkpoint_dict = json.loads(file.readline().decode())
            f = file
        else:
            f = open(file, "rb")
            checkpoint_dict = json.loads(f.readline().decode())

        model_hyperparams = checkpoint_dict.get("model_hyperparams", {})
        metadata = checkpoint_dict.get("metadata", {})
        # No need to create Jax array here, the initializer calls `asarray` on step
        step = metadata.get("step", 0)

        # Rebuild the model skeleton
        if key is not None:
            model_skeleton = make_fn(key=key, **model_hyperparams)
        else:
            model_skeleton = make_fn(**model_hyperparams)

        # Initialize corresponding optimizer state skeleton
        filter_fn = filter_spec if filter_spec is not None else eqx.is_inexact_array
        diff, _ = eqx.partition(model_skeleton, filter_fn)
        opt_state_skeleton = optimizer.init(diff)

        skeleton = cls(
            model=model_skeleton,
            opt_state=opt_state_skeleton,
            step=step,
            metadata=metadata,
        )

        try:
            state = eqx.tree_deserialise_leaves(f, skeleton, **kwargs)
        finally:
            if not hasattr(file, "read"):
                f.close()

        return state


class Trainer(eqx.Module):
    """Trainer class to orchestrate training loop updates.

    Handles gradient steps, loss evaluation, JIT compilation, and batching.

    Args:
        optimizer: The optax GradientTransformation optimizer.
        loss_fn: Functional loss function taking (model, batch, training) and
            returning a scalar loss or `(loss, aux)` if `has_aux` is `True`.
            `training` is `True` when called from `train_step` and `False`
            when called from `eval_step`, so the loss function can forward it
            as e.g. `inference=not training` to models with dropout
            or other train/eval-dependent behavior.
        filter_spec: Optional PyTree filter spec (or callable) indicating which parameters
            are learnable. Defaults to `equinox.is_inexact_array`.
        has_aux: If `True`, `loss_fn` returns `(loss, aux)` instead of a just
            loss, mirroring `jax.value_and_grad` own `has_aux` convention.
            `aux` is carried through unchanged. Useful for e.g. returning a
            `ComposedMetric` per-term values (via its own
            `return_components=True`). Changes `train_step` return to
            `(state, loss, aux)` and `eval_step` to `(loss, aux)`.

    ??? info "Internal Attributes"
        These fields store the trainer configuration.

        * **optimizer** (`optax.GradientTransformation`): The optax optimizer instance.
        * **loss_fn** (`Callable[[eqx.Module, Any, bool], Float[Array, ""]]`): Functional
            loss function, called with a `training` flag (see `Args`).
        * **filter_spec** (`Any`): Filter specification PyTree or callable for parameter updates.
        * **has_aux** (`bool`): Whether `loss_fn` returns an auxiliary value alongside the loss.
    """

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    loss_fn: Callable[[eqx.Module, Any, bool], Float[Array, ""]] = eqx.field(
        static=True
    )
    filter_spec: Any = eqx.field(static=True)
    has_aux: bool = eqx.field(static=True)

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable[[eqx.Module, Any, bool], Float[Array, ""]],
        filter_spec: Any = None,
        has_aux: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.filter_spec = filter_spec
        self.has_aux = has_aux

    def create_train_state(
        self,
        model: eqx.Module,
        metadata: dict[str, Any] | None = None,
    ) -> TrainState:
        """Creates a fresh TrainState for training.

        Args:
            model: The Equinox model to train.
            metadata: Optional dictionary of starting training metadata.

        Returns:
            A new TrainState instance.
        """
        filter_fn = (
            self.filter_spec if self.filter_spec is not None else eqx.is_inexact_array
        )
        diff, _ = eqx.partition(model, filter_fn)
        opt_state = self.optimizer.init(diff)
        return TrainState(model=model, opt_state=opt_state, step=0, metadata=metadata)

    @eqx.filter_jit
    def train_step(
        self, state: TrainState, batch: Any
    ) -> tuple[TrainState, Any] | tuple[TrainState, Any, Any]:
        """Performs a single JIT-compiled functional optimization step.

        Args:
            state: The current TrainState.
            batch: Data batch PyTree passed to `loss_fn`.

        Returns:
            `(new_state, loss)` or `(new_state, loss, aux)` if `has_aux`.
        """
        filter_fn = (
            self.filter_spec if self.filter_spec is not None else eqx.is_inexact_array
        )

        # Partition model into trainable (diff) and non-trainable (static) leaves
        diff, static = eqx.partition(state.model, filter_fn)

        def step_loss_fn(diff_parts):
            combined_model = eqx.combine(diff_parts, static)
            return self.loss_fn(combined_model, batch, True)

        if self.has_aux:
            (loss_val, aux), grads = eqx.filter_value_and_grad(
                step_loss_fn, has_aux=True
            )(diff)
        else:
            loss_val, grads = eqx.filter_value_and_grad(step_loss_fn)(diff)

        updates, opt_state = self.optimizer.update(grads, state.opt_state, diff)
        new_diff = eqx.apply_updates(diff, updates)

        # Combine back into a complete model PyTree
        model = eqx.combine(new_diff, static)

        new_state = TrainState(
            model=model,
            opt_state=opt_state,
            step=state.step + 1,
            metadata=state.metadata,
        )

        if self.has_aux:
            return new_state, loss_val, aux
        return new_state, loss_val

    @eqx.filter_jit
    def eval_step(
        self, state: TrainState, batch: Any
    ) -> Float[Array, ""] | tuple[Float[Array, ""], tuple[Float[Array, ""], ...]]:
        """Performs a single JIT-compiled evaluation forward pass.

        Unlike `train_step`, this neither computes gradients nor updates the
        optimizer. It calls `loss_fn` with `training=False` so the model
        can disable dropout and other train-only stochastic behavior, and
        returns the resulting loss value directly.

        Args:
            state: The current TrainState.
            batch: Data batch PyTree passed to `loss_fn`.

        Returns:
            The batch loss value, or `(loss, aux)` if `has_aux`.
        """
        return self.loss_fn(state.model, batch, False)
