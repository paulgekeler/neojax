"""Benchmark neojax performance."""

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from neojax.models import FNO


def benchmark_jax(n_batches, batch_size, n_spatial, modes, width, n_layers, data_x):
    """Benchmarks JAX inference performance.

    Args:
        n_batches: Number of batches to run.
        batch_size: Batch size of each run.
        n_spatial: Spatial resolution.
        modes: Number of modes.
        width: Hidden width.
        n_layers: Number of FNO layers.
        data_x: Input data for benchmarking.

    Returns:
        Tuple of (JIT compile time, mean batch inference time).
    """
    key = jr.key(0)
    model_key = jr.split(key)[0]

    # Init Jax Model
    jax_model = FNO(
        key=model_key,
        in_channels=1,
        out_channels=1,
        hidden_channels=width,
        n_layers=n_layers,
        modes=(modes,),
    )

    # vmap the model forward pass over batch dimension (0-th axis)
    jax_fn = eqx.filter_jit(jax.vmap(jax_model))

    # Prepare batch data
    batches = []
    for i in range(n_batches):
        start_idx = (i * batch_size) % len(data_x)
        end_idx = start_idx + batch_size
        batch = data_x[start_idx:end_idx]
        # Handle wrap-around if dataset is too small
        if len(batch) < batch_size:
            extra = data_x[: batch_size - len(batch)]
            batch = jnp.concatenate([batch, extra], axis=0)
        batches.append(batch)

    # JIT Compile Time (warmup with the exact shape)
    start = time.time()
    _ = jax_fn(batches[0]).block_until_ready()
    jit_compile_time = time.time() - start

    # Inference (JIT)
    start = time.time()
    for i in range(n_batches):
        _ = jax_fn(batches[i]).block_until_ready()
    jax_jit_runtime = (time.time() - start) / n_batches

    return jit_compile_time, jax_jit_runtime


def train_jax(model, train_x, train_y, lr=1e-3, steps=100):
    """Trains the JAX model and measures time.

    Args:
        model: JaxFNO model.
        train_x: Training inputs.
        train_y: Training labels.
        lr: Learning rate.
        steps: Number of training steps.

    Returns:
        Tuple of (trained model, compile time, training step execution time).
    """
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y) ** 2)

    @eqx.filter_jit
    def make_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Warmup with the EXACT same shape to compile the function
    start_comp = time.time()
    model, opt_state, loss = make_step(model, opt_state, train_x, train_y)
    jax.block_until_ready(loss)
    compile_time = time.time() - start_comp

    start = time.time()
    for i in range(steps):
        model, opt_state, loss = make_step(model, opt_state, train_x, train_y)
    # Ensure completion on GPU
    jax.block_until_ready(loss)
    elapsed = time.time() - start
    return model, compile_time, elapsed


def run_benchmarks(
    dataset_file: str,
    stat_file: str,
    train_steps: int,
    train_batch_size: int,
    inf_batch_size: int,
    n_batches: int,
):
    """Run neojax benchmarks."""
    with np.load(dataset_file) as dataset:
        X = dataset["X"]
        Y = dataset["Y"]

    # convert to jnp arrays
    X = jnp.array(X)
    Y = jnp.array(Y)

    spatial_resolutions = [128, 256, 512, 1024]
    width = 64
    modes = 16
    n_layers = 4

    stats = []

    for res in spatial_resolutions:
        print(f"\nBenchmarking resolution: {res} with batch size: {inf_batch_size}")

        # Resize data for resolution
        indices = jnp.linspace(0, X.shape[-1] - 1, res).astype(int)
        x_res = X[:, :, indices]

        jax_jit_compile, jax_runtime = benchmark_jax(
            n_batches, inf_batch_size, res, modes, width, n_layers, x_res
        )

        stats.append(
            {
                "Type": "Inference",
                "Resolution": res,
                "Implementation": "JAX (Equinox/JIT)",
                "Compile Time (s)": jax_jit_compile,
                "Inference Time (ms)": jax_runtime * 1000,
                "Batch Size": inf_batch_size,
            }
        )

    # Training Comparison
    print(
        f"\nBenchmarking JAX Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    # Take a batch of train_batch_size at 1024 resolution
    # Resize X and Y to 1024 resolution
    indices_1024 = jnp.linspace(0, X.shape[-1] - 1, 1024).astype(int)
    x_train = X[:train_batch_size, :, indices_1024]
    y_train = Y[:train_batch_size, :, indices_1024]

    # Jax Training
    key = jr.key(0)
    jax_model = FNO(
        key=key,
        in_channels=1,
        out_channels=1,
        hidden_channels=width,
        n_layers=n_layers,
        modes=(modes,),
    )
    _, jax_train_compile, jax_train_time = train_jax(
        jax_model, x_train, y_train, steps=train_steps
    )

    print(f"JAX Training Compiled in: {jax_train_compile:.4f}s")
    print(f"JAX Training Execution ({train_steps} steps): {jax_train_time:.4f}s")

    stats.append(
        {
            "Type": "Training",
            "Implementation": "JAX (JIT)",
            "Compile Time (s)": jax_train_compile,
            "Training Time (s)": jax_train_time,
            "Steps": train_steps,
            "Batch Size": train_batch_size,
        }
    )

    with open(stat_file, "w") as f:
        json.dump(stats, f)


def parse_args():
    """Parse cmd args."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_file", type=str, required=True)
    argparser.add_argument("--stat_file", type=str, required=True)
    argparser.add_argument("--train_steps", type=int, default=1000)
    argparser.add_argument("--train_batch_size", type=int, default=64)
    argparser.add_argument("--inf_batch_size", type=int, default=32)
    argparser.add_argument("--n_batches", type=int, default=50)
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(
        args.dataset_file,
        args.stat_file,
        args.train_steps,
        args.train_batch_size,
        args.inf_batch_size,
        args.n_batches,
    )
