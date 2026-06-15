"""Benchmark 2D FNO performance in JAX using camlab-ethz/FNS-KF."""

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


def benchmark_jax_2d(n_batches, batch_size, n_spatial, modes, width, n_layers, data_x):
    """Benchmarks 2D JAX inference performance.

    Args:
        n_batches: Number of batches to run.
        batch_size: Batch size of each run.
        n_spatial: Spatial resolution.
        modes: Tuple of modes.
        width: Hidden width.
        n_layers: Number of FNO layers.
        data_x: Input data for benchmarking.

    Returns:
        Tuple of (JIT compile time, mean batch inference time).
    """
    key = jr.key(0)
    model_key = jr.split(key)[0]

    # Init Jax Model for 2D FNO
    jax_model = FNO(
        key=model_key,
        in_channels=2,
        out_channels=2,
        hidden_channels=width,
        n_layers=n_layers,
        modes=modes,  # (modes, modes)
    )

    # vmap the model forward pass over batch dimension (0-th axis)
    jax_fn = eqx.filter_jit(jax.vmap(jax_model))

    # Prepare batch data
    batches = []
    for i in range(n_batches):
        start_idx = (i * batch_size) % len(data_x)
        end_idx = start_idx + batch_size
        batch = data_x[start_idx:end_idx]
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


def train_jax_2d(model, train_x, train_y, lr=1e-3, steps=100):
    """Trains the 2D JAX model and measures time.

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
    """Run 2D JAX benchmarks."""
    # Load dataset
    try:
        import h5py

        print(f"Loading 2D dataset from {dataset_file}...")
        with h5py.File(dataset_file, "r") as f:
            if "solution" in f:
                sol = f["solution"]
                # Solution shape: (N, T, C, H, W)
                train_size = max(train_batch_size, inf_batch_size * n_batches)
                print(f"Reading {train_size} samples...")
                X = sol[:train_size, 0]  # First step shape: (N, 2, 128, 128)
                Y = sol[:train_size, 10]  # t=10 target shape: (N, 2, 128, 128)
            else:
                raise KeyError("'solution' not found in dataset file")
    except Exception as e:
        print(f"Could not load NetCDF dataset: {e}. Falling back to synthetic 2D data.")
        # Generate synthetic data
        X = np.random.randn(500, 2, 128, 128).astype(np.float32)
        Y = np.random.randn(500, 2, 128, 128).astype(np.float32)

    X = jnp.array(X)
    Y = jnp.array(Y)

    spatial_resolutions = [32, 64, 128]
    width = 64
    modes = (12, 12)  # 2D Fourier modes
    n_layers = 4

    stats = []

    for res in spatial_resolutions:
        print(
            f"\nBenchmarking resolution: {res}x{res} with batch size: {inf_batch_size}"
        )
        # Resize grid
        H, W = X.shape[-2], X.shape[-1]
        indices_h = np.linspace(0, H - 1, res).astype(int)
        indices_w = np.linspace(0, W - 1, res).astype(int)
        x_res = X[:, :, indices_h, :][:, :, :, indices_w]

        jax_jit_compile, jax_runtime = benchmark_jax_2d(
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
        f"\nBenchmarking 2D JAX Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    # Take a batch of train_batch_size at 128x128 resolution
    x_train = X[:train_batch_size]
    y_train = Y[:train_batch_size]

    key = jr.key(0)
    jax_model = FNO(
        key=key,
        in_channels=2,
        out_channels=2,
        hidden_channels=width,
        n_layers=n_layers,
        modes=modes,
    )
    _, jax_train_compile, jax_train_time = train_jax_2d(
        jax_model, x_train, y_train, steps=train_steps
    )

    print(f"JAX 2D Training Compiled in: {jax_train_compile:.4f}s")
    print(f"JAX 2D Training Execution ({train_steps} steps): {jax_train_time:.4f}s")

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
    argparser.add_argument("--dataset_file", type=str, default="FNS-KF.nc")
    argparser.add_argument("--stat_file", type=str, required=True)
    argparser.add_argument("--train_steps", type=int, default=500)
    argparser.add_argument("--train_batch_size", type=int, default=32)
    argparser.add_argument("--inf_batch_size", type=int, default=16)
    argparser.add_argument("--n_batches", type=int, default=30)
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
