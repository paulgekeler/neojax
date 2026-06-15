"""Benchmark torch performance."""

import argparse
import json
import time

import numpy as np
import torch
from neuralop.models import FNO


def synchronize_device(device):
    """Synchronize the specified device."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def benchmark_torch(
    n_batches, batch_size, n_spatial, modes, width, n_layers, data_x, device="cuda"
):
    """Benchmarks PyTorch inference performance (Eager and Compiled).

    Args:
        n_batches: Number of batches to run.
        batch_size: Batch size of each run.
        n_spatial: Spatial resolution.
        modes: Number of modes.
        width: Hidden width.
        n_layers: Number of FNO layers.
        data_x: Input data for benchmarking.
        device: Device to run on.

    Returns:
        Tuple of (Eager runtime, Compile time, Compiled runtime).
    """
    if isinstance(modes, int):
        modes = (modes,)

    torch_model = FNO(
        n_modes=modes,
        hidden_channels=width,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    ).to(device)

    # Prepare batches
    batches = []
    for i in range(n_batches):
        start_idx = (i * batch_size) % len(data_x)
        end_idx = start_idx + batch_size
        batch = data_x[start_idx:end_idx]
        if len(batch) < batch_size:
            extra = data_x[: batch_size - len(batch)]
            batch = torch.cat([batch, extra], dim=0)
        batches.append(batch.to(device))

    # Standard Torch Inference (Eager)
    torch_model.eval()
    with torch.no_grad():
        # Warmup
        _ = torch_model(batches[0])
        synchronize_device(device)

        start = time.time()
        for i in range(n_batches):
            _ = torch_model(batches[i])
        synchronize_device(device)
        torch_runtime = (time.time() - start) / n_batches

    # Torch Compile
    start = time.time()
    compiled_model = torch.compile(torch_model)
    # Warmup / Compile trigger with the exact batch shape
    with torch.no_grad():
        _ = compiled_model(batches[0])
    synchronize_device(device)
    torch_compile_time = time.time() - start

    # Compiled Inference
    with torch.no_grad():
        start = time.time()
        for i in range(n_batches):
            _ = compiled_model(batches[i])
        synchronize_device(device)
        torch_compiled_runtime = (time.time() - start) / n_batches

    return torch_runtime, torch_compile_time, torch_compiled_runtime


def train_torch_eager(model, train_x, train_y, lr=1e-3, steps=100, device="cuda"):
    """Trains the eager PyTorch model and measures time.

    Args:
        model: TorchFNO model.
        train_x: Training inputs.
        train_y: Training labels.
        lr: Learning rate.
        steps: Number of training steps.
        device: Device to run on.

    Returns:
        Training step execution time.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    # Warmup
    pred = model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    synchronize_device(device)

    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
    synchronize_device(device)
    elapsed = time.time() - start
    return elapsed


def train_torch_compiled(model, train_x, train_y, lr=1e-3, steps=100, device="cuda"):
    """Trains the compiled PyTorch model and measures time, excluding compilation.

    Args:
        model: TorchFNO model.
        train_x: Training inputs.
        train_y: Training labels.
        lr: Learning rate.
        steps: Number of training steps.
        device: Device to run on.

    Returns:
        Tuple of (compile time, training step execution time).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    compiled_model = torch.compile(model)

    # Warmup / Compilation step
    start_comp = time.time()
    pred = compiled_model(train_x)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    synchronize_device(device)
    compile_time = time.time() - start_comp

    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = compiled_model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
    synchronize_device(device)
    elapsed = time.time() - start
    return compile_time, elapsed


def run_benchmarks(
    dataset_file: str,
    stat_file: str,
    train_steps: int,
    train_batch_size: int,
    inf_batch_size: int,
    n_batches: int,
    device: str,
):
    """Run neuraloperator benchmarks."""
    with np.load(dataset_file) as dataset:
        X = dataset["X"]
        Y = dataset["Y"]

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    spatial_resolutions = [128, 256, 512, 1024]
    width = 64
    modes = 16
    n_layers = 4

    stats = []

    for res in spatial_resolutions:
        print(f"\nBenchmarking resolution: {res} with batch size: {inf_batch_size}")
        # Resize data for resolution
        indices = torch.linspace(0, X.shape[-1] - 1, res).long()
        x_res = X[:, :, indices]

        torch_runtime, torch_compile_time, torch_compiled_runtime = benchmark_torch(
            n_batches,
            inf_batch_size,
            res,
            modes,
            width,
            n_layers,
            x_res,
            device,
        )

        stats.append(
            {
                "Type": "Inference",
                "Resolution": res,
                "Implementation": "PyTorch (Eager)",
                "Compile Time (s)": 0,
                "Inference Time (ms)": torch_runtime * 1000,
                "Batch Size": inf_batch_size,
            }
        )
        stats.append(
            {
                "Type": "Inference",
                "Resolution": res,
                "Implementation": "PyTorch (Compiled)",
                "Compile Time (s)": torch_compile_time,
                "Inference Time (ms)": torch_compiled_runtime * 1000,
                "Batch Size": inf_batch_size,
            }
        )

    # Training Comparison
    print(
        f"\nBenchmarking PyTorch Eager Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    indices_1024 = torch.linspace(0, X.shape[-1] - 1, 1024).long()
    x_train = X[:train_batch_size, :, indices_1024].to(device)
    y_train = Y[:train_batch_size, :, indices_1024].to(device)

    # PyTorch Eager Training
    torch_model_eager = FNO(
        n_modes=(modes,),
        hidden_channels=width,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    ).to(device)

    torch_train_eager_time = train_torch_eager(
        torch_model_eager, x_train, y_train, steps=train_steps, device=device
    )
    print(
        f"PyTorch Eager Training Execution ({train_steps} steps): {torch_train_eager_time:.4f}s"
    )

    stats.append(
        {
            "Type": "Training",
            "Implementation": "PyTorch (Eager)",
            "Compile Time (s)": 0.0,
            "Training Time (s)": torch_train_eager_time,
            "Steps": train_steps,
            "Batch Size": train_batch_size,
        }
    )

    print(
        f"\nBenchmarking PyTorch Compiled Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    # PyTorch Compiled Training
    torch_model_compiled = FNO(
        n_modes=(modes,),
        hidden_channels=width,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    ).to(device)

    torch_train_compile_time, torch_train_compiled_time = train_torch_compiled(
        torch_model_compiled, x_train, y_train, steps=train_steps, device=device
    )
    print(f"PyTorch Compiled Training Compiled in: {torch_train_compile_time:.4f}s")
    print(
        f"PyTorch Compiled Training Execution ({train_steps} steps): {torch_train_compiled_time:.4f}s"
    )

    stats.append(
        {
            "Type": "Training",
            "Implementation": "PyTorch (Compiled)",
            "Compile Time (s)": torch_train_compile_time,
            "Training Time (s)": torch_train_compiled_time,
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
    argparser.add_argument("--device", type=str, default="cuda")
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device detection and fallback
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"CUDA not available. Falling back to device: {device}")

    run_benchmarks(
        args.dataset_file,
        args.stat_file,
        args.train_steps,
        args.train_batch_size,
        args.inf_batch_size,
        args.n_batches,
        device,
    )
