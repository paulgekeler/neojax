"""Benchmark 2D FNO performance in PyTorch using camlab-ethz/FNS-KF."""

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


def benchmark_torch_2d(
    n_batches,
    batch_size,
    n_spatial,
    modes,
    width,
    n_layers,
    data_x,
    device="cuda",
):
    """Benchmarks 2D PyTorch inference performance (Eager and Compiled).

    Args:
        n_batches: Number of batches.
        batch_size: Batch size.
        n_spatial: Spatial resolution.
        modes: Tuple of modes.
        width: Hidden width.
        n_layers: Number of FNO layers.
        data_x: Input data for benchmarking.
        device: Device to run on.

    Returns:
        Tuple of (Eager runtime, Compile time, Compiled runtime).
    """
    torch_model = FNO(
        n_modes=modes,  # (modes, modes)
        hidden_channels=width,
        in_channels=2,
        out_channels=2,
        n_layers=n_layers,
    ).to(device)

    # Prepare batch data
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


def train_torch_eager_2d(model, train_x, train_y, lr=1e-3, steps=100, device="cuda"):
    """Trains the eager 2D PyTorch model and measures time.

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


def train_torch_compiled_2d(model, train_x, train_y, lr=1e-3, steps=100, device="cuda"):
    """Trains the compiled 2D PyTorch model and measures time, excluding compilation.

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
    """Run 2D PyTorch benchmarks."""
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
                Y = sol[:train_size, 10]  # target shape: (N, 2, 128, 128)
            else:
                raise KeyError("'solution' not found in dataset file")
    except Exception as e:
        print(f"Could not load NetCDF dataset: {e}. Falling back to synthetic 2D data.")
        # Generate synthetic data
        X = np.random.randn(500, 2, 128, 128).astype(np.float32)
        Y = np.random.randn(500, 2, 128, 128).astype(np.float32)

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    spatial_resolutions = [32, 64, 128]
    width = 64
    modes = (12, 12)  # 2D modes
    n_layers = 4

    stats = []

    for res in spatial_resolutions:
        print(
            f"\nBenchmarking resolution: {res}x{res} with batch size: {inf_batch_size}"
        )
        # Resize grid
        H, W = X.shape[-2], X.shape[-1]
        indices_h = torch.linspace(0, H - 1, res).long()
        indices_w = torch.linspace(0, W - 1, res).long()
        x_res = X[:, :, indices_h, :][:, :, :, indices_w]

        torch_runtime, torch_compile_time, torch_compiled_runtime = benchmark_torch_2d(
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
        f"\nBenchmarking 2D PyTorch Eager Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    x_train = X[:train_batch_size].to(device)
    y_train = Y[:train_batch_size].to(device)

    # PyTorch Eager Training
    torch_model_eager = FNO(
        n_modes=modes,
        hidden_channels=width,
        in_channels=2,
        out_channels=2,
        n_layers=n_layers,
    ).to(device)

    torch_train_eager_time = train_torch_eager_2d(
        torch_model_eager, x_train, y_train, steps=train_steps, device=device
    )
    print(
        f"PyTorch 2D Eager Training Execution ({train_steps} steps): {torch_train_eager_time:.4f}s"
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
        f"\nBenchmarking 2D PyTorch Compiled Training performance ({train_steps} steps, batch size: {train_batch_size})..."
    )
    # PyTorch Compiled Training
    torch_model_compiled = FNO(
        n_modes=modes,
        hidden_channels=width,
        in_channels=2,
        out_channels=2,
        n_layers=n_layers,
    ).to(device)

    torch_train_compile_time, torch_train_compiled_time = train_torch_compiled_2d(
        torch_model_compiled,
        x_train,
        y_train,
        steps=train_steps,
        device=device,
    )
    print(f"PyTorch 2D Compiled Training Compiled in: {torch_train_compile_time:.4f}s")
    print(
        f"PyTorch 2D Compiled Training Execution ({train_steps} steps): {torch_train_compiled_time:.4f}s"
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
    argparser.add_argument("--dataset_file", type=str, default="FNS-KF.nc")
    argparser.add_argument("--stat_file", type=str, required=True)
    argparser.add_argument("--train_steps", type=int, default=500)
    argparser.add_argument("--train_batch_size", type=int, default=32)
    argparser.add_argument("--inf_batch_size", type=int, default=16)
    argparser.add_argument("--n_batches", type=int, default=30)
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
