import time

import equinox as eqx
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from neuraloperator.models import FNO as TorchFNO

from neojax.models import FNO as JaxFNO


def get_burgers_data(n_samples, n_spatial, key):
    # Simplified data gen similar to the notebook but enough for benchmarking
    # Shape: (samples, channels, spatial)
    x_key, y_key = jr.split(key)
    x = jr.normal(x_key, (n_samples, 1, n_spatial))
    y = jr.normal(y_key, (n_samples, 1, n_spatial))
    return x, y


def benchmark_jax(n_samples, n_spatial, modes, width, n_layers):
    key = jr.PRNGKey(0)
    model_key, data_key = jr.split(key)

    # Init Jax Model
    jax_model = JaxFNO(
        key=model_key,
        in_channels=1,
        out_channels=1,
        hidden_channels=width,
        n_layers=n_layers,
        modes=modes,
    )

    x, _ = get_burgers_data(n_samples, n_spatial, data_key)

    # 1. JIT Compile Time
    start = time.time()
    jax_fn = eqx.filter_jit(jax_model)
    # Warmup
    _ = jax_fn(x[0]).block_until_ready()
    jit_compile_time = time.time() - start

    # 2. Inference (JIT)
    start = time.time()
    for i in range(n_samples):
        _ = jax_fn(x[i]).block_until_ready()
    jax_jit_runtime = (time.time() - start) / n_samples

    return jit_compile_time, jax_jit_runtime


def benchmark_torch(n_samples, n_spatial, modes, width, n_layers, device="cuda"):
    # Convert modes to tuple if int
    if isinstance(modes, int):
        modes = (modes,)

    torch_model = TorchFNO(
        n_modes=modes,
        hidden_channels=width,
        in_channels=1,
        out_channels=1,
        n_layers=n_layers,
    ).to(device)

    x = torch.randn(n_samples, 1, n_spatial).to(device)

    # 1. Standard Torch Inference
    torch_model.eval()
    with torch.no_grad():
        # Warmup
        _ = torch_model(x[0:1])
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for i in range(n_samples):
            _ = torch_model(x[i : i + 1])
        if device == "cuda":
            torch.cuda.synchronize()
        torch_runtime = (time.time() - start) / n_samples

    # 2. Torch Compile
    start = time.time()
    compiled_model = torch.compile(torch_model)
    # Warmup / Compile trigger
    with torch.no_grad():
        _ = compiled_model(x[0:1])
    if device == "cuda":
        torch.cuda.synchronize()
    torch_compile_time = time.time() - start

    # 3. Compiled Inference
    with torch.no_grad():
        start = time.time()
        for i in range(n_samples):
            _ = compiled_model(x[i : i + 1])
        if device == "cuda":
            torch.cuda.synchronize()
        torch_compiled_runtime = (time.time() - start) / n_samples

    return torch_runtime, torch_compile_time, torch_compiled_runtime


def run_benchmarks():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on {device}...")

    spatial_resolutions = [128, 256, 512, 1024]
    width = 64
    modes = 16
    n_layers = 4
    n_samples = 50

    results = []

    for res in spatial_resolutions:
        print(f"Benchmarking resolution: {res}")
        jax_jit_compile, jax_runtime = benchmark_jax(
            n_samples, res, modes, width, n_layers
        )
        torch_runtime, torch_compile_time, torch_compiled_runtime = benchmark_torch(
            n_samples, res, modes, width, n_layers, device
        )

        results.append(
            {
                "Resolution": res,
                "Implementation": "JAX (Equinox/JIT)",
                "Compile Time (s)": jax_jit_compile,
                "Inference Time (ms)": jax_runtime * 1000,
            }
        )
        results.append(
            {
                "Resolution": res,
                "Implementation": "PyTorch (Eager)",
                "Compile Time (s)": 0,
                "Inference Time (ms)": torch_runtime * 1000,
            }
        )
        results.append(
            {
                "Resolution": res,
                "Implementation": "PyTorch (Compiled)",
                "Compile Time (s)": torch_compile_time,
                "Inference Time (ms)": torch_compiled_runtime * 1000,
            }
        )

    df = pd.DataFrame(results)

    # Plotting
    sns.set_theme(style="whitegrid")

    # Inference Time Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="Resolution",
        y="Inference Time (ms)",
        hue="Implementation",
        marker="o",
    )
    plt.title("FNO Inference Performance: JAX vs PyTorch (A100)")
    plt.ylabel("Inference Time (ms) - Lower is better")
    plt.savefig("benchmarks/inference_comparison.png")

    # Compile Time Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df[df["Compile Time (s)"] > 0],
        x="Resolution",
        y="Compile Time (s)",
        hue="Implementation",
    )
    plt.title("JIT/Compile Time Comparison")
    plt.savefig("benchmarks/compile_comparison.png")

    print("Benchmarks completed. Results saved in 'benchmarks/' directory.")
    return df


if __name__ == "__main__":
    run_benchmarks()
