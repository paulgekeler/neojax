"""Plot the benchmarking results."""

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(
    stat_file_neop: str,
    stat_file_nj: str,
    benchmark_imgs_dir: str,
    output_name: str = "performance_dashboard.png",
):
    """Plot comparison results."""
    with open(stat_file_neop) as f:
        stats_neuralop = json.load(f)

    with open(stat_file_nj) as f:
        stats_neojax = json.load(f)

    # Detect schema version and parse
    is_new_schema = any("Type" in entry for entry in stats_neojax + stats_neuralop)

    if is_new_schema:
        neojax_inf = [e for e in stats_neojax if e.get("Type") == "Inference"]
        neuralop_inf = [e for e in stats_neuralop if e.get("Type") == "Inference"]
        df_inf = pd.DataFrame(neojax_inf + neuralop_inf)

        neojax_train = [e for e in stats_neojax if e.get("Type") == "Training"]
        neuralop_train = [e for e in stats_neuralop if e.get("Type") == "Training"]
        df_train = pd.DataFrame(neojax_train + neuralop_train)

        train_steps = 1000
        train_bs = 64
        if not df_train.empty:
            train_steps = df_train.iloc[0].get("Steps", 1000)
            train_bs = df_train.iloc[0].get("Batch Size", 64)
    else:
        # Fallback to old schema parsing
        jax_train_time = stats_neojax[-1].get(
            "Training Time (s)", stats_neojax[-1].get("Training Time")
        )
        torch_train_time = stats_neuralop[-1].get(
            "Training Time (s)", stats_neuralop[-1].get("Training Time")
        )
        df_train = pd.DataFrame(
            [
                {
                    "Implementation": "JAX (JIT)",
                    "Training Time (s)": jax_train_time,
                    "Compile Time (s)": 0.0,
                },
                {
                    "Implementation": "PyTorch (Eager)",
                    "Training Time (s)": torch_train_time,
                    "Compile Time (s)": 0.0,
                },
            ]
        )
        train_steps = 100
        train_bs = 100
        df_inf = pd.DataFrame(stats_neojax[:-1] + stats_neuralop[:-1])

    print("\n--- Training Performance Comparison ---")
    for _, row in df_train.iterrows():
        print(f"{row['Implementation']}: {row['Training Time (s)']:.4f}s")

    # Print speedup compared to PyTorch Eager
    eager_row = df_train[df_train["Implementation"] == "PyTorch (Eager)"]
    if not eager_row.empty:
        eager_time = eager_row.iloc[0]["Training Time (s)"]
        for _, row in df_train.iterrows():
            if row["Implementation"] != "PyTorch (Eager)":
                speedup = eager_time / row["Training Time (s)"]
                print(
                    f"Speedup of {row['Implementation']} vs PyTorch (Eager): {speedup:.2f}x"
                )

    # Print speedup of JAX JIT vs PyTorch Compiled
    compiled_row = df_train[df_train["Implementation"] == "PyTorch (Compiled)"]
    jax_row = df_train[df_train["Implementation"] == "JAX (JIT)"]
    if not compiled_row.empty and not jax_row.empty:
        torch_time = compiled_row.iloc[0]["Training Time (s)"]
        jax_time = jax_row.iloc[0]["Training Time (s)"]
        print(f"JAX (JIT) Speedup vs PyTorch (Compiled): {torch_time / jax_time:.2f}x")

    # Set up 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Inference Latency (Log-Log) [Top-Left]
    sns.lineplot(
        data=df_inf,
        x="Resolution",
        y="Inference Time (ms)",
        hue="Implementation",
        marker="o",
        ax=axes[0, 0],
    )
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    inf_bs = df_inf.iloc[0].get("Batch Size", 1) if not df_inf.empty else 1
    axes[0, 0].set_title(f"Inference Latency (Log-Log, Steady-State BS={inf_bs})")
    axes[0, 0].set_ylabel("Latency per Batch (ms)")
    axes[0, 0].grid(True, which="both", ls="-", alpha=0.5)

    # 2. Relative Speedup vs PyTorch Eager [Top-Right]
    eager_times = df_inf[df_inf["Implementation"] == "PyTorch (Eager)"].set_index(
        "Resolution"
    )["Inference Time (ms)"]
    df_inf["Speedup vs Eager"] = df_inf.apply(
        lambda row: eager_times[row["Resolution"]] / row["Inference Time (ms)"],
        axis=1,
    )

    sns.barplot(
        data=df_inf[df_inf["Implementation"] != "PyTorch (Eager)"],
        x="Resolution",
        y="Speedup vs Eager",
        hue="Implementation",
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(1.0, ls="--", color="red", alpha=0.6)
    axes[0, 1].set_title(
        "Relative Inference Speedup vs PyTorch Eager (Higher is Better)"
    )
    axes[0, 1].set_ylabel("Speedup (x)")

    # 3. Training Comparison (Execution Time only) [Bottom-Left]
    sns.barplot(data=df_train, x="Implementation", y="Training Time (s)", ax=axes[1, 0])
    axes[1, 0].set_title(
        f"Training Execution Time for {train_steps} Steps (BS={train_bs})"
    )
    axes[1, 0].set_ylabel("Execution Time (seconds)")
    axes[1, 0].set_xlabel("")

    # 4. Compilation Overhead Comparison [Bottom-Right]
    compile_data = []
    try:
        # Find resolution (could be 128 for 1D, or 32 for 2D)
        resolutions = sorted(df_inf["Resolution"].unique())
        first_res = resolutions[0] if len(resolutions) > 0 else 128

        jax_inf_comp = df_inf[
            (df_inf["Implementation"] == "JAX (Equinox/JIT)")
            & (df_inf["Resolution"] == first_res)
        ].iloc[0]["Compile Time (s)"]
        torch_inf_comp = df_inf[
            (df_inf["Implementation"] == "PyTorch (Compiled)")
            & (df_inf["Resolution"] == first_res)
        ].iloc[0]["Compile Time (s)"]
        compile_data.append(
            {
                "Implementation": "JAX (JIT)",
                "Task": f"Inference Compile (Res {first_res})",
                "Compile Time (s)": jax_inf_comp,
            }
        )
        compile_data.append(
            {
                "Implementation": "PyTorch (Compiled)",
                "Task": f"Inference Compile (Res {first_res})",
                "Compile Time (s)": torch_inf_comp,
            }
        )
    except Exception:
        pass  # Skip if inference compile keys not found

    try:
        jax_train_comp = df_train[df_train["Implementation"] == "JAX (JIT)"].iloc[0][
            "Compile Time (s)"
        ]
        torch_train_comp = df_train[
            df_train["Implementation"] == "PyTorch (Compiled)"
        ].iloc[0]["Compile Time (s)"]
        compile_data.append(
            {
                "Implementation": "JAX (JIT)",
                "Task": "Training Compile",
                "Compile Time (s)": jax_train_comp,
            }
        )
        compile_data.append(
            {
                "Implementation": "PyTorch (Compiled)",
                "Task": "Training Compile",
                "Compile Time (s)": torch_train_comp,
            }
        )
    except Exception:
        pass  # Skip if training compile keys not found

    if compile_data:
        df_compile = pd.DataFrame(compile_data)
        sns.barplot(
            data=df_compile,
            x="Task",
            y="Compile Time (s)",
            hue="Implementation",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Compilation Time (One-time JIT Overhead)")
        axes[1, 1].set_ylabel("Compilation Time (seconds)")
        axes[1, 1].set_xlabel("")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Compilation data not available",
            ha="center",
            va="center",
        )
        axes[1, 1].set_title("Compilation Time")

    import os

    os.makedirs(benchmark_imgs_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{benchmark_imgs_dir}/{output_name}", dpi=300)

    print(
        f"\nBenchmarks completed. Results saved in '{benchmark_imgs_dir}/{output_name}'."
    )


def parse_args():
    """Parse cmd args."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--stat_file_neojax", type=str, required=True)
    argparser.add_argument("--stat_file_neuralop", type=str, required=True)
    argparser.add_argument("--benchmark_imgs_dir", type=str, required=True)
    argparser.add_argument(
        "--output_name", type=str, default="performance_dashboard.png"
    )
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_results(
        args.stat_file_neuralop,
        args.stat_file_neojax,
        args.benchmark_imgs_dir,
        args.output_name,
    )
