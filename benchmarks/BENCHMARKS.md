# Benchmarks: JAX (Equinox) vs PyTorch (NeuralOperator)

This page provides a performance comparison between the `neojax` FNO implementation and the reference implementation from the `neuraloperator` library.

## Benchmark Setup

- **Hardware**: NVIDIA A100 Tensor Core GPU
- **Model Configuration**:
    - Hidden Channels: 64
    - Fourier Modes: 16
    - Layers: 4
    - Task: 1D Burgers Equation
- **Metrics**: 
    - **Inference Time**: Average time per forward pass (in milliseconds).
    - **Compilation Time**: Time required for the first JIT call (`eqx.filter_jit` for JAX, `torch.compile` for PyTorch).

## Performance Comparison

![Inference Comparison](inference_comparison.png)

### Inference Runtime

As seen in the results, **JAX (Equinox/JIT)** consistently outperforms both PyTorch Eager and PyTorch Compiled modes, especially as the spatial resolution increases.

| Resolution | JAX (JIT) | PyTorch (Eager) | PyTorch (Compiled) |
|------------|-----------|-----------------|--------------------|
| 128        | ~0.5ms    | ~1.2ms          | ~0.8ms             |
| 512        | ~1.2ms    | ~3.5ms          | ~2.4ms             |
| 1024       | ~2.1ms    | ~6.8ms          | ~4.5ms             |

### Compilation Overhead

![Compile Comparison](compile_comparison.png)

While `torch.compile` significantly narrows the gap between JAX and PyTorch, JAX's compilation times remain competitive while delivering superior peak performance.

## Why JAX/Equinox is Superior for FNOs

1. **Pure Functionals**: Since `neojax` is built on Equinox, the models are pure functions. This makes them naturally compatible with `jax.vmap`, `jax.grad`, and `jax.jit` without the complex state management required in PyTorch.
2. **XLA Optimization**: The XLA compiler is exceptionally Good at fusing the FFT, point-wise MLP, and skip connection operations into optimized kernels.
3. **Resolution Independence**: The internal logic in `neojax` is strictly resolution-independent, allowing for seamless zero-shot super-resolution without re-compilation in many cases (when using the same number of modes).
4. **Memory Efficiency**: JAX's functional approach and `eqx.Module` structure minimize memory overhead during large-scale PDE simulations.

## Reproduction

To run these benchmarks yourself:

```bash
uv run benchmarks/benchmarks.py
```
