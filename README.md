# neojax

**neojax** (**Ne**ural **O**perators in JAX) is an implementation of Neural Operators built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). It provides a clean, modular API inspired by the original [neuraloperator](https://github.com/neuraloperator/neuraloperator) library.

Currently, **neojax** is in its early stages. Only the **Fourier Neural Operator (FNO)** is available, with support for:
- 1D, 2D, 3D, 4D, and 5D coordinate-based inputs.
- Domain Padding.
- Channel-MLP (Pointwise MLP) expansions.
- Various skip connections (Soft-Gating, Linear, Identity).

It is designed to be fully compatible with all JAX features such as `vmap`, `jit`, and `grad`.

#### Installation
Install the python package via pypi
```bash
pip3 install neojax-operators
```

#### Quickstart
neojax exposes a similar API to neuraloperators and equinox and should therefore be familiar to use:
```python
from neojax.models import FNO
fno = FNO(n_modes=(64, 64),
        hidden_channels=64,
        in_channels=2,
        out_channels=1
    )
```

For an introduction to model training in equinox, please refer to the respective [introductory guide](https://docs.kidger.site/equinox/examples/mnist/).

Example notebooks will follow soon.

#### Benchmarks
Coming soon (Please refer to `BENCHMARKS.md` for a performance comparison of `neojax` and `neuraloperator`.)

#### Motivation
Jax has become ubiquitous in Scientific Machine Learning (SciML) and Scientific Computing. This is largely due to its core design, which embraces mathematical and functional transformations (like `jit`, `vmap`, and `grad`) and seamlessly integrates with NumPy-like paradigms. However, despite Neural Operators fundamentally shaping the SciML landscape and being frequently used for solving PDEs, a comprehensive, native Jax implementation has been notably missing. `neojax` was created to bridge this gap, bringing the performance, predictability, and ecosystem of Jax to the Neural Operator community.

#### Design Choices
Although originally conceived as a direct port of the PyTorch `neuraloperator` library, `neojax` evolved into a ground-up, Jax-native re-implementation. This approach avoids the pitfalls of forcing PyTorch idioms into a functional framework and significantly reduces internal complexity. By building directly on `equinox`, `neojax` aligns perfectly with Jax's pure-functional design principles while maintaining a clean, accessible, and class-based API.

#### Roadplan
The first and currently only supported Neural Operator is a simple Fourier Neural Operator (FNO). In upcoming releases more models and components will be added.

#### Contributions
If you'd like to contribute any features, models, or fix implementation errors, please do so. Any contributions are appreciated. Have a look at the `CONTRIBUTING.md` guide for details on how to do so.

#### Citation
If you use `neojax` in your research, please cite it using the following BibTeX entry:

```bibtex
@software{neojax,
  author = {Paul Gekeler},
  title = {neojax: Neural Operators in Jax},
  year = {2026},
  url = {https://github.com/paulgekeler/neojax}
}
```
