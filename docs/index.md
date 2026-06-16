# Neojax

**neojax** (**Ne**ural **O**perators in JAX) is an implementation of Neural Operators built on top of [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). It provides a clean, modular API inspired by the original [neuraloperator](https://github.com/neuraloperator/neuraloperator) library.

Currently, **neojax** is in its early stages. Only the following models and features are available:

- **Fourier Neural Operator (FNO)**.
  - Symmetrical domain padding (`DomainPadding`).
  - Pointwise MLP (Channel-MLP) expansions for improved expressivity.
  - Various skip connections (Linear, Soft-Gating, Identity).
  - Normalization in FNO blocks.
- **Tucker-factorized FNO**
- **Deep Operator Network (DeepONet)**.
- N-dimensional coordinate-based operator learning.
- Grid-based positional embeddings (`GridEmbeddingNd`).
- (Relative) $L^{p}$-loss.
- General $W^{k,p}$ Sobolev loss.
- Loss Compositions.
- Data Normalization and Scaling.
  - Various Normalizer classes.
  - Physical scales to non-dimensionalize inputs.

It is designed to be fully compatible with all JAX features such as `vmap`, `jit`, and `grad`.

#### Installation
Install the python package via pypi
```bash
pip3 install neojax-operators
```

#### Quickstart
neojax exposes a similar API to neuraloperators and equinox and should therefore be familiar to use:
```python
import jax.numpy as jnp
from neojax.models import FNO

fno = FNO(modes=(12, 12),
        hidden_channels=64,
        in_channels=2,
        out_channels=1
    )

x = jnp.ones((2, 64, 64))

pred = fno(x)
```

For a more detailed introduction refer to the [Examples](examples/01_basics.md).

#### Benchmarks
Please refer to the [Benchmarks](benchmarks.md).

#### Motivation
Jax has become ubiquitous in Scientific Machine Learning (SciML) and Scientific Computing. This is largely due to its core design, which embraces mathematical and functional transformations (like `jit`, `vmap`, and `grad`) and seamlessly integrates with NumPy-like paradigms. However, despite Neural Operators fundamentally shaping the SciML landscape and being frequently used for solving PDEs, a comprehensive, native Jax implementation has been notably missing. `neojax` was created to bridge this gap, bringing the performance, predictability, and ecosystem of Jax to the Neural Operator community.

#### Design Choices
Although originally conceived as a direct port of the PyTorch `neuraloperator` library, `neojax` evolved into a ground-up, Jax-native re-implementation. This approach avoids the pitfalls of forcing PyTorch idioms into a functional framework and significantly reduces internal complexity. By building directly on `equinox`, `neojax` aligns perfectly with Jax's pure-functional design principles while maintaining a clean, accessible, and class-based API.

#### Roadplan
The first supported Neural Operators are a Fourier Neural Operator (FNO), Tucker-factorized FNO and a DeepONet. In upcoming releases more models and components will be added in roughly the following order:

1. Resampling of outputs to arbitrary domain sizes
2. UNO, LocalNO, SFNO, RNO, and others
3. Dataset utilities using `grain`
4. Training utilities
5. Graph Neural Operators (need some type of point-cloud support first)
6. Domain class to wrap regular grid and irregular mesh domains
7. Irregular mesh domains (probably `meshio` integration for parsing formats)
8. Domain utilities

:fontawesome-solid-arrow-right: mid- to long-term transitioning to a domain/operator-centric approach architecture, allowing for arbitrary domains


#### Contributions
If you'd like to contribute any features, models, or fix implementation errors, please do so. Any contributions are appreciated. Have a look at the `CONTRIBUTING.md` guide for details on how to do so. I am also open to advice on restructuring and any other design choices that could be improved.

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
