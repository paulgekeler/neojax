# neojax

> [!WARNING]
> **Work in Progress:** This project is in its early stages of development and is currently **not functional**. It is being shared for development and architectural purposes only.

This is neojax (**Ne**ural **O**perators in **jax**), a port of the popular [neural operators](https://github.com/neuraloperator/neuraloperator) pytorch library to jax. It is built on top of [jax](https://github.com/jax-ml/jax) and [equinox](https://github.com/patrick-kidger/equinox) and provides the same API as the original pytorch library.

#### Installation
Install the python package via pypi
```bash
pip3 install neojax-operators
```

#### Quickstart
neojax exposes the same API as neuraloperators and can therefore be used as a drop-in replacement:
```python
from neojax.models import FNO
operator = FNO(n_modes=(64, 64),
               hidden_channels=64,
               in_channels=2,
               out_channels=1)
```