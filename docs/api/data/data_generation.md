# Data Generation API

**Neojax** provides some data generation functions. It currently only supports generating data for Burgers equation.

### Setup

Generating data requires the installation of the optional `diffrax` dependency.

```bash
pip3 install "neojax-operators[gen]"
```

or `pip3 install ".[gen]"` from the project root.

## 1D Burgers Equation

Burgers equation is a classical example of a 1D non-linear PDE used in Neural Operator publications:

$$
\begin{aligned}
\frac{\partial}{\partial t}u(x, t) + \frac{1}{2}\frac{\partial}{\partial x}(u(x,t))^2 &= \nu \frac{\partial^2}{\partial x^2}u(x,t), \; \quad x \in (0,2\pi), t \in (0, \infty) \\
u(x, 0) &= u_0(x), \quad \quad \quad \quad x \in (0, 2\pi)
\end{aligned}
$$

with a fixed viscosity $\nu = 10^{-1}$. The initial condition is drawn from $\mu = \mathcal{N}(0, C)$, i.e., $u_0 \sim \mu$, where

$$
C = 625(-\frac{d^2}{dx^2} + 25I)^{-2}.
$$

::: neojax.data.generation.burgers_1d.generate_burgers_1d