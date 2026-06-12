### Example 2: Training a FNO on Burgers Equation with **neojax**

[:material-download: Download Notebook](02_fno_burgers.ipynb){ .md-button }

In this second example, we showcase how to train a Fourier Neural Operator on Burgers equation.

To run this example, we first install and import the necessary python dependencies:

```python
!pip3 install equinox diffrax optax neojax-operators jaxtyping
```

```python
# we use diffrax to solve burgers equation and generate data
import diffrax
import equinox as eqx
# we use optax for gradient optimizers
import optax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from neojax.models import FNO
from neojax.nn import GridEmbeddingNd
```

#### Burgers Equation
Burgers equation is a classical example of a 1d non-linear PDE used in Neural Operator publications:

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

(Skip the following part if you are more interested in model training and performance).

#### Generating data
We solve Burgers equation using a pseudo-spectral split step method (see [^1]), implemented in **Diffrax**. Using **Diffrax** allows us to use adaptive step sizes when solving the non-linear part which is faster than the very small fixed step size used [^1].

We generate a total of 1000 training and 200 test samples. Generation takes around 6 minutes on a CPU but is considerably faster on a GPU.

[^1]: Kovachki, N. et al. "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs". JMLR 2023, https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf.

```python
def burgers_explicit_vf(t: Array, u: Array, args: tuple[float, Array, Array]) -> Array:
    """Computes the non-linear advection term in Fourier space.

    Args:
        t: Time steps. Placeholder for compatibility.
        u: Function values.
        args: Additional vector field arguments.

    Returns:
        Non-linear advection term in Fourier space.
    """
    nu, ik, ik2 = args
    # Compute the non-linear advection term: -u * (du/dx)
    # Move back to spatial domain to handle the multiplication
    u_spatial = jnp.fft.irfft(u)
    # Compute du/dx in Fourier space, then move to spatial domain
    dudx_spatial = jnp.fft.irfft(ik * u)
    advection_spatial = -u_spatial * dudx_spatial
    # Move the advection term back to Fourier space
    advection_ft = jnp.fft.rfft(advection_spatial)
    return advection_ft

def burgers_implicit_vf(t: Array, u: Array, args: tuple[float, Array, Array]) -> Array:
    """Computes the linear diffusion term exactly in Fourier space.

    Args:
        t: Time steps. Placeholder for compatibility.
        u: Function values.
        args: Additional vector field arguments.

    Returns:
        Linear diffusion term in Fourier space.
    """
    nu, ik, ik2 = args
    # Compute the linear diffusion term exactly in Fourier space: ν * (d²u/dx²)
    # Mathematically: ν * (ik)² * û
    diffusion_ft = nu * ik2 * u
    return diffusion_ft

def solve_single_trajectory(u0_ft: Array, nu: float, ik: Array, ik2: Array, t1: float) -> Array:
    """Solves Burgers equation for a single initial condition.

    Args:
        u0_ft:
        nu: Viscosity
        ik:
        ik2:
        t1: Maximum time.

    Returns:
        Solution trajectory.
    """
    # KenCarp5 is an IMEX (Implicit-Explicit) solver. 
    # It expects a MultiTerm where the first term is treated explicitly 
    # and the second implicitly (usually the stiff part).
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(burgers_explicit_vf),
        diffrax.ODETerm(burgers_implicit_vf)
    )
    solver = diffrax.KenCarp5()
    # Save only the initial state (t=0) and the final state (t=t1) to save memory
    saveat = diffrax.SaveAt(t0=True, t1=True)

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=t1,
        dt0=0.001,
        y0=u0_ft,
        args=(nu, ik, ik2),
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5)
    )
    return sol.ys

solve_batched_trajectories = jax.vmap(
    solve_single_trajectory,
    in_axes=(0, None, None, None, None)
)

def generate_fno_initial_conditions(n_samples: int, n_grid_points: int = 8192, seed: int = 42) -> Array:
    """Generates periodic GRF initial conditions.

    Args:
        n_samples: Number of samples.
        n_grid_points: Number of spatial grid points.
        seed: Jax random seed.
    """
    key = jr.key(seed)
    r_key, i_key = jr.split(key)
    # Compute the structural wavenumbers k for a periodic [0, 1] domain
    k = jnp.fft.rfftfreq(n_grid_points, d=1.0/n_grid_points) * (2.0 * jnp.pi)
    # Sample standard complex Gaussian white noise in Fourier space
    noise_real = jr.normal(r_key, (n_samples, n_grid_points // 2 + 1))
    noise_imag = jr.normal(i_key, (n_samples, n_grid_points // 2 + 1))
    # Scale complex components to preserve variance across the FFT
    white_noise_ft = (noise_real + 1j * noise_imag) * jnp.sqrt(n_grid_points)
    # Apply the square-root covariance operator filter: C^{1/2} = 25 / (k^2 + 25)
    covariance_filter = 25.0 / (k**2 + 25.0)
    u0_ft = white_noise_ft * covariance_filter[None, :]
    return u0_ft

def normalize_datasets(train_x: Array, train_y: Array, test_x: Array, test_y: Array) -> tuple[Array, Array, Array, Array]:
    """Normalizes the train and test datasets.

    Args:
        train_x: Training input dataset.
        train_y: Training label dataset.
        test_x: Testing input dataset.
        test_y: Testing label dataset.

    Returns:
        Normalized training and test datasets.
    """
    x_mean = jnp.mean(train_x)
    x_std = jnp.std(train_x)

    y_mean = jnp.mean(train_y)
    y_std = jnp.std(train_y)

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std

    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std
    return train_x, train_y, test_x, test_y

def generate_burgers_dataset(
    n_samples: int = 1000,
    n_grid_points: int = 1024,
    nu: float = 0.1,
    t1: float = 1.0,
    batch_size: int = 100,
    seed: int = 42
) -> tuple[Array, Array]:
    """Generates Burgers equation training dataset.

    Generation is batched to limit memory usage.

    Args:
        n_samples: Number of samples.
        n_grid_points: Number of spatial grid points.
        nu: Viscosity.
        t1: Maximum time.
        batch_size: Batch size for generation to save memory.
        seed: Jax random seed.
    """
    if n_grid_points > 1024:
        import warnings
        warnings.warn(f"Resolution {n_grid_points} is high! Generation might take longer.")
    # Precompute the wavenumbers for the spectral derivatives
    k = jnp.fft.rfftfreq(n_grid_points, d=1.0 / n_grid_points) * (2.0 * jnp.pi)
    ik = 1j * k
    ik2 = -(k**2)
    print(f"Generating {n_samples} initial conditions from GRF...")
    u0_ft = generate_fno_initial_conditions(n_samples=n_samples, n_grid_points=n_grid_points, seed=seed)

    print(f"Compiling and solving physics trajectories via Diffrax at {n_grid_points} resolution...")
    jit_batched_solver = jax.jit(solve_batched_trajectories, static_argnums=(4,))

    all_results = []
    for i in range(0, n_samples, batch_size):
        print(f"Solving batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}...")
        u0_batch = u0_ft[i : i + batch_size]
        res = jit_batched_solver(u0_batch, nu, ik, ik2, t1)
        all_results.append(res)

    batched_solutions_ft = jnp.concatenate(all_results, axis=0)
    print("Converting data back to spatial domain...")
    # batched_solutions_ft shape is (num_samples, 2, N//2 + 1)
    inputs_spatial = jnp.fft.irfft(batched_solutions_ft[:, 0, :], axis=-1)
    labels_spatial = jnp.fft.irfft(batched_solutions_ft[:, 1, :], axis=-1)

    # expand channel dim for compatibility
    inputs_spatial = inputs_spatial[:, None, :]
    labels_spatial = labels_spatial[:, None, :]
    # normalize data
    inputs_spatial_train, labels_spatial_train, inputs_spatial_test, labels_spatial_test = normalize_datasets(inputs_spatial[:1000], labels_spatial[:1000], inputs_spatial[1000:], labels_spatial[1000:])
    inputs_spatial = jnp.concat([inputs_spatial_train, inputs_spatial_test], axis=0)
    labels_spatial = jnp.concat([labels_spatial_train, labels_spatial_test], axis=0)
    return inputs_spatial, labels_spatial

# generate 1000 training pairs and 200 testing pairs
X, Y = generate_burgers_dataset(n_samples=1200, n_grid_points=1024, nu=0.1, t1=1.0, batch_size=100)
```

#### Loss Function
We also use the standard relative $L_2$ error

$$
\text{Relative } L^2 \text{ Error} = \frac{\Vert \hat{y} - y\Vert_{L_2}}{\Vert y \Vert_{L_2}}
$$

were $\hat{y}$ is the network prediction and $y$ the ground truth.

```python
def rel_l2_error(y: Array, y_hat: Array) -> Array:
    """Computes average relative L2 error over a batch."""
    flat_pred = jnp.reshape(y_hat, (y_hat.shape[0], -1))
    flat_true = jnp.reshape(y, (y.shape[0], -1))

    error_norms = jnp.linalg.norm(flat_pred - flat_true, axis=-1)
    true_norms = jnp.linalg.norm(flat_true, axis=-1)

    relative_errors = error_norms / (true_norms + 1e-7)

    return jnp.mean(relative_errors)
```

#### Training
We use common training settings:
- Adam optimizer
- Cosine decay scheduler
- Train for 500 training epochs
- Initial lr of 0.001
- The hidden channel dimensions $d_{v_i}$ = 64
- The number of modes is set to 16

Because of its tight integration in the **JAX** ecosystem, we can leverage **Optax** gradient optimizers and **Equinox** weight updates during training.

```python
# Initialize the model
fno_key = jr.key(seed=1)
grid_embedding = GridEmbeddingNd(in_channels=1, ndim=1)
fno = FNO(
    key=fno_key,
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_layers=4,
    modes=(16,),
    positional_embedding=grid_embedding,
)

# Split dataset into train and test
train_x = X[:1000]
train_y = Y[:1000]
test_x = X[1000:]
test_y = Y[1000:]

# Create batch sizes and number of epochs
batch_size = 32
num_epochs = 500
key = jr.key(0)

# Compute number of total steps
total_steps = num_epochs * (len(train_x) // batch_size)

# Create a Cosine Decay Schedule
cosine_schedule = optax.schedules.cosine_decay_schedule(0.01, total_steps)
# Create Adam optimizer
optimizer = optax.adam(cosine_schedule)
# Initialize the optimizer state with the model parameters
# Here eqx.filter filters out all non-trainable parameters
opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))

# Create a loss function with our relative L2 error
def loss_fn(model, xb, y_true):
    # vmap model over batch
    y_pred = jax.vmap(model)(xb)
    return rel_l2_error(y_true, y_pred)

# Create a training step function that handles
# 1. Computation of loss and gradients
# 2. Optimizer updates
# -> We jit this block for jit-compilation
@eqx.filter_jit
def training_step(model, opt_state, xb, yb):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xb, yb)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Start the training loop
for epoch in range(num_epochs):
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, train_x.shape[0])

    for start in range(0, train_x.shape[0], batch_size):
        batch_idx = perm[start:start + batch_size]
        fno, opt_state, train_loss = training_step(
            fno,
            opt_state,
            train_x[batch_idx],
            train_y[batch_idx],
        )

    if (epoch + 1) % 10 == 0:
        test_pred = jax.vmap(fno)(test_x)
        test_rel_l2 = rel_l2_error(test_y, test_pred)
        print(
            f"Epoch {epoch + 1:03d} | loss={train_loss:.3e} "
            f"| test rel L2={test_rel_l2:.3e}"
        )
```

Although the final test loss could surely be improved by tweaking the model further, we have successfully learned the underlying operator.