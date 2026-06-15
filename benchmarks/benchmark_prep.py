"""Generate benchmarking dataset."""

import argparse
import warnings

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array


def burgers_explicit_vf(t: Array, u: Array, args: tuple[float, Array, Array]) -> Array:
    """Computes the non-linear advection term in Fourier space.

    Args:
        t: Time steps. Placeholder for compatibility.
        u: Function values in Fourier space.
        args: Additional vector field arguments (nu, ik, ik2).

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
        u: Function values in Fourier space.
        args: Additional vector field arguments (nu, ik, ik2).

    Returns:
        Linear diffusion term in Fourier space.
    """
    nu, ik, ik2 = args
    # Compute the linear diffusion term exactly in Fourier space: ν * (d²u/dx²)
    # Mathematically: ν * (ik)² * û
    diffusion_ft = nu * ik2 * u
    return diffusion_ft


def solve_single_trajectory(
    u0_ft: Array, nu: float, ik: Array, ik2: Array, t1: float
) -> Array:
    """Solves Burgers equation for a single initial condition.

    Args:
        u0_ft: Initial condition in Fourier space.
        nu: Viscosity.
        ik: Wavenumbers for first derivative.
        ik2: Wavenumbers for second derivative.
        t1: Maximum time.

    Returns:
        Solution trajectory at t0 and t1.
    """
    # KenCarp5 is an IMEX (Implicit-Explicit) solver.
    # It expects a MultiTerm where the first term is treated explicitly
    # and the second implicitly (usually the stiff part).
    terms = diffrax.MultiTerm(
        diffrax.ODETerm(burgers_explicit_vf), diffrax.ODETerm(burgers_implicit_vf)
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
        stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
    )
    return sol.ys


solve_batched_trajectories = jax.vmap(
    solve_single_trajectory, in_axes=(0, None, None, None, None)
)


def generate_fno_initial_conditions(
    n_samples: int, n_grid_points: int = 8192, seed: int = 42
) -> Array:
    """Generates periodic GRF initial conditions.

    Args:
        n_samples: Number of samples.
        n_grid_points: Number of spatial grid points.
        seed: Jax random seed.

    Returns:
        Initial conditions in Fourier space.
    """
    key = jr.key(seed)
    r_key, i_key = jr.split(key)
    # Compute the structural wavenumbers k for a periodic [0, 1] domain
    k = jnp.fft.rfftfreq(n_grid_points, d=1.0 / n_grid_points) * (2.0 * jnp.pi)
    # Sample standard complex Gaussian white noise in Fourier space
    noise_real = jr.normal(r_key, (n_samples, n_grid_points // 2 + 1))
    noise_imag = jr.normal(i_key, (n_samples, n_grid_points // 2 + 1))
    # Scale complex components to preserve variance across the FFT
    white_noise_ft = (noise_real + 1j * noise_imag) * jnp.sqrt(n_grid_points)
    # Apply the square-root covariance operator filter: C^{1/2} = 25 / (k^2 + 25)
    covariance_filter = 25.0 / (k**2 + 25.0)
    u0_ft = white_noise_ft * covariance_filter[None, :]
    return u0_ft


def normalize_datasets(
    train_x: Array, train_y: Array, test_x: Array, test_y: Array
) -> tuple[Array, Array, Array, Array]:
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
    n_samples: int = 1200,
    n_grid_points: int = 1024,
    nu: float = 0.1,
    t1: float = 1.0,
    batch_size: int = 100,
    seed: int = 42,
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

    Returns:
        Tuple of (inputs, labels) in spatial domain.
    """
    if n_grid_points > 1024:
        warnings.warn(
            f"Resolution {n_grid_points} is high! Generation might take longer.",
            stacklevel=1,
        )
    # Precompute the wavenumbers for the spectral derivatives
    k = jnp.fft.rfftfreq(n_grid_points, d=1.0 / n_grid_points) * (2.0 * jnp.pi)
    ik = 1j * k
    ik2 = -(k**2)
    print(f"Generating {n_samples} initial conditions from GRF...")
    u0_ft = generate_fno_initial_conditions(
        n_samples=n_samples, n_grid_points=n_grid_points, seed=seed
    )

    print(
        f"Compiling and solving physics trajectories via Diffrax at {n_grid_points} resolution..."
    )
    jit_batched_solver = jax.jit(solve_batched_trajectories, static_argnums=(4,))

    all_results = []
    for i in range(0, n_samples, batch_size):
        print(
            f"Solving batch {i // batch_size + 1}/{(n_samples - 1) // batch_size + 1}..."
        )
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
    (
        inputs_spatial_train,
        labels_spatial_train,
        inputs_spatial_test,
        labels_spatial_test,
    ) = normalize_datasets(
        inputs_spatial[:1000],
        labels_spatial[:1000],
        inputs_spatial[1000:1200],
        labels_spatial[1000:1200],
    )
    inputs_spatial = jnp.concatenate(
        [inputs_spatial_train, inputs_spatial_test], axis=0
    )
    labels_spatial = jnp.concatenate(
        [labels_spatial_train, labels_spatial_test], axis=0
    )
    return inputs_spatial, labels_spatial


def parse_args():
    """Parse cmd args."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_file", type=str, required=True)
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_file = args.dataset_file
    # Generate data
    X, Y = generate_burgers_dataset(
        n_samples=1200, n_grid_points=1024, nu=0.1, t1=1.0, batch_size=100
    )
    X_np, Y_np = np.array(X), np.array(Y)

    np.savez(out_file, X=X_np, Y=Y_np)
