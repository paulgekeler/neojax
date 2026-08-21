"""Implementation of Burgers 1D Dataset Generation."""

import warnings

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.datasets.raw_dataset import RawDataset


def burgers_explicit_vf(
    t: Float[Array, "n_tsteps"],
    u: Float[Array, "n_tsteps"],
    args: tuple[float, Float[Array, "n_tsteps"], Float[Array, "n_tsteps"]],
) -> Float[Array, "n_tsteps"]:
    """Computes the non-linear advection term in Fourier space.

    Args:
        t: Time steps. Placeholder for compatibility.
        u: Function values.
        args: Additional vector field arguments (nu, first spectral derivative, second spectral derivative).

    Returns:
        Non-linear advection term in Fourier space.
    """
    nu, ik, ik2 = args
    # Move to spatial domain to handle the multiplication
    u_spatial = jnp.fft.irfft(u)
    # Compute du/dx in Fourier space, then move to spatial domain
    dudx_spatial = jnp.fft.irfft(ik * u)
    # Compute the non-linear advection term: -u * (du/dx)
    advection_spatial = -u_spatial * dudx_spatial
    # Move the advection term back to Fourier space
    advection_ft = jnp.fft.rfft(advection_spatial)
    return advection_ft


def burgers_implicit_vf(
    t: Float[Array, "n_tsteps"],
    u: Float[Array, "n_tsteps"],
    args: tuple[float, Float[Array, "n_tsteps"], Float[Array, "n_tsteps"]],
) -> Float[Array, "n_tsteps"]:
    """Computes the linear diffusion term exactly in Fourier space.

    Args:
        t: Time steps. Placeholder for compatibility.
        u: Function values.
        args: Additional vector field arguments (nu, first spectral derivative, second spectral derivative).

    Returns:
        Linear diffusion term in Fourier space.
    """
    nu, ik, ik2 = args
    # Compute the linear diffusion term exactly in Fourier space: ν * (d²u/dx²)
    # Mathematically: ν * (ik)² * û
    diffusion_ft = nu * ik2 * u
    return diffusion_ft


def solve_single_trajectory(
    u0_ft: Float[Array, "n_points"],
    nu: float,
    ik: Float[Array, "n_tsteps"],
    ik2: Float[Array, "n_tsteps"],
    t1: float,
) -> Float[Array, "2 n_points"]:
    """Solves Burgers equation for a single initial condition.

    Args:
        u0_ft: Initial condition.
        nu: Viscosity.
        ik: Spectral representation of first derivative.
        ik2: Spectral representation of second derivative.
        t1: Maximum time.

    Returns:
        Solution trajectory at t=0 and t=1.
    """
    # KenCarp5 expects MultiTerm of explicit and implicit part
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


def generate_fno_initial_conditions(
    n_samples: int, n_grid_points: int = 8192, seed: int = 42
) -> Float[Array, "1 {n_grid_points}"]:
    """Generates periodic GRF initial conditions.

    Args:
        n_samples: Number of samples.
        n_grid_points: Number of spatial grid points. Default is 8192.
        seed: Jax random seed. Default is 42.

    Returns:
        Fourier transform of initial condition.
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


def generate_burgers_1d(
    n_samples: int = 1000,
    n_grid_points: int = 1024,
    nu: float = 0.1,
    t1: float = 1.0,
    batch_size: int = 100,
    seed: int = 42,
    use_bundle: bool = False,
) -> RawDataset | BundleDataset:
    """Generates 1D Burgers equation dataset.

    Solves equation using a pseudo-spectral split step method as in the reference.

    Generation is batched to limit memory usage.

    Args:
        n_samples: Number of samples. Default is 1000.
        n_grid_points: Number of spatial grid points. Default is 1024.
        nu: Viscosity. Default is 0.1.
        t1: Maximum time. Default is 1.0.
        batch_size: Batch size for generation to save memory. Default is 100.
        seed: Jax random seed. Default is 42.
        use_bundle: Whether to return a `BundleDataset` instead of
            `RawDataset`. Default is `False.`

    Returns:
        An instance of either `RawDataset` or `BundleDataset`.
        If `RawDataset`, assigns attrs `inputs` and `labels`,
        else assigns `fields` of `BundleDataset` as concatenated `inputs`
        and `labels` along first axis (and assigns `corrds`).

    ??? cite
        [Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs](https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf)

        ```bibtex
        @article{kovachki2023neural,
            title={Neural operator: Learning maps between
            function spaces with applications to pdes},
            author={Kovachki, Nikola and Li, Zongyi and Liu,
            Burigede and Azizzadenesheli, Kamyar and Bhattacharya,
            Kaushik and Stuart, Andrew and Anandkumar, Anima},
            journal={Journal of Machine Learning Research},
            volume={24},
            number={89},
            pages={1--97},
            year={2023}
        }
        ```
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

    # vmap and jit solver
    solve_batched_trajectories = jax.vmap(
        solve_single_trajectory, in_axes=(0, None, None, None, None)
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

    coords = jnp.arange(0, t1, n_grid_points)[None, :]

    if use_bundle:
        return BundleDataset(coords=coords, fields=batched_solutions_ft)
    else:
        return RawDataset(inputs=inputs_spatial, labels=labels_spatial)
