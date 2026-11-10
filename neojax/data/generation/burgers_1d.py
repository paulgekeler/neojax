# Copyright 2026 Paul Gekeler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Burgers 1D Dataset Generation."""

import warnings

import exponax as ex
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float

from neojax.data.datasets.bundle_dataset import BundleDataset
from neojax.data.datasets.raw_dataset import RawDataset


def generate_fno_initial_conditions(
    n_samples: int, n_grid_points: int = 8192, seed: int = 42
) -> Float[Array, "{n_samples} 1 {n_grid_points}"]:
    r"""Generates periodic GRF initial conditions on the unit interval.

    Samples from $\mathcal{N}(0, 625(-\Delta + 25I)^{-2})$ by filtering
    complex white noise in Fourier space.

    Args:
        n_samples: Number of samples.
        n_grid_points: Number of spatial grid points. Default is 8192.
        seed: Jax random seed. Default is 42.

    Returns:
        Initial conditions in the spatial domain with a singleton channel axis.
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
    return jnp.fft.irfft(u0_ft, n=n_grid_points, axis=-1)[:, None, :]


def generate_burgers_1d(
    n_samples: int = 1000,
    n_grid_points: int = 1024,
    nu: float = 0.1,
    t1: float = 1.0,
    n_steps: int = 100,
    batch_size: int = 1000,
    seed: int = 42,
    use_bundle: bool = False,
) -> RawDataset | BundleDataset:
    r"""Generates 1D Burgers equation dataset.

    Solves the viscous Burgers equation on the periodic unit interval with the
    exponential time differencing Runge-Kutta (ETDRK) pseudo-spectral solver from
    [exponax](https://github.com/Ceyron/exponax). The stiff diffusion term is
    integrated exactly, so a few fixed steps already reach a relative accuracy
    of about `1e-5`.

    Generation is batched to limit memory usage.

    Args:
        n_samples: Number of samples. Default is 1000.
        n_grid_points: Number of spatial grid points. Default is 1024.
        nu: Viscosity. Default is 0.1.
        t1: Final time. Default is 1.0.
        n_steps: Number of fixed ETDRK steps from `t=0` to `t=t1`. Default is 100.
        batch_size: Number of trajectories solved at once. Default is 1000.
        seed: Jax random seed. Default is 42.
        use_bundle: Whether to return a `BundleDataset` instead of
            `RawDataset`. Default is `False.`

    Returns:
        An instance of either `RawDataset` or `BundleDataset`.
        If `RawDataset`, holds `inputs` (the initial condition) and `labels`
        (the solution at `t1`), each of shape `(n_samples, 1, n_grid_points)`.
        If `BundleDataset`, holds both time steps as `fields` of shape
        `(n_samples, 2, 1, n_grid_points)` and the grid as `coords` of shape
        `(1, n_grid_points)`.

    ??? cite
        [APEBench: A Benchmark for Autoregressive
        Neural Emulators of PDEs](https://arxiv.org/abs/2411.00180)

        ```bibtex
        @article{koehler2024apebench,
            title={{APEBench}: A Benchmark for Autoregressive
            Neural Emulators of {PDE}s},
            author={Koehler, Felix and Niedermayr, Simon
            and Westermann, R{\"u}diger and Thuerey, Nils},
            journal={Advances in Neural Information Processing Systems},
            volume={38},
            year={2024}
        }
        ```
    """
    if n_grid_points > 8192:
        warnings.warn(
            f"Resolution {n_grid_points} is high! Generation might take longer.",
            stacklevel=2,
        )
    u0 = generate_fno_initial_conditions(
        n_samples=n_samples, n_grid_points=n_grid_points, seed=seed
    )

    stepper = ex.stepper.Burgers(
        num_spatial_dims=1,
        domain_extent=1.0,
        num_points=n_grid_points,
        dt=t1 / n_steps,
        diffusivity=nu,
    )
    solve = jax.jit(jax.vmap(ex.repeat(stepper, n_steps)))
    u1 = jnp.concatenate(
        [solve(u0[i : i + batch_size]) for i in range(0, n_samples, batch_size)]
    )

    if use_bundle:
        coords = jnp.linspace(0.0, 1.0, n_grid_points, endpoint=False)[None, :]
        return BundleDataset(coords=coords, fields=jnp.stack([u0, u1], axis=1))
    return RawDataset(inputs=u0, labels=u1)
