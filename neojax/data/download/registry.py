"""Dataset registry for standardized PDE problems."""

from typing import Any

# Registry mapping dataset names to their downloader backend and configurations.
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # Hugging Face PDEgym datasets (camlab-ethz)
    "ace": {
        "description": "Allen-Cahn equation from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/ACE",
        },
    },
    "ce_crp": {
        "description": (
            "Compressible Euler equations (Centrally Rarefaction/Pressure problem) "
            "from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-CRP",
        },
    },
    "ce_gauss": {
        "description": (
            "Compressible Euler equations with Gaussian initial conditions from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-Gauss",
        },
    },
    "ce_kh": {
        "description": (
            "Compressible Euler equations (Kelvin-Helmholtz instability) from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-KH",
        },
    },
    "ce_rm": {
        "description": (
            "Compressible Euler equations (Richtmyer-Meshkov problem) from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-RM",
        },
    },
    "ce_rp": {
        "description": "Compressible Euler equations (Riemann problem) from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-RP",
        },
    },
    "ce_rpui": {
        "description": (
            "Compressible Euler equations (Riemann problem variant) from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/CE-RPUI",
        },
    },
    "fns_kf": {
        "description": (
            "Forced incompressible Navier-Stokes equations with Kolmogorov forcing "
            "from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/FNS-KF",
        },
    },
    "gce_rt": {
        "description": (
            "Gravity-driven compressible Euler (Rayleigh-Taylor instability) "
            "from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/GCE-RT",
        },
    },
    "helmholtz": {
        "description": "Solutions to the Helmholtz equation from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/Helmholtz",
        },
    },
    "ns_bb": {
        "description": (
            "Navier-Stokes equations (Backwards-facing step variant) from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-BB",
        },
    },
    "ns_gauss": {
        "description": (
            "Navier-Stokes equations with Gaussian initial conditions from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-Gauss",
        },
    },
    "ns_pwc": {
        "description": "Navier-Stokes equations (Piecewise constant) from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-PwC",
        },
    },
    "ns_sines": {
        "description": (
            "Navier-Stokes equations with sine-based initial conditions from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-Sines",
        },
    },
    "ns_sl": {
        "description": "Navier-Stokes equations (Shear Layer) from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-SL",
        },
    },
    "ns_svs": {
        "description": (
            "Navier-Stokes equations (Small-scale vortex structures) from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/NS-SVS",
        },
    },
    "poisson_gauss": {
        "description": "Poisson equation with Gaussian sources from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/Poisson-Gauss",
        },
    },
    "se_af": {
        "description": "Compressible Euler equations (Steady Airfoil) from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/SE-AF",
        },
    },
    "wave_gauss": {
        "description": (
            "Wave equation with Gaussian initial conditions and wave speeds from PDEgym"
        ),
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/Wave-Gauss",
        },
    },
    "wave_layer": {
        "description": "Wave equation with layered wave speeds from PDEgym",
        "downloader": "huggingface",
        "params": {
            "repo_id": "camlab-ethz/Wave-Layer",
        },
    },
    # Standard Zenodo / HTTP datasets for verification
    "burgers_1d": {
        "description": "1D Burgers' equation dataset from PDEBench",
        "downloader": "zenodo",
        "params": {
            "record_id": "7710323",
            "filenames": ["1D_Burgers_Salsa_T10.hdf5"],
        },
    },
    "darcy_flow_2d": {
        "description": "2D Darcy Flow dataset from PDEBench",
        "downloader": "zenodo",
        "params": {
            "record_id": "7710923",
            "filenames": ["2D_DarcyFlow_beta1.0_Train.hdf5"],
        },
    },
    # PDEBench datasets from DaRUS (doi:10.18419/darus-2986)
    "pdebench_advection_1d": {
        "description": "1D Advection dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                f"1D_Advection_Sols_beta{b}.hdf5"
                for b in ["0.1", "0.2", "0.4", "0.7", "1.0", "2.0", "4.0", "7.0"]
            ],
        },
    },
    "pdebench_burgers_1d": {
        "description": "1D Burgers' equation dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                f"1D_Burgers_Sols_Nu{nu}.hdf5"
                for nu in [
                    "0.001",
                    "0.002",
                    "0.004",
                    "0.01",
                    "0.02",
                    "0.04",
                    "0.1",
                    "0.2",
                    "0.4",
                    "1.0",
                    "2.0",
                    "4.0",
                ]
            ],
        },
    },
    "pdebench_cfd_1d_random": {
        "description": (
            "1D Compressible Flow (random initial conditions) dataset from "
            "PDEBench (DaRUS)"
        ),
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5",
                "1D_CFD_Rand_Eta0.1_Zeta0.1_periodic_Train.hdf5",
                "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_periodic_Train.hdf5",
                "1D_CFD_Rand_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5",
            ],
        },
    },
    "pdebench_cfd_1d_shock": {
        "description": "1D Compressible Flow (shock tube) dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5",
            ],
        },
    },
    "pdebench_diffusion_sorption_1d": {
        "description": "1D Diffusion-Sorption equation dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "1D_diff-sorp_NA_NA.h5",
            ],
        },
    },
    "pdebench_cfd_2d_random": {
        "description": (
            "2D Compressible Flow (random initial conditions) dataset from "
            "PDEBench (DaRUS)"
        ),
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
                "2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5",
                "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
                "2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
                "2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5",
                "2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
            ],
        },
    },
    "pdebench_cfd_2d_turbulent": {
        "description": "2D Compressible Flow (turbulent) dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "2D_CFD_Turb_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
                "2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5",
            ],
        },
    },
    "pdebench_darcy_2d": {
        "description": "2D Darcy Flow dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                f"2D_DarcyFlow_beta{b}_Train.hdf5"
                for b in ["0.01", "0.1", "1.0", "10.0", "100.0"]
            ],
        },
    },
    "pdebench_diffusion_reaction_2d": {
        "description": "2D Diffusion-Reaction equation dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "2D_diff-react_NA_NA.h5",
            ],
        },
    },
    "pdebench_reaction_diffusion_1d": {
        "description": "1D Reaction-Diffusion (FitzHugh-Nagumo) dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                f"ReacDiff_Nu{nu}_Rho{rho}.hdf5"
                for nu in ["0.5", "1.0", "2.0", "5.0"]
                for rho in ["1.0", "2.0", "5.0", "10.0"]
            ],
        },
    },
    "pdebench_brusselator_2d": {
        "description": "2D Reaction-Diffusion Brusselator dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "2D_rdb_NA_NA.h5",
            ],
        },
    },
    "pdebench_cfd_3d_random": {
        "description": "3D Compressible Flow (random initial conditions) dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
                "3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
            ],
        },
    },
    "pdebench_cfd_3d_turbulent": {
        "description": "3D Compressible Flow (turbulent) dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
            ],
        },
    },
    "pdebench_sod_1d": {
        "description": "1D Sod Shock Tube dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [
                "Sod6.hdf5",
            ],
        },
    },
    "pdebench_incompressible_2d": {
        "description": "2D Incompressible Inhomogeneous Navier-Stokes dataset from PDEBench (DaRUS)",
        "downloader": "dataverse",
        "params": {
            "doi": "10.18419/darus-2986",
            "filenames": [f"ns_incom_inhom_2d_512-{i}.h5" for i in range(275)],
        },
    },
}

__all__ = [
    "DATASET_REGISTRY",
]
