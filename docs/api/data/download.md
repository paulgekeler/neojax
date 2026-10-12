# Dataset Downloading API

*The Downloading API was co-authored with [Gemini](https://gemini.google.com)*

The `neojax.data` module provides robust dataset downloading utilities to fetch standardized Partial Differential Equation (PDE) datasets from various public hosting services (like Hugging Face Hub, Zenodo, DaRUS Stuttgart Dataverse, or direct HTTP/HTTPS URLs).

---

## Downloader Architecture

Scientific PDE datasets are typically large (ranging from hundreds of megabytes to tens of gigabytes). To download these files reliably over standard internet connections, `neojax` implements a resilient download pipeline:

* **Connection Retries**: Automatically retries downloads on network timeouts or transient errors using exponential backoff.
* **Range-Based Resume**: Checks the size of existing partial downloads (marked with a `.part` extension) and sends HTTP `Range` requests to resume downloading from the offset where the connection was cut, preventing wasted bandwidth.
* **Progress Tracking**: Uses `tqdm` for interactive progress bars when available, falling back to a clean terminal-printing progress bar otherwise.
* **Checksum Verification**: Verifies finished files against expected MD5 or SHA256 hashes, deleting and re-downloading if a mismatch is found.

---

## Registered Datasets

The registry contains **37 standardized PDE datasets** (including 20 **PDEgym** datasets from Hugging Face, 2 verification datasets from **Zenodo**, and 15 complete **PDEBench** datasets from Stuttgart's DaRUS repository).

??? cite "PDEgym"

    [Poseidon: Efficient Foundation Models for PDEs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/84e1b1ec17bb11c57234e96433022a9a-Abstract-Conference.html)

    [Huggingface collection](https://huggingface.co/collections/camlab-ethz/pdegym)
    
    ```bibtex
    @inproceedings{herde2024_poseidon,
        author = {Herde, Maximilian and Raoni\'{c}, Bogdan and Rohner, Tobias and K\"{a}ppeli, Roger and Molinaro, Roberto and de B\'{e}zenac, Emmanuel and Mishra, Siddhartha},
        booktitle = {Advances in Neural Information Processing Systems},
        doi = {10.52202/079017-2311},
        editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
        pages = {72525--72624},
        publisher = {Curran Associates, Inc.},
        title = {Poseidon: Efficient Foundation Models for PDEs},
        url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/84e1b1ec17bb11c57234e96433022a9a-Paper-Conference.pdf},
        volume = {37},
        year = {2024}
    }
    ```

??? cite "PDEBench"

    [PDEBENCH: An Extensive Benchmark for Scientific
    Machine Learning](https://papers.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf)

    ```bibtex
    @article{takamoto2022pdebench,
        title={Pdebench: An extensive benchmark for scientific machine learning},
        author={Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and MacKinlay, Daniel and Alesiani, Francesco and Pfl{\"u}ger, Dirk and Niepert, Mathias},
        journal={Advances in neural information processing systems},
        volume={35},
        pages={1596--1611},
        year={2022}
    }
    ```

### Hugging Face (`camlab-ethz/pdegym`)
The PDEgym collection represents a diverse range of physical systems used for pretraining and benchmarking operator foundation models:

| Key | Description |
| :--- | :--- |
| `ace` | Allen-Cahn equation |
| `ce_crp` | Compressible Euler equations (Centrally Rarefaction/Pressure problem) |
| `ce_gauss` | Compressible Euler equations with Gaussian initial conditions |
| `ce_kh` | Compressible Euler equations (Kelvin-Helmholtz instability) |
| `ce_rm` | Compressible Euler equations (Richtmyer-Meshkov problem) |
| `ce_rp` | Compressible Euler equations (Riemann problem) |
| `ce_rpui` | Compressible Euler equations (Riemann problem variant) |
| `fns_kf` | Forced incompressible Navier-Stokes equations with Kolmogorov forcing |
| `gce_rt` | Gravity-driven compressible Euler (Rayleigh-Taylor instability) |
| `helmholtz` | Solutions to the Helmholtz equation |
| `ns_bb` | Navier-Stokes equations (Backwards-facing step variant) |
| `ns_gauss` | Navier-Stokes equations with Gaussian initial conditions |
| `ns_pwc` | Navier-Stokes equations (Piecewise constant) |
| `ns_sines` | Navier-Stokes equations with sine-based initial conditions |
| `ns_sl` | Navier-Stokes equations (Shear Layer) |
| `ns_svs` | Navier-Stokes equations (Small-scale vortex structures) |
| `poisson_gauss` | Poisson equation with Gaussian sources |
| `se_af` | Compressible Euler equations (Steady Airfoil) |
| `wave_gauss` | Wave equation with Gaussian initial conditions and wave speeds |
| `wave_layer` | Wave equation with layered wave speeds |

### Zenodo (PDEBench Verification)
Quick reference datasets:

| Key | Description |
| :--- | :--- |
| `burgers_1d` | 1D Burgers' equation Salsa T10 verification dataset |
| `darcy_flow_2d` | 2D Darcy Flow beta 1.0 verification dataset |

### DaRUS Stuttgart Dataverse (PDEBench)
The complete set of PDEBench problems resolved dynamically using Dataverse file ID queries:

| Key | Description |
| :--- | :--- |
| `pdebench_advection_1d` | 1D Advection dataset (8 files) |
| `pdebench_burgers_1d` | 1D Burgers' equation dataset (12 files) |
| `pdebench_cfd_1d_random` | 1D Compressible Flow with random initial conditions (4 files) |
| `pdebench_cfd_1d_shock` | 1D Compressible Flow with shock tube boundaries (1 file) |
| `pdebench_diffusion_sorption_1d` | 1D Diffusion-Sorption equation dataset (1 file) |
| `pdebench_cfd_2d_random` | 2D Compressible Flow with random initial conditions (6 files) |
| `pdebench_cfd_2d_turbulent` | 2D Compressible Flow with turbulent dynamics (2 files) |
| `pdebench_darcy_2d` | 2D Darcy Flow dataset with variable betas (5 files) |
| `pdebench_diffusion_reaction_2d` | 2D Diffusion-Reaction equation dataset (1 file) |
| `pdebench_reaction_diffusion_2d` | 2D Reaction-Diffusion FitzHugh-Nagumo dataset (16 files) |
| `pdebench_brusselator_2d` | 2D Reaction-Diffusion Brusselator dataset (1 file) |
| `pdebench_cfd_3d_random` | 3D Compressible Flow with random initial conditions (2 files) |
| `pdebench_cfd_3d_turbulent` | 3D Compressible Flow with turbulent dynamics (1 file) |
| `pdebench_sod_1d` | 1D Sod Shock Tube dataset (1 file) |
| `pdebench_incompressible_2d` | 2D Incompressible Inhomogeneous Navier-Stokes dataset (275 files) |

---

## API Reference

::: neojax.data.download.dataset.download_dataset

::: neojax.data.download.http_downloader.HTTPDownloader

::: neojax.data.download.huggingface_downloader.HuggingFaceDownloader

::: neojax.data.download.zenodo_downloader.ZenodoDownloader

::: neojax.data.download.dataverse_downloader.DataverseDownloader

