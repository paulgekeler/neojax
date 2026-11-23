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

# ruff: noqa: D100, D103
# noxfile.py
import os
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"

JAX_CACHE_DIR = Path(".nox/jax_cache").resolve()


@nox.session(
    python=["3.11", "3.12", "3.13", "3.14"],
    allow_parallel=True,
)
def tests(session):
    session.run_install(
        "uv",
        "sync",
        "--locked",
        "--all-extras",
        "--dev",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )

    cache_dir = Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", JAX_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)

    session.run(
        "pytest",
        "neojax/tests/",
        *session.posargs,
        env={
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        },
    )
