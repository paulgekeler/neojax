# Contributing to neojax

Thank you for your interest in contributing! We welcome any type of contribution. This document outlines the process for setting up your development environment and submitting changes.

## Development Setup

We use `pyproject.toml` to manage dependencies. The usage of [uv](https://github.com/astral-sh/uv) is desired (as it is faster and ensures lockfile consistency) but not required.

### Option A: Using `uv` (Desired)

If you have `uv` installed, you can set up the environment and install all dependencies (including `dev` and `docs` extras) by running:

```bash
# Sync the workspace and install all extras
uv sync --all-extras
```

Alternatively, if you want to install them in an existing virtual environment using `uv pip`:

```bash
uv pip install -e ".[dev,docs]"
```

### Option B: Using standard `pip`

If you prefer using standard Python tools:

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package in editable mode along with `dev` and `docs` dependencies:
   ```bash
   pip install -e ".[dev,docs]"
   ```

## Type Annotations & Runtime Type Checking

To ensure code correctness and documentation clarity, we enforce type annotations throughout the codebase:

*   **`jaxtyping` + `beartype`:** All functions, classes, methods, and variables must be fully type-annotated using `jaxtyping` (for JAX arrays, shapes, and types) and standard Python typing.
*   **Runtime Verification:** During testing, we use `beartype` to check type annotations at runtime. This is set up via an import hook in [conftest.py](neojax/tests/conftest.py). Unannotated or incorrectly annotated code will trigger runtime failures during test runs.

Keep the annotations concise and to the point, preferably `"b"` for batch, `"c"` for channels, etc. Use `...` for unspecified dimensions.

Example annotation using `jaxtyping`:
```python
from jaxtyping import Array, Float, PRNGKeyArray

def compute_loss(
    x: Float[Array, "b in_d"],
    target: Float[Array, "b out_d"]
) -> Float[Array, ""]:
    ...
```

## Running Tests

The test suite is built with `pytest` and all tests are located in the `neojax/tests/` directory.

If you are using `uv`:
```bash
uv run pytest neojax/tests/
```

If you are using standard tools (with your virtual environment activated):
```bash
pytest neojax/tests/
```

Make sure all tests pass before submitting your changes. It is highly recommended to write new tests for any new features or bug fixes.

## Git Commit Guidelines

To maintain a clean and understandable project history, please follow these guidelines when making commits:

*   **One logical change per commit:** Keep your commits focused. Do not mix unrelated changes (e.g., refactoring code and adding a new feature) in a single commit.
*   **Commit Message Prefixes:** Each commit message must be prefixed to denote the type and scope of the change:
    *   `feat(<component>):` For new features (e.g., `feat(losses): add SobolevLoss`).
    *   `fix(<component>):` For bug fixes (e.g., `fix(operators): correct shape mismatch`).
    *   `doc:` For documentation-only changes (e.g., `doc: update contributing guide`).
    *   `chore:` For version bumps, directory management, `.gitignore` updates, and dependency maintenance.
*   **Write clear commit messages:** 
    *   Use the imperative mood in the subject line (e.g., `feat(losses): add feature` not `added feature` or `adds feature`).
    *   Keep the subject line short (under 50 characters if possible).
    *   Provide context or explain the *why* in the commit body if the change is complex.

## Pull Request Process

When you are ready to submit your changes, please open a Pull Request (PR):

1.  **Fork the repository** and create a new branch for your feature or bugfix.
2.  **Ensure tests pass.** Run the test suite locally using the test commands above.
3.  **Keep PRs focused:** Similar to commits, a PR should address a single issue or add a single feature.
4.  **Provide a clear description:** Explain what the PR does, why it is needed, and any specific areas reviewers should focus on.
5.  **Respond to feedback:** Be prepared to make modifications based on the code review. 

We look forward to your contributions!