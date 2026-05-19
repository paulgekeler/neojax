# Contributing to neojax

Thank you for your interest in contributing! We welcome any type of contribution. This document outlines the process for setting up your development environment and submitting changes.

## Development Setup

We use `pyproject.toml` to manage dependencies. To set up the project in development mode and install the necessary development dependencies, run the following command from the root of the project:

```bash
pip install -e ".[dev]"
```

*Note: Depending on your specific environment, you may need to use a virtual environment (like `venv` or `conda`) before running the install command.*

## Running Tests

The test suite is built with `pytest` and all tests are located in the `neojax/tests/` directory. To run the tests, execute:

```bash
pytest neojax/tests/
```

Make sure all tests pass before submitting your changes. It is highly recommended to write new tests for any new features or bug fixes.

## Git Commit Guidelines

To maintain a clean and understandable project history, please follow these guidelines when making commits:

*   **One logical change per commit:** Keep your commits focused. Do not mix unrelated changes (e.g., refactoring code and adding a new feature) in a single commit.
*   **Write clear commit messages:** 
    *   Use the imperative mood in the subject line (e.g., "Add feature" not "Added feature" or "Adds feature").
    *   Keep the subject line short (under 50 characters if possible).
    *   Provide context or explain the *why* in the commit body if the change is complex.

## Pull Request Process

When you are ready to submit your changes, please open a Pull Request (PR):

1.  **Fork the repository** and create a new branch for your feature or bugfix.
2.  **Ensure tests pass.** Run the test suite locally.
3.  **Keep PRs focused:** Similar to commits, a PR should address a single issue or add a single feature.
4.  **Provide a clear description:** Explain what the PR does, why it is needed, and any specific areas reviewers should focus on.
5.  **Respond to feedback:** Be prepared to make modifications based on the code review. 

We look forward to your contributions!