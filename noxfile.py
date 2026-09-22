# ruff: noqa: D100, D103
# noxfile.py
import nox


@nox.session(python=["3.11", "3.12", "3.13"])
def tests(session):
    session.run_install(
        "uv",
        "sync",
        "--all-extras",
        "--dev",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )
    session.run("pytest", "neojax/tests/")
