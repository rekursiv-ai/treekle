"""Both treekle wheels, built and run outside the checkout."""

from __future__ import annotations

from pathlib import Path

import subprocess

import pytest

from treekle.lib.testing.wheel import Wheel, built_wheel


pytestmark = pytest.mark.cli_uv


def test_the_native_wheel_runs_the_rust_core() -> None:
    wheel = built_wheel("treekle", anchor=__file__)

    assert wheel.run(_check("native", wheel.project)).startswith("native: ")


def test_the_pure_wheel_runs_the_python_walkers(tmp_path: Path) -> None:
    """The universal wheel serves every platform the native ones do not cover."""
    project = built_wheel("treekle", anchor=__file__).project
    built = subprocess.run(  # noqa: S603 -- Fixed uv argv over the project's own script.
        ["uv", "run", str(project / "scripts" / "build_pure_wheel.py"), str(tmp_path)],  # noqa: S607 -- uv from PATH, as every gate runs it.
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    wheel = Wheel(path=Path(built.strip()), project=project)

    assert wheel.run(_check("python", project)).startswith("python: ")


def _check(implementation: str, project: Path) -> str:
    """Return code running ``scripts/check_wheel.py`` with warnings as errors."""
    script = project / "scripts" / "check_wheel.py"
    # A warning is the Rust core's fallback notice, so it must fail the run.
    return (
        "import runpy, sys, warnings; warnings.simplefilter('error'); "
        f"sys.argv = ['check_wheel.py', {implementation!r}]; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
