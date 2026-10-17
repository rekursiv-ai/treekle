"""The fallback wheel must remain installable without a native extension."""

from __future__ import annotations

from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Final, cast
from zipfile import ZipFile

import os
import subprocess
import sys
import tomllib

import pytest


_CWD: Final = Path(__file__).resolve().parent

ROOT = _CWD.parent

# The wheel builds from the public repository's ``pyproject.toml``, which the
# monorepo keeps in ``.export`` and does not build from.
pytestmark = pytest.mark.skipif(
    not (ROOT / "pyproject.toml").is_file(),
    reason="needs the public repository layout",
)


@pytest.fixture(scope="module")
def pure_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("dist")
    result = subprocess.run(  # noqa: S603 -- Runs this repository's own build script.
        [sys.executable, str(ROOT / "scripts/build_pure_wheel.py"), str(output)],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def test_pure_wheel_uses_project_metadata_but_no_native_files(
    pure_wheel: Path,
) -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    version = cast("str", config["project"]["version"])
    metadata_version = cast(
        "str",
        config["tool"]["hatch"]["build"]["targets"]["wheel"]["core-metadata-version"],
    )
    assert pure_wheel.name == f"treekle-{version}-py3-none-any.whl"
    with ZipFile(pure_wheel) as archive:
        names = archive.namelist()
        assert "treekle/serialize.py" in names
        assert "treekle/deserialize.py" in names
        assert "treekle/_rust.pyi" in names
        assert "treekle/py.typed" in names
        assert not any(
            name.endswith((".so", ".pyd", "_test.py", "conftest.py")) for name in names
        )
        metadata = archive.read(f"treekle-{version}.dist-info/METADATA").decode()
        assert metadata.startswith(f"Metadata-Version: {metadata_version}\n")
        assert f"\nVersion: {version}\n" in metadata
        wheel_info = archive.read(f"treekle-{version}.dist-info/WHEEL")
        assert b"Root-Is-Purelib: true" in wheel_info
        assert b"Tag: py3-none-any" in wheel_info


def test_the_release_builds_with_the_hatchling_the_tests_use() -> None:
    script = (ROOT / "scripts/build_pure_wheel.py").read_text().splitlines()
    block = script[script.index("# /// script") + 1 : script.index("# ///")]
    metadata = tomllib.loads("\n".join(line.removeprefix("# ") for line in block))
    assert metadata["dependencies"] == [
        f"hatchling=={installed_version('hatchling')}",
    ]


def _check_wheel(mode: str, installed: Path) -> subprocess.CompletedProcess[str]:
    """Run ``check_wheel.py`` against the treekle found in ``installed``."""
    env = {
        key: value for key, value in os.environ.items() if key != "TREEKLE_PURE_PYTHON"
    }
    env["PYTHONPATH"] = str(installed)
    return subprocess.run(  # noqa: S603 -- Runs this repository's own check script.
        [
            sys.executable,
            "-P",
            "-W",
            "error",
            str(ROOT / "scripts/check_wheel.py"),
            mode,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("mode", "code", "says"),
    [("python", 0, "python: "), ("native", 1, "treekle._rust is missing")],
)
def test_check_wheel_accepts_only_the_implementation_a_wheel_has(
    pure_wheel: Path,
    tmp_path: Path,
    mode: str,
    code: int,
    says: str,
) -> None:
    with ZipFile(pure_wheel) as archive:
        archive.extractall(tmp_path)
    result = _check_wheel(mode, tmp_path)
    assert result.returncode == code, result.stderr
    assert says in result.stdout + result.stderr


@pytest.mark.skipif(
    not any(
        path.suffix in {".so", ".pyd"} for path in (ROOT / "treekle").glob("_rust.*")
    ),
    reason="the Rust core is not built in place",
)
def test_check_wheel_accepts_a_built_core() -> None:
    result = _check_wheel("native", ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("native: ")


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
