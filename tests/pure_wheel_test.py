"""The release build uses the same hatchling version as validation."""

from __future__ import annotations

from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Final

import tomllib


_CWD: Final[Path] = Path(__file__).resolve().parent


def test_the_release_builds_with_the_hatchling_the_tests_use() -> None:
    root = _CWD.parent
    script = (root / "scripts/build_pure_wheel.py").read_text().splitlines()
    block = script[script.index("# /// script") + 1 : script.index("# ///")]
    metadata = tomllib.loads("\n".join(line.removeprefix("# ") for line in block))
    assert metadata["dependencies"] == [
        f"hatchling=={installed_version('hatchling')}",
    ]


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
