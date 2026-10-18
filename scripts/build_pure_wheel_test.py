"""``build_pure_wheel`` builds from the project root into the named directory."""

from __future__ import annotations

from pathlib import Path

import sys

import pytest

from scripts import build_pure_wheel


def test_builds_from_the_root_and_prints_the_wheel_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built: list[tuple[str, Path]] = []

    def build_wheel(directory: str) -> str:
        built.append((directory, Path.cwd()))
        return "treekle-0-py3-none-any.whl"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_pure_wheel, "build_wheel", build_wheel)
    monkeypatch.setattr(sys, "argv", ["build_pure_wheel.py", "out/dist"])

    assert build_pure_wheel.main() == 0
    output = tmp_path / "out" / "dist"
    assert output.is_dir()
    assert built == [(str(output), build_pure_wheel.ROOT)]
    assert Path.cwd() == build_pure_wheel.ROOT
    expected = output / "treekle-0-py3-none-any.whl"
    assert capsys.readouterr().out == f"{expected}\n"


@pytest.mark.parametrize("argv", [(), ("a", "b")])
def test_rejects_anything_but_one_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    monkeypatch.setattr(sys, "argv", ["build_pure_wheel.py", *argv])
    with pytest.raises(SystemExit) as caught:
        build_pure_wheel.main()
    assert caught.value.code == "usage: build_pure_wheel.py OUTPUT_DIRECTORY"


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
