"""``check_wheel`` accepts only the implementation the installed wheel should run."""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from types import ModuleType

import sys

import pytest

from scripts import check_wheel
from treekle import deserialize, serialize


_CORE = "treekle._rust"


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["check_wheel.py", *argv])
    return check_wheel.main()


def _without_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, _CORE, None)
    monkeypatch.setattr(serialize, "to_plain", serialize._python_to_plain)
    monkeypatch.setattr(deserialize, "from_plain", deserialize._python_from_plain)


def _with_core(monkeypatch: pytest.MonkeyPatch) -> None:
    core = ModuleType(_CORE)
    core.__spec__ = ModuleSpec(_CORE, None, origin="/wheel/_rust.so")
    vars(core).update(
        to_plain=serialize._python_to_plain,
        from_plain=deserialize._python_from_plain,
    )
    monkeypatch.setitem(sys.modules, _CORE, core)
    monkeypatch.setattr(serialize, "to_plain", serialize._python_to_plain)
    monkeypatch.setattr(deserialize, "from_plain", deserialize._python_from_plain)


@pytest.mark.parametrize("argv", [(), ("both",), ("native", "python")])
def test_rejects_a_bad_mode(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="usage"):
        _run(monkeypatch, *argv)


def test_accepts_the_python_walkers_without_a_core(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _without_core(monkeypatch)
    assert _run(monkeypatch, "python") == 0
    assert capsys.readouterr().out == f"python: {serialize.__file__}\n"


def test_accepts_the_core_walkers_with_a_core(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _with_core(monkeypatch)
    assert _run(monkeypatch, "native") == 0
    assert capsys.readouterr().out == "native: /wheel/_rust.so\n"


def test_rejects_a_native_wheel_without_a_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_core(monkeypatch)
    with pytest.raises(SystemExit, match="_rust is missing"):
        _run(monkeypatch, "native")


def test_rejects_a_python_wheel_with_a_core(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_core(monkeypatch)
    with pytest.raises(SystemExit, match="_rust is present"):
        _run(monkeypatch, "python")


def test_rejects_python_walkers_that_were_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_core(monkeypatch)
    monkeypatch.setattr(serialize, "to_plain", repr)
    with pytest.raises(SystemExit, match="does not run its python implementation"):
        _run(monkeypatch, "python")


def test_rejects_walkers_that_do_not_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_core(monkeypatch)
    monkeypatch.setattr(deserialize, "_python_from_plain", _plain_only)
    monkeypatch.setattr(deserialize, "from_plain", _plain_only)
    with pytest.raises(SystemExit, match="does not round-trip"):
        _run(monkeypatch, "python")


def _plain_only(data: object, target: object) -> object:
    """Return ``data`` unread, as a walker that ignores ``target`` would."""
    del target
    return data


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
