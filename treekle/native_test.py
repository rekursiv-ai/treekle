from __future__ import annotations

from types import ModuleType

import sys

import pytest

from treekle import native


def _python() -> str:
    return "python"


def _core(*, setup_raises: bool, has_walker: bool) -> ModuleType:
    core = ModuleType("treekle._rust")

    def setup(module: dict[str, object]) -> None:
        module["seen"] = True
        if setup_raises:
            raise AttributeError("treekle.serialize has no OLD_HELPER")

    core.__dict__["setup_encode"] = setup
    if has_walker:
        core.__dict__["to_plain"] = _native_walker
    return core


def _native_walker() -> str:
    return "native"


def _choose(module: dict[str, object]) -> object:
    return native.native(_python, name="to_plain", setup="setup_encode", module=module)


@pytest.mark.parametrize("setting", ["1", "true", "yes"])
def test_the_environment_variable_keeps_python(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    monkeypatch.setenv("TREEKLE_PURE_PYTHON", setting)
    monkeypatch.setitem(
        sys.modules,
        "treekle._rust",
        _core(setup_raises=False, has_walker=True),
    )
    module: dict[str, object] = {"__package__": "treekle"}
    assert _choose(module) is _python
    assert "seen" not in module


def test_a_missing_core_keeps_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREEKLE_PURE_PYTHON", "")
    monkeypatch.setitem(sys.modules, "treekle._rust", None)
    assert _choose({"__package__": "treekle"}) is _python


@pytest.mark.parametrize("setting", ["", "0"])
def test_a_matching_core_replaces_python_once_set_up(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
) -> None:
    monkeypatch.setenv("TREEKLE_PURE_PYTHON", setting)
    monkeypatch.setitem(
        sys.modules,
        "treekle._rust",
        _core(setup_raises=False, has_walker=True),
    )
    module: dict[str, object] = {"__package__": "treekle"}
    assert _choose(module) is _native_walker
    assert module["seen"] is True


@pytest.mark.parametrize(
    ("setup_raises", "has_walker", "says"),
    [(True, True, "OLD_HELPER"), (False, False, "to_plain")],
)
def test_a_stale_core_keeps_python_and_warns_at_the_importer(
    monkeypatch: pytest.MonkeyPatch,
    setup_raises: bool,
    has_walker: bool,
    says: str,
) -> None:
    monkeypatch.setenv("TREEKLE_PURE_PYTHON", "")
    monkeypatch.setitem(
        sys.modules,
        "treekle._rust",
        _core(setup_raises=setup_raises, has_walker=has_walker),
    )
    with pytest.warns(RuntimeWarning, match=says) as caught:
        chosen = _choose({"__package__": "treekle"})
    assert chosen is _python
    assert "treekle's Rust core does not match" in str(caught[0].message)


def test_a_core_missing_its_walker_is_not_set_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TREEKLE_PURE_PYTHON", "")
    monkeypatch.setitem(
        sys.modules,
        "treekle._rust",
        _core(setup_raises=False, has_walker=False),
    )
    module: dict[str, object] = {"__package__": "treekle"}
    with pytest.warns(RuntimeWarning):
        _choose(module)
    assert "seen" not in module


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
