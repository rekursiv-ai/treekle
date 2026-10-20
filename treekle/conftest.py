"""Run the walker tests against both implementations of ``to_plain``/``from_plain``.

The Rust core replaces the Python walkers at import, so without this a build with
the core never runs the Python walkers that serve every machine without one.
Test modules opt in with ``pytestmark = pytest.mark.usefixtures("implementation")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import sys

import pytest

from treekle import deserialize, serialize, wire


if TYPE_CHECKING:
    from types import ModuleType


_NATIVE: Final = {"to_plain": serialize.to_plain, "from_plain": deserialize.from_plain}
_PYTHON: Final = {
    "to_plain": serialize._python_to_plain,  # noqa: SLF001 -- The walker under test.
    "from_plain": deserialize._python_from_plain,  # noqa: SLF001 -- The walker under test.
}


@pytest.fixture(
    params=[
        pytest.param(
            "native",
            marks=pytest.mark.skipif(
                _NATIVE == _PYTHON,
                reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
            ),
        ),
        "python",
    ],
)
def implementation(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Bind the walkers and ``wire``'s JSON functions to the implementation under test."""
    param = cast("str", request.param)
    if param == "native":
        return param
    # ``parse``, ``wire``, and the test modules bound the names at import.
    modules = [
        module
        for name, module in sys.modules.items()
        if name == "treekle" or name.startswith("treekle.")
    ]
    for module in [*modules, cast("ModuleType", request.module)]:
        for name, function in _NATIVE.items():
            if getattr(module, name, None) is function:
                monkeypatch.setattr(module, name, _PYTHON[name])
    # ``wire`` also holds the core's JSON functions; without them it reads and
    # writes through ``json`` and the walkers above.
    for name in ("_loads", "_dumps", "_typed_loads"):
        monkeypatch.setattr(wire, name, None)
    return param
