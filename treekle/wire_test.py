"""Tests for ``treekle.wire``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json
import math
import os
import subprocess
import sys
import textwrap
import types

import pytest

from treekle import json_dumps, json_loads, to_plain, wire
from treekle.deserialize import _python_from_plain
from treekle.serialize import _python_to_plain
from treekle.types import Hooks, ReadError


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle.types import Dialect


pytestmark = pytest.mark.usefixtures("implementation")


@dataclass(kw_only=True, slots=True)
class AttributeOrder:
    z: int = 1
    a: int = 2


class TestJsonLoads:
    def test_text_reads_as_a_mutable_tree(self) -> None:
        assert json_loads('{"a": [1, 2.5, null]}') == {"a": [1, 2.5, None]}
        assert json_loads(b"[true]") == [True]

    def test_non_finite_floats_read(self) -> None:
        loaded = json_loads("[NaN, Infinity]")
        assert isinstance(loaded, list)
        assert math.isnan(cast(float, loaded[0]))
        assert loaded[1] == math.inf

    def test_bad_text_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            json_loads("{")

    def test_a_target_reads_the_document_as_that_type(self) -> None:
        assert json_loads('{"x": [1, 2]}', dict[str, tuple[int, int]]) == {
            "x": (1, 2),
        }
        assert json_loads("null", None) is None

    def test_a_target_the_document_does_not_fit_raises(self) -> None:
        with pytest.raises(ReadError, match="cannot read 'a' as int"):
            json_loads('["a"]', list[int])

    def test_options_reach_from_plain(self) -> None:
        with pytest.raises(ReadError, match="cannot read '3' as int"):
            json_loads('"3"', int)
        assert json_loads('"3"', int, strict=False) == 3
        with pytest.raises(ReadError, match="allow_imports"):
            json_loads('{"py/type": "pathlib.Path"}', object)
        assert (
            json_loads(
                '{"py/type": "pathlib.Path"}',
                object,
                allow_imports=True,
            )
            is Path
        )
        hooks: Hooks = {Path: (str, Path)}
        assert json_loads(
            '{"py/hook": ["pathlib.Path", "/a"]}',
            object,
            hooks=hooks,
        ) == Path("/a")


class TestJsonDumps:
    def test_an_object_writes_as_json_dumps_of_its_tree(self) -> None:
        value = {"p": Path("/a"), "n": [1.5, None, "\u00e9"]}
        assert json_dumps(value) == json.dumps(to_plain(value))

    def test_layout_options_reach_json(self) -> None:
        value = {"b": [1, 2], "a": "\u00e9"}
        tree = to_plain(value)
        assert json_dumps(value, indent=2, sort_keys=True) == json.dumps(
            tree,
            indent=2,
            sort_keys=True,
        )
        assert json_dumps(value, ensure_ascii=False, separators=(",", ":")) == (
            '{"b":[1,2],"a":"\u00e9"}'
        )

    def test_options_reach_to_plain(self) -> None:
        value = Path("/a")
        hooks: Hooks = {type(value): (str, Path)}
        hooked = json_dumps(value, hooks=hooks)
        assert hooked == json.dumps(to_plain(value, hooks=hooks))
        assert '"py/hook"' in hooked
        assert json_dumps(math.inf, dialect="python") == "Infinity"
        assert json_dumps(math.inf) == '{"py/float": "inf"}'
        attributes = AttributeOrder()
        assert json_dumps(attributes, sort=False) == json.dumps(
            to_plain(attributes, sort=False),
        )
        assert json_dumps(attributes) != json_dumps(attributes, sort=False)

    def test_an_indent_too_wide_to_allocate_raises_as_json_does(self) -> None:
        # Run apart: a failed native allocation aborts the whole process.
        completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    from treekle import json_dumps

                    for dumps in (json.dumps, json_dumps):
                        try:
                            dumps([1], indent=2**60)
                        except MemoryError:
                            print("MemoryError")
                    """,
                ),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.split() == ["MemoryError", "MemoryError"]

    @pytest.mark.parametrize("dialect", ["yaml"])
    def test_an_unknown_dialect_is_rejected(self, dialect: str) -> None:
        with pytest.raises(ValueError, match="dialect"):
            json_dumps(1, dialect=cast("Dialect", dialect))


def test_json_is_used_as_it_is_at_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # A replacement made after import serves, called once per document, as
    # Python's path names ``json.loads``/``json.dumps`` at the call.
    seen: list[str] = []
    loads = cast("Callable[..., object]", json.loads)
    dumps = cast("Callable[..., str]", json.dumps)

    def counted_loads(*args: object, **options: object) -> object:
        seen.append("loads")
        return loads(*args, **options)

    def counted_dumps(*args: object, **options: object) -> str:
        seen.append("dumps")
        return dumps(*args, **options)

    monkeypatch.setattr(json, "loads", counted_loads)
    monkeypatch.setattr(json, "dumps", counted_dumps)
    assert json_loads("[1]") == [1]
    assert json_loads("[1]", list[int]) == [1]
    assert json_dumps([1]) == "[1]"
    assert seen == ["loads", "loads", "dumps"]


def test_the_cores_json_functions_take_every_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Whether a core is built decides which branch serves; both must forward
    # every option, so this runs on any build.
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def recorded(name: str) -> Callable[..., object]:
        def call(*args: object, **options: object) -> object:
            calls.append((name, args, options))
            return name

        return call

    monkeypatch.setattr(
        wire,
        "_CORE_JSON",
        {
            "loads": recorded("loads"),
            "dumps": recorded("dumps"),
            "typed_loads": recorded("typed_loads"),
        },
    )
    assert json_loads("[1]") == "loads"
    hooks: Hooks = {Path: (str, Path)}
    assert (
        json_dumps(
            [1],
            dialect="python",
            hooks=hooks,
            sort=False,
            ensure_ascii=False,
            separators=(",", ":"),
            indent=2,
            sort_keys=True,
        )
        == "dumps"
    )
    assert (
        json_loads("[1]", list[int], hooks=hooks, allow_imports=True, strict=False)
        == "typed_loads"
    )
    assert calls == [
        ("loads", ("[1]",), {}),
        (
            "dumps",
            ([1],),
            {
                "dialect": "python",
                "hooks": hooks,
                "sort": False,
                "ensure_ascii": False,
                "separators": (",", ":"),
                "indent": 2,
                "sort_keys": True,
            },
        ),
        (
            "typed_loads",
            ("[1]", list[int]),
            {"hooks": hooks, "allow_imports": True, "strict": False},
        ),
    ]


_SERVED: Final[dict[tuple[bool, bool], frozenset[str]]] = {
    (True, True): frozenset({"loads", "dumps", "typed_loads"}),
    (True, False): frozenset({"loads", "dumps"}),
    (False, True): frozenset({"loads", "typed_loads"}),
    (False, False): frozenset(),
}
"""The core's JSON functions that serve, by whether its writer and reader do."""


def _walkers(monkeypatch: pytest.MonkeyPatch, *, writes: bool, reads: bool) -> None:
    """Make ``wire`` see the core's writer and reader serving, or not."""
    monkeypatch.setattr(wire, "to_plain", len if writes else _python_to_plain)
    monkeypatch.setattr(wire, "from_plain", len if reads else _python_from_plain)


@pytest.mark.parametrize(("writes", "reads"), list(_SERVED))
def test_the_core_serves_json_only_where_its_walkers_do(
    monkeypatch: pytest.MonkeyPatch,
    writes: bool,
    reads: bool,
) -> None:
    core = types.ModuleType("treekle._rust")
    for name in ("json_loads_plain", "json_dumps", "json_loads_typed"):
        setattr(core, name, len)
    monkeypatch.setitem(sys.modules, "treekle._rust", core)
    _walkers(monkeypatch, writes=writes, reads=reads)
    found = wire._core_json()
    assert set(found) == _SERVED[writes, reads]
    assert all(function is len for function in found.values())


def test_a_core_without_json_functions_is_left_to_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "treekle._rust", types.ModuleType("_rust"))
    _walkers(monkeypatch, writes=True, reads=True)
    with pytest.warns(RuntimeWarning, match="Rust core does not match"):
        assert wire._core_json() == {}


def test_a_core_without_json_functions_warns_and_uses_json() -> None:
    # A core built before the JSON functions existed still sets up its walkers.
    completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import types
                import warnings

                core = types.ModuleType("treekle._rust")
                core.setup_encode = core.setup_decode = lambda module: None
                core.to_plain = lambda obj, **options: obj
                core.from_plain = lambda data, target, **options: data
                sys.modules["treekle._rust"] = core
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    from treekle import wire

                assert wire._CORE_JSON == {}
                print(wire.json_dumps([1]), wire.json_loads("[2]"), wire.json_loads("[3]", list))
                print(
                    [
                        str(warning.message).split(" (")[0]
                        for warning in caught
                        if warning.category is RuntimeWarning
                    ],
                )
                """,
            ),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "TREEKLE_PURE_PYTHON": ""},
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "[1] [2] [3]",
        '["treekle\'s Rust core does not match this treekle"]',
    ]


def test_pure_python_serves_every_wire_step() -> None:
    # The choice is made at import, so only a fresh interpreter sees it; an
    # installed core must not serve a step its unset-up walkers would need.
    completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                from dataclasses import dataclass

                from treekle import json_dumps, json_loads, wire

                @dataclass
                class Point:
                    x: int

                assert wire._CORE_JSON == {}
                text = json_dumps(Point(1))
                assert json_loads(text, Point) == Point(1)
                print(text)
                """,
            ),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "TREEKLE_PURE_PYTHON": "1"},
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == '{"py/object": "__main__.Point", "x": 1}'


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
