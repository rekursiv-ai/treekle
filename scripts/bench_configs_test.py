"""The config benchmark builds analogues every library round-trips, and renders both tables.

msgspec and jsonpickle are not treekle dependencies, so stand-ins replace them:
jsonpickle by pickle, msgspec by a converter that builds the requested Struct.
"""

from __future__ import annotations

from types import GenericAlias, SimpleNamespace, UnionType
from typing import TYPE_CHECKING, cast, get_args, get_origin

import argparse
import dataclasses
import functools
import gc
import pickle
import sys
import textwrap

import pytest

from scripts import bench_configs


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path


_TOY = """
from dataclasses import dataclass, field


@dataclass
class Leaf:
    x: int = 1


@dataclass
class Toy:
    name: str = "toy"
    rate: float = 0.5
    leaf: Leaf = field(default_factory=Leaf)
    leaves: list[object] = field(default_factory=lambda: [Leaf(), None])
    mixed: list[object] = field(default_factory=lambda: [[1], (2,)])
    maps: list[object] = field(default_factory=lambda: [{"a": 1}, Leaf()])
    empty: list[object] = field(default_factory=list)
    pair: tuple[object, ...] = (1, Leaf())
    unit: tuple[()] = ()
    keys: dict[object, int] = field(default_factory=lambda: {1: 2, (3, 4): 5})
    fn: object = len


def exp000():
    return Toy()
"""


_TWINS = """
from dataclasses import dataclass, field


@dataclass
class Left:
    x: int = 1


@dataclass
class Right:
    y: int = 2


@dataclass
class Twins:
    items: list[object] = field(default_factory=lambda: [Left(), Right()])


def exp000():
    return Twins()
"""


@pytest.fixture
def bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., None]]:
    """Return a runner of the benchmark over a fake loop checkout."""
    baselines = tmp_path / "fakepriml/baselines"
    for name, body in {
        "toy": _TOY,
        "nan": "def exp000():\n    return {'x': float('nan')}\n",
        "broken": "def exp000():\n    raise RuntimeError\n",
        "twins": _TWINS,
    }.items():
        (baselines / name).mkdir(parents=True)
        (baselines / name / "experiments.py").write_text(textwrap.dedent(body))
    (baselines / "toy/experiments_test.py").write_text("def exp000(): ...\n")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setitem(sys.modules, "msgspec", _fake_msgspec())
    monkeypatch.setitem(sys.modules, "jsonpickle", _fake_jsonpickle())
    main = bench_configs.main
    before = set(sys.modules)

    def run(*columns: str) -> None:
        argv = [
            "bench",
            "--priml",
            "fakepriml",
            "--budget-sec",
            "1e-4",
            "--repeat",
            "2",
            "--cold-repeat",
            "1",
        ]
        monkeypatch.setattr(sys, "argv", [*argv, "--columns", *columns])
        assert main() == 0

    yield run
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_every_library_round_trips_the_analogue(
    bench: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench(
        "msgspec-dataclass",
        "msgspec-struct",
        "treekle-rust",
        "treekle-python",
        "pickle",
        "jsonpickle",
    )
    out = capsys.readouterr().out
    assert "skipped broken: exp000 raised" in out
    encode, rest = out.split("**Decode (analogue)**")
    decode, summary = rest.split("**Summary**")
    header = [
        "config",
        "msgspec-dataclass",
        "msgspec-struct",
        "treekle-rust",
        "treekle-python",
        "pickle",
        "jsonpickle",
        "msgspec-dataclass vs msgspec-struct",
        "treekle-rust vs msgspec-struct",
        "treekle-python vs msgspec-struct",
        "pickle vs msgspec-struct",
        "jsonpickle vs msgspec-struct",
    ]
    assert _rows(encode)[0] == header
    assert _rows(decode)[0] == header
    decoded = {row[0]: row for row in _rows(decode)}
    # Without tags, a dataclass among dicts under ``object`` decodes as a dict.
    assert decoded["toy"].count("❌") == 2
    assert decoded["toy"][1] == "❌"
    # The msgspec stand-ins hand back the same float object, which equals itself.
    assert decoded["nan"].count("❌") == 8
    assert decoded["twins"].count("❌") == 2
    assert decoded["twins"][1] == "❌"
    assert decoded["**total**"].count("❌") == 10
    assert "❌" not in encode
    table = _rows(summary)
    assert table[0] == [
        "library",
        "warm encode (ms)",
        "×",
        "warm decode (ms)",
        "×",
        "cold encode (ms)",
        "×",
        "cold decode (ms)",
        "×",
    ]
    rows = {row[0]: row for row in table[1:]}
    assert rows["msgspec-struct (reference)"][2::2] == ["1.0"] * 4
    # Decode fails alike cold as warm; only msgspec-struct's stand-in reads
    # every config back, so every other decode total is ❌, with no multiple.
    failed = {name: row[3:5] for name, row in rows.items() if row[3] == "❌"}
    assert failed == {
        name: ["❌", ""] for name in rows if name != "msgspec-struct (reference)"
    }
    assert {name: row[7:9] == ["❌", ""] for name, row in rows.items()} == {
        name: name in failed for name in rows
    }
    # Every encode total is a time, with its multiple of msgspec-struct's.
    for row in rows.values():
        for cell in (*row[1:3], *row[5:7]):
            assert float(cell) >= 0
    # Rows run in one fixed order, whatever order the columns were asked in.
    assert [row[0] for row in table[1:]] == [
        "msgspec-struct (reference)",
        "msgspec-dataclass",
        "treekle-rust",
        "pickle",
        "treekle-python",
        "jsonpickle",
    ]
    # Every line of a table is as wide as its header, so the raw text aligns.
    for block in (encode, decode, summary):
        lines = [line for line in block.splitlines() if line.startswith("|")]
        assert len({len(line) for line in lines}) == 1


def test_without_msgspec_the_tables_have_no_ratio_columns(
    bench: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench("treekle-rust", "pickle")
    out = capsys.readouterr().out
    tables = [rows for block in out.split("**") if (rows := _rows(block))]
    assert [rows[0] for rows in tables] == [
        ["config", "treekle-rust", "pickle"],
        ["config", "treekle-rust", "pickle"],
        [
            "library",
            "warm encode (ms)",
            "warm decode (ms)",
            "cold encode (ms)",
            "cold decode (ms)",
        ],
    ]
    assert "×" not in out
    assert "vs msgspec-struct" not in out


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (0.0, "0"),
        (0.5133, "0.51"),
        (0.846, "0.85"),
        (1.0, "1.0"),
        (1.6249, "1.6"),
        (9.96, "10"),
        (36.58, "37"),
        (123.4, "120"),
        (None, "❌"),
    ],
)
def test_every_number_shows_two_significant_figures(
    value: float | None,
    shown: str,
) -> None:
    assert bench_configs._cell(value) == shown


def test_the_flags_default_to_every_library_on_installed_priml() -> None:
    parser = argparse.ArgumentParser()
    bench_configs._add_arguments(parser)
    assert vars(parser.parse_args([])) == {
        "priml": "priml",
        "columns": list(bench_configs._CODECS),
        "budget_sec": 0.002,
        "repeat": 5,
        "cold_repeat": 3,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["--columns", "nope"])


def test_treekle_decodes_into_the_analogue_type() -> None:
    calls: list[tuple[object, object, bool]] = []

    def from_plain(data: object, target: object, *, allow_imports: bool) -> object:
        calls.append((data, target, allow_imports))
        return data

    decode, check = bench_configs._prepare_treekle(
        3,
        to_plain=str,
        from_plain=from_plain,
    )
    assert decode() == "3"
    assert calls == [("3", int, True)]
    assert check(3)


def test_a_cold_call_is_timed_once_with_the_collector_held_off() -> None:
    seen: list[bool] = []

    def call() -> str:
        seen.append(gc.isenabled())
        return "done"

    us, result = bench_configs._once(call)
    assert result == "done"
    assert us >= 0
    assert seen == [False]
    assert gc.isenabled()


def test_a_cold_decode_that_raises_or_differs_is_refused() -> None:
    def raises() -> object:
        raise TypeError

    assert bench_configs._cold_decode_us(raises, lambda _: True) is None
    assert bench_configs._cold_decode_us(lambda: 1, lambda _: False) is None
    assert bench_configs._cold_decode_us(lambda: 1, lambda _: True) is not None


def test_each_fresh_mirror_has_classes_of_its_own() -> None:
    @dataclasses.dataclass(kw_only=True, slots=True)
    class Leaf:
        x: int = 1

    first, second = bench_configs._fresh(Leaf()), bench_configs._fresh(Leaf())
    assert type(first) is not type(second)
    assert type(first).__module__ != type(second).__module__
    assert type(first).__name__.startswith("Leaf")
    assert dataclasses.asdict(cast("Leaf", first)) == {"x": 1}


def _rows(block: str) -> list[list[str]]:
    """Return the cells of each line of a markdown table, the rule left out."""
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines()
        if line.startswith("|") and not line.startswith("|-")
    ]


def _fake_jsonpickle() -> SimpleNamespace:
    pickler = SimpleNamespace(flatten=pickle.dumps)
    unpickler = SimpleNamespace(restore=pickle.loads)
    return SimpleNamespace(
        Pickler=functools.partial(_keyed, pickler),
        Unpickler=functools.partial(_keyed, unpickler),
    )


def _keyed(made: SimpleNamespace, *, keys: bool) -> SimpleNamespace:
    assert keys
    return made


def _fake_msgspec() -> SimpleNamespace:
    return SimpleNamespace(
        to_builtins=_to_builtins,
        convert=_convert,
        defstruct=_defstruct,
    )


def _to_builtins(
    obj: object,
    *,
    enc_hook: Callable[[object], object] | None = None,
) -> object:
    if enc_hook is not None:
        return enc_hook(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {
            f.name: _to_builtins(cast("object", getattr(obj, f.name)))
            for f in dataclasses.fields(obj)
        }
        tag = getattr(obj, "__struct_tag__", None)
        return fields if tag is None else {"$type": tag, **fields}
    if isinstance(obj, (list, tuple)):
        items = [_to_builtins(item) for item in cast("list[object]", obj)]
        return items if isinstance(obj, list) else tuple(items)
    if isinstance(obj, dict):
        return {
            k: _to_builtins(v) for k, v in cast("dict[object, object]", obj).items()
        }
    return obj


def _defstruct(
    name: str,
    fields: Iterable[tuple[str, object]],
    **options: object,
) -> type:
    assert options == {"tag": True, "tag_field": "$type", "kw_only": True}
    made = dataclasses.make_dataclass(name, list(fields), kw_only=True)
    setattr(made, "__struct_tag__", name)  # noqa: B010 -- Not a declared attribute.
    return made


# A dict picks its union member by ``$type`` among Struct twins; among plain
# dataclasses, which carry no tag, a union of two is refused.
def _convert(value: object, target: object) -> object:
    """Rebuild ``value`` as ``target``, as msgspec would."""
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        fields = {
            field.name: _convert(
                cast("dict[str, object]", value)[field.name],
                cast("object", field.type),
            )
            for field in dataclasses.fields(target)
        }
        return cast("object", target(**fields))
    if isinstance(target, UnionType):
        members = cast("tuple[type, ...]", get_args(target))
        if isinstance(value, dict):
            mapping = cast("dict[str, object]", value)
            classes = [m for m in members if dataclasses.is_dataclass(m)]
            tagged = {getattr(m, "__struct_tag__", None): m for m in classes}
            if None in tagged and len(classes) > 1:
                raise TypeError("Type unions may not contain more than one dataclass")
            return _convert(mapping, tagged.get(mapping.get("$type"), classes[0]))
        named: dict[str, object] = {member.__name__: member for member in members}
        return _convert(value, named[type(value).__name__])
    if isinstance(target, GenericAlias):
        args = cast("tuple[object, ...]", get_args(target))
        if isinstance(value, dict):
            mapping = cast("dict[object, object]", value)
            return {key: _convert(item, args[1]) for key, item in mapping.items()}
        items = cast("list[object] | tuple[object, ...]", value)
        if get_origin(target) is tuple:
            return tuple(
                _convert(item, arg) for item, arg in zip(items, args, strict=True)
            )
        return [_convert(item, args[0]) for item in items]
    return value


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
