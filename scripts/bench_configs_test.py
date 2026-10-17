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
        "msgspec",
        "msgspec-dataclass",
        "treekle-rust",
        "treekle-python",
        "pickle",
        "jsonpickle",
    )
    out = capsys.readouterr().out
    assert "skipped broken: exp000 raised" in out
    encode, rest = out.split("**Decode (analogue)**")
    decode, summary = rest.split("**Summary**")
    header = "| config | msgspec | msgspec-dataclass | treekle-rust | treekle-python | pickle | jsonpickle | msgspec-dataclass vs msgspec | treekle-rust vs msgspec | treekle-python vs msgspec | pickle vs msgspec | jsonpickle vs msgspec |"
    assert header in encode
    assert header in decode
    toy = next(line for line in decode.splitlines() if line.startswith("| toy |"))
    # Without tags, a dataclass among dicts under ``object`` decodes as a dict.
    assert toy.count("❌") == 2
    assert toy.split(" | ")[2] == "❌"
    assert toy.endswith("% |")
    nan = next(line for line in decode.splitlines() if line.startswith("| nan |"))
    # The msgspec stand-ins hand back the same float object, which equals itself.
    assert nan.count("❌") == 8
    twins = next(line for line in decode.splitlines() if line.startswith("| twins |"))
    assert twins.count("❌") == 2
    assert twins.split(" | ")[2] == "❌"
    total = next(line for line in decode.splitlines() if "**total**" in line)
    assert total.count("❌") == 10
    assert "❌" not in encode
    assert (
        "| library | encode (ms) | encode (%) | decode (ms) | decode (%) |" in summary
    )
    rows = {
        line.split(" | ")[0]: line
        for line in summary.splitlines()
        if line.startswith("| ")
    }
    assert rows["| msgspec"].split(" | ")[2] == "+0%"
    assert rows["| msgspec"].endswith("| +0% |")
    assert rows["| pickle"].endswith("| ❌ | ❌ |")
    assert rows["| pickle"].count("%") == 1
    # Rows run fastest encode first, by the totals the encode table prints.
    totals = next(line for line in encode.splitlines() if "**total**" in line)
    names = [str(name) for name in header.split(" | ")[1:7]]
    by_name = dict(zip(names, totals.split(" | ")[1:7], strict=True))
    order = [line.split(" | ")[0].removeprefix("| ") for line in summary.splitlines()]
    order = [name for name in order if name in by_name]
    assert len(order) == 6
    assert [float(by_name[name]) for name in order] == sorted(
        float(by_name[name]) for name in order
    )


def test_without_msgspec_the_tables_have_no_percent_columns(
    bench: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench("treekle-rust", "pickle")
    out = capsys.readouterr().out
    assert "| config | treekle-rust | pickle |\n" in out
    assert "| library | encode (ms) | decode (ms) |\n" in out
    assert "%" not in out


def test_the_flags_default_to_every_library_on_installed_priml() -> None:
    parser = argparse.ArgumentParser()
    bench_configs._add_arguments(parser)
    assert vars(parser.parse_args([])) == {
        "priml": "priml",
        "columns": list(bench_configs._CODECS),
        "budget_sec": 0.002,
        "repeat": 5,
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
    if isinstance(obj, list | tuple):
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
