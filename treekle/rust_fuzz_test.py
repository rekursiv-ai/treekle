"""Property tests that the Rust core writes and reads as Python's walk does.

Each case makes dataclasses at runtime -- slotted, dict-backed or frozen, their
fields leaves, containers, unions and other records -- then draws a value of
one and the options to write and read it with. ``to_plain``, ``json_dumps``,
``from_plain`` and ``json_loads`` must each give what Python's walk gives, or
raise what it raises. A read is also given the written tree with one node
replaced, so the two must fail alike as well.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, make_dataclass
from enum import IntEnum, StrEnum
from functools import partial
from pathlib import PurePosixPath
from types import GenericAlias, MappingProxyType
from typing import Literal, cast
from uuid import UUID

import builtins
import copy
import datetime as dt
import itertools
import json
import math
import operator
import re

from hypothesis import (
    assume,
    given,
    strategies as st,
)

import pytest

from treekle import deserialize, serialize, wire
from treekle.types import ReadError


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class _Level(IntEnum):
    LOW = 1
    HIGH = 2


type _Schema = tuple[object, st.SearchStrategy[object]]
"""A declared type, and the values of it to draw."""


_LEAVES: list[_Schema] = [
    (int, st.integers()),
    (float, st.floats() | st.integers()),
    (str, st.text(max_size=6)),
    (bool, st.booleans()),
    (type(None), st.none()),
    (bytes, st.binary(max_size=4)),
    (complex, st.complex_numbers()),
    (_Color, st.sampled_from(_Color)),
    (_Level, st.sampled_from(_Level)),
    (PurePosixPath, st.text("ab/", max_size=5).map(PurePosixPath)),
    (UUID, st.uuids()),
    (dt.datetime, st.datetimes(timezones=st.none() | st.just(dt.UTC))),
    (dt.date, st.dates()),
    (Literal["a", "b"], st.sampled_from(["a", "b"])),
]


_WRONG = st.one_of(*(values for _, values in _LEAVES))
"""A leaf of any type, to stand where its declared type does not fit."""


_KEYS = st.text(max_size=6) | st.sampled_from(
    ["py/object", "py/tuple", "py/reduce", "py/type", "py/float", "json://1"],
)
"""Mapping keys, including ones that read like tags."""


_NODES = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()
    | st.text(max_size=4)
    | st.lists(st.integers() | st.text(max_size=3), max_size=2)
    | st.dictionaries(_KEYS, st.integers() | st.text(max_size=3), max_size=2)
)
"""Plain nodes to put in place of a written one."""


_CONCRETE: list[Mapping[type, type]] = [
    {},
    {Mapping: MappingProxyType},
    {Sequence: tuple},
]


_NAMES = ("a", "b", "x", "y", "value", "items")
"""Field names a record draws from."""


_SERIAL = itertools.count()
"""Numbers each record class made, so each has its own name in this module."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _Case:
    """A value, the type it is declared as, and how to write and read it."""

    value: object
    target: object
    sort: bool
    dialect: Literal["json", "python"]
    mutable: bool
    concrete: Mapping[type, type]
    strict: bool
    ensure_ascii: bool
    indent: int | None
    sort_keys: bool


@st.composite
def _cases(draw: st.DrawFn) -> _Case:
    """Draw a record type, a value holding its instances, and the options."""
    kind, values = draw(_records(depth=2))
    items = draw(st.lists(values, min_size=1, max_size=3))
    if draw(st.booleans()):
        items.append(items[0])
    value, target = draw(
        st.sampled_from(
            [(items[0], kind), (items, GenericAlias(list, (kind,))), (items, object)],
        ),
    )
    return _Case(
        value=value,
        target=target,
        sort=draw(st.booleans()),
        dialect=draw(st.sampled_from(["json", "python"])),
        mutable=draw(st.booleans()),
        concrete=draw(st.sampled_from(_CONCRETE)),
        strict=draw(st.booleans()),
        ensure_ascii=draw(st.booleans()),
        indent=draw(st.none() | st.just(2)),
        sort_keys=draw(st.booleans()),
    )


@st.composite
def _records(draw: st.DrawFn, *, depth: int) -> _Schema:
    """Draw a new dataclass whose fields go ``depth`` types deep, and its values."""
    names = draw(st.lists(st.sampled_from(_NAMES), unique=True, max_size=4))
    fields = [(name, *draw(_schemas(depth=depth))) for name in names]
    slots = draw(st.booleans())
    name = f"_Record{next(_SERIAL)}"
    kind = make_dataclass(
        name,
        [(field, draw(_hints(hint))) for field, hint, _ in fields],
        slots=slots,
        frozen=draw(st.booleans()),
        module=__name__,
    )
    globals()[name] = kind
    values = st.builds(
        kind,
        **{field: _sometimes_wrong(each) for field, _, each in fields},
    )
    if slots:
        return kind, values
    return kind, st.tuples(values, st.none() | _WRONG).map(_with_extra)


def _with_extra(drawn: tuple[object, object]) -> object:
    """Return the instance given, with the attribute given set beyond its fields."""
    value, extra = drawn
    if extra is not None:
        vars(value)["extra"] = extra
    return value


@st.composite
def _hints(draw: st.DrawFn, hint: object) -> object:
    """Draw ``hint`` itself, its name as text, or text naming nothing."""
    name = getattr(hint, "__name__", None)
    named = isinstance(name, str) and (
        globals().get(name) is hint or getattr(builtins, name, None) is hint
    )
    forms = [hint, "_Unresolved"] + ([name] if named else [])
    return draw(st.sampled_from(forms))


@st.composite
def _schemas(draw: st.DrawFn, *, depth: int) -> _Schema:
    """Draw a declared type at most ``depth`` containers deep, and its values."""
    kinds = ["leaf"]
    if depth:
        kinds += [
            "list",
            "tuple",
            "fixed",
            "sequence",
            "dict",
            "int keys",
            "mapping",
            "set",
            "optional",
            "either",
            "object",
            "record",
        ]
    kind = draw(st.sampled_from(kinds))
    if kind == "leaf":
        return draw(st.sampled_from(_LEAVES))
    if kind == "set":
        members = st.frozensets(st.integers() | st.sampled_from(_Color), max_size=3)
        return draw(
            st.sampled_from(
                [
                    (GenericAlias(frozenset, (int | _Color,)), members),
                    (GenericAlias(set, (int,)), members.map(set)),
                ],
            ),
        )
    if kind == "record":
        return draw(_records(depth=depth - 1))
    item, items = draw(_schemas(depth=depth - 1))
    items = _sometimes_wrong(items)
    many = st.lists(items, max_size=3)
    entries = st.dictionaries(_KEYS, items, max_size=3)
    if kind == "list":
        return GenericAlias(list, (item,)), many
    if kind == "tuple":
        return GenericAlias(tuple, (item, ...)), many.map(tuple)
    if kind == "sequence":
        return GenericAlias(Sequence, (item,)), many | many.map(tuple)
    if kind == "dict":
        return GenericAlias(dict, (str, item)), entries
    if kind == "int keys":
        return GenericAlias(dict, (int, item)), st.dictionaries(
            st.integers(),
            items,
            max_size=3,
        )
    if kind == "mapping":
        return GenericAlias(Mapping, (str, item)), entries | entries.map(
            MappingProxyType,
        )
    if kind == "optional":
        return operator.or_(item, None), st.none() | items
    if kind == "object":
        return object, items
    second, seconds = draw(_schemas(depth=depth - 1))
    if kind == "fixed":
        return GenericAlias(tuple, (item, second)), st.tuples(items, seconds)
    return operator.or_(item, second), items | seconds


def _sometimes_wrong(values: st.SearchStrategy[object]) -> st.SearchStrategy[object]:
    """Return ``values``, now and then a leaf of another type instead."""
    return st.one_of(values, values, values, _WRONG)


def _python_tree(case: _Case) -> object:
    """Return Python's tree for the case's value; skip a value Python rejects."""
    try:
        return serialize._python_to_plain(
            case.value,
            target=case.target,
            sort=case.sort,
            dialect=case.dialect,
            mutable=case.mutable,
            default_concrete_types=case.concrete,
        )
    except (TypeError, ValueError):
        assume(False)
        raise


def _places(tree: object) -> list[Callable[[object], None]]:
    """Return a setter for each node below the root of ``tree``'s lists and dicts."""
    places: list[Callable[[object], None]] = []
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            items = cast("list[object]", node)
            places.extend(
                partial(_set_item, items, index) for index in range(len(items))
            )
            pending.extend(items)
        elif isinstance(node, dict):
            entries = cast("dict[object, object]", node)
            places.extend(partial(_set_entry, entries, key) for key in entries)
            pending.extend(entries.values())
    return places


def _set_item(items: list[object], index: int, node: object) -> None:
    items[index] = node


def _set_entry(entries: dict[object, object], key: object, node: object) -> None:
    entries[key] = node


def _read(read: Callable[[], object]) -> object:
    """Return ``_outcome`` of a read, its value as Python writes it."""
    return _outcome(lambda: serialize._python_to_plain(read()))


# An error names an object by ``repr``, which holds its address: two walks that
# agree still build two objects, so the address is not compared.
def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result shape, or the type and text of what it raised."""
    try:
        return ("ok", _shape(call()))
    except ReadError as error:
        bad = [(list(at), invalid.reason) for at, invalid in error.bad.items()]
        partial_value = serialize._python_to_plain(error.partial)
        return ("raise", "ReadError", _unplaced(error), bad, _shape(partial_value))
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either walker raises.
        return ("raise", type(error).__name__, _unplaced(error))


def _unplaced(error: Exception) -> str:
    """Return the error's text with each object address removed."""
    return re.sub(r" at 0x[0-9a-f]+", "", str(error))


def _shape(tree: object) -> object:
    """Return ``tree`` with each container's exact type made visible."""
    name = type(tree).__name__
    if isinstance(tree, (dict, MappingProxyType)):
        items = cast("dict[object, object]", tree).items()
        return (name, [(_shape(key), _shape(value)) for key, value in items])
    if isinstance(tree, (list, tuple)):
        return (name, [_shape(item) for item in cast("list[object]", tree)])
    if isinstance(tree, float) and math.isnan(tree):
        return ("nan",)
    return (name, tree)


_needs_rust = pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)


@_needs_rust
@given(case=_cases())
def test_to_plain_matches_python(case: _Case) -> None:
    def write(walker: Callable[..., object]) -> object:
        return _outcome(
            lambda: walker(
                case.value,
                target=case.target,
                sort=case.sort,
                dialect=case.dialect,
                mutable=case.mutable,
                default_concrete_types=case.concrete,
            ),
        )

    assert write(serialize.to_plain) == write(serialize._python_to_plain)


@_needs_rust
@given(case=_cases())
def test_json_dumps_matches_python(case: _Case) -> None:
    actual = _outcome(
        lambda: wire.json_dumps(
            case.value,
            target=case.target,
            sort=case.sort,
            dialect=case.dialect,
            default_concrete_types=case.concrete,
            ensure_ascii=case.ensure_ascii,
            indent=case.indent,
            sort_keys=case.sort_keys,
        ),
    )

    expected = _outcome(
        lambda: json.dumps(
            serialize._python_to_plain(
                case.value,
                target=case.target,
                sort=case.sort,
                dialect=case.dialect,
                default_concrete_types=case.concrete,
            ),
            ensure_ascii=case.ensure_ascii,
            indent=case.indent,
            sort_keys=case.sort_keys,
        ),
    )
    assert actual == expected


@_needs_rust
@given(case=_cases(), data=st.data())
def test_from_plain_matches_python(case: _Case, data: st.DataObject) -> None:
    tree = _python_tree(case)
    # A tree no walk wrote may name any code to import and call. Only lists and
    # dicts take a new node, so only a mutable tree is changed.
    allow_imports = not (case.mutable and data.draw(st.booleans()))
    if not allow_imports:
        tree = copy.deepcopy(tree)
        places = _places(tree)
        if places:
            data.draw(st.sampled_from(places))(data.draw(_NODES))

    def read(reader: Callable[..., object]) -> object:
        return _read(
            lambda: reader(
                tree,
                case.target,
                allow_imports=allow_imports,
                strict=case.strict,
                default_concrete_types=case.concrete,
            ),
        )

    assert read(deserialize.from_plain) == read(deserialize._python_from_plain)


@_needs_rust
@given(case=_cases())
def test_json_loads_matches_python(case: _Case) -> None:
    tree = _python_tree(case)
    try:
        text = json.dumps(tree)
    except (TypeError, ValueError):
        assume(False)
        raise

    actual = _read(
        lambda: wire.json_loads(
            text,
            case.target,
            allow_imports=True,
            strict=case.strict,
            default_concrete_types=case.concrete,
        ),
    )

    expected = _read(
        lambda: deserialize._python_from_plain(
            json.loads(text),
            case.target,
            allow_imports=True,
            strict=case.strict,
            default_concrete_types=case.concrete,
        ),
    )
    assert actual == expected


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
