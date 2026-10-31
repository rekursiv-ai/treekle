"""Property tests of the contract in ``treekle``'s package docstring.

Each case draws a declared type ``T``, a value of it, and a
``default_concrete_types``; rule 5 says the value reads back with every node's
type kept, through either walker, and both walkers write and read it alike.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass
from types import GenericAlias, MappingProxyType, UnionType
from typing import Final, cast

from hypothesis import (
    given,
    strategies as st,
)

import pytest

from treekle import deserialize, serialize


@dataclass(frozen=True, kw_only=True, slots=True)
class Box:
    item: object = None


@dataclass(frozen=True, kw_only=True, slots=True)
class Holder:
    numbers: Sequence[int] = ()
    ratio: float = 0.0
    box: Box | None = None


type _Hint = type | GenericAlias | UnionType
"""A declared type built at runtime, which supports ``|``."""


type _Schema = tuple[_Hint, st.SearchStrategy[object]]
"""A declared type, and the values of it to draw."""


_LEAVES: list[_Schema] = [
    (int, st.integers()),
    (float, st.floats(allow_nan=False) | st.integers()),
    (str, st.text(max_size=4)),
    (bool, st.booleans()),
    (type(None), st.none()),
    (int | str, st.integers() | st.text(max_size=4)),
]


_KEYS: Final = st.text(max_size=8) | st.sampled_from(
    ["py/tuple", "py/object", "json://1"],
)
"""Mapping keys, including ones that read like tags."""


_CONCRETE: list[Mapping[type, type]] = [
    {},
    {Mapping: MappingProxyType},
    {Sequence: tuple},
    {Mapping: MappingProxyType, Sequence: tuple, AbstractSet: set},
]


@st.composite
def _schemas(draw: st.DrawFn, depth: int) -> _Schema:
    """Draw a declared type at most ``depth`` containers deep, and its values."""
    kinds = ["leaf"]
    if depth:
        kinds += [
            "list",
            "tuple",
            "fixed",
            "sequence",
            "mapping",
            "proxy",
            "set",
            "optional",
            "either",
            "object",
            "box",
            "holder",
        ]
    kind = draw(st.sampled_from(kinds))
    if kind == "leaf":
        return draw(st.sampled_from(_LEAVES))
    if kind == "set":
        members = st.frozensets(st.integers(), max_size=3)
        return draw(
            st.sampled_from(
                [
                    (frozenset[int], members),
                    (set[int], members.map(set)),
                    (AbstractSet[int], members | members.map(set)),
                ],
            ),
        )
    if kind == "holder":
        boxes = st.none() | st.builds(Box, item=st.integers())
        numbers = st.lists(st.integers(), max_size=3)
        return Holder, st.builds(
            Holder,
            numbers=numbers | numbers.map(tuple),
            ratio=st.floats(allow_nan=False) | st.integers(),
            box=boxes,
        )
    inner: _Schema = draw(_schemas(depth - 1))
    item, items = inner
    many = st.lists(items, max_size=3)
    entries = st.dictionaries(_KEYS, items, max_size=3)
    if kind == "list":
        return GenericAlias(list, (item,)), many
    if kind == "tuple":
        return GenericAlias(tuple, (item, ...)), many.map(tuple)
    if kind == "sequence":
        return GenericAlias(Sequence, (item,)), many | many.map(tuple)
    if kind == "mapping":
        return GenericAlias(Mapping, (str, item)), entries | entries.map(
            MappingProxyType,
        )
    if kind == "proxy":
        return GenericAlias(MappingProxyType, (str, item)), entries.map(
            MappingProxyType,
        )
    if kind == "optional":
        optional = item | None
        assert isinstance(optional, type | GenericAlias | UnionType)
        return optional, st.none() | items
    if kind == "object":
        return object, items
    if kind == "box":
        return Box, items.map(lambda value: Box(item=value))
    other: _Schema = draw(_schemas(depth - 1))
    second, seconds = other
    if kind == "fixed":
        return GenericAlias(tuple, (item, second)), st.tuples(items, seconds)
    either = item | second
    assert isinstance(either, type | GenericAlias | UnionType)
    return either, items | seconds


@st.composite
def _cases(draw: st.DrawFn) -> tuple[object, object, Mapping[type, type]]:
    """Draw ``(T, value, default_concrete_types)``."""
    declared, values = draw(_schemas(3))
    return declared, draw(values), draw(st.sampled_from(_CONCRETE))


def _kinds(value: object) -> object:
    """Return ``value`` with each node shown as its type and its parts' kinds."""
    if isinstance(value, Box):
        return Box, [("item", _kinds(value.item))]
    if isinstance(value, Holder):
        return Holder, [
            ("numbers", _kinds(value.numbers)),
            ("ratio", _kinds(value.ratio)),
            ("box", _kinds(value.box)),
        ]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return type(mapping), [(key, _kinds(item)) for key, item in mapping.items()]
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return type(items), [_kinds(item) for item in items]
    if isinstance(value, (set, frozenset)):
        members = cast(AbstractSet[object], value)
        return type(members), frozenset(_kinds(member) for member in members)
    return type(value), value


type _ToPlain = Callable[..., object]
type _FromPlain = Callable[..., object]


_WALKERS: list[tuple[_ToPlain, _FromPlain]] = [
    (serialize.to_plain, deserialize.from_plain),
    (serialize._python_to_plain, deserialize._python_from_plain),
]


_BUILT = _WALKERS[0] != _WALKERS[1]


@pytest.mark.parametrize(
    ("to_plain", "from_plain"),
    _WALKERS if _BUILT else _WALKERS[1:],
    ids=["native", "python"] if _BUILT else ["python"],
)
@given(case=_cases())
def test_a_value_round_trips_with_every_type_kept(
    to_plain: _ToPlain,
    from_plain: _FromPlain,
    case: tuple[object, object, Mapping[type, type]],
) -> None:
    declared, value, concrete = case
    for target in (declared, object):
        tree = to_plain(value, target=target, default_concrete_types=concrete)
        back = from_plain(
            tree,
            target,
            default_concrete_types=concrete,
            allow_imports=True,
        )

        assert _kinds(back) == _kinds(value)


@pytest.mark.skipif(not _BUILT, reason="the Rust core is not built")
@given(case=_cases())
def test_both_walkers_write_and_read_a_value_alike(
    case: tuple[object, object, Mapping[type, type]],
) -> None:
    declared, value, concrete = case
    tree = serialize.to_plain(value, target=declared, default_concrete_types=concrete)
    python_tree = serialize._python_to_plain(
        value,
        target=declared,
        default_concrete_types=concrete,
    )
    native = deserialize.from_plain(
        tree,
        declared,
        default_concrete_types=concrete,
        allow_imports=True,
    )
    python = deserialize._python_from_plain(
        tree,
        declared,
        default_concrete_types=concrete,
        allow_imports=True,
    )

    assert _kinds(tree) == _kinds(python_tree)
    assert _kinds(native) == _kinds(python)


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
