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
from functools import partial
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


type _Values = Callable[[st.DrawFn], object]
"""Draws one value of a declared type."""


type _Schema = tuple[_Hint, _Values]
"""A declared type, and how to draw a value of it."""


# Not st.text()'s default alphabet, characters(codec="utf-8"): Hypothesis tabulates
# that codec by encoding all 1.1M code points, 0.5s on CI's cold cache. UTF-8
# encodes every code point but the surrogates, so this is the same alphabet.
_CHARS: Final = st.characters(exclude_categories=["Cs"])


_LEAVES: Final = st.sampled_from(
    [
        (int, st.integers()),
        (float, st.floats(allow_nan=False) | st.integers()),
        (str, st.text(_CHARS, max_size=4)),
        (bool, st.booleans()),
        (type(None), st.none()),
        (int | str, st.integers() | st.text(_CHARS, max_size=4)),
    ],
)


_MEMBERS: Final = st.frozensets(st.integers(), max_size=3)


_SETS: Final = st.sampled_from(
    [
        (frozenset[int], _MEMBERS),
        (set[int], _MEMBERS.map(set)),
        (AbstractSet[int], _MEMBERS | _MEMBERS.map(set)),
    ],
)


_NUMBERS: Final = st.lists(st.integers(), max_size=3)


_HOLDERS: Final = st.builds(
    Holder,
    numbers=_NUMBERS | _NUMBERS.map(tuple),
    ratio=st.floats(allow_nan=False) | st.integers(),
    box=st.none() | st.builds(Box, item=st.integers()),
)


_KEYS: Final = st.text(_CHARS, max_size=8) | st.sampled_from(
    ["py/tuple", "py/object", "json://1"],
)
"""Mapping keys, including ones that read like tags."""


_SIZES: Final = st.integers(min_value=0, max_value=3)
"""How many items a drawn container holds."""


_COIN: Final = st.booleans()


_KINDS: Final = st.sampled_from(
    [
        "leaf",
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
    ],
)


_CONCRETE_TYPES: list[Mapping[type, type]] = [
    {},
    {Mapping: MappingProxyType},
    {Sequence: tuple},
    {Mapping: MappingProxyType, Sequence: tuple, AbstractSet: set},
]


_CONCRETE: Final = st.sampled_from(_CONCRETE_TYPES)


@st.composite
def _cases(draw: st.DrawFn) -> tuple[object, object, Mapping[type, type]]:
    """Draw ``(T, value, default_concrete_types)``."""
    declared, values = _schema(draw, depth=3)
    return declared, values(draw), draw(_CONCRETE)


# Values draw through plain functions over the fixed strategies above, never a
# strategy built per draw: Hypothesis validates each new strategy, which cost 4x
# the walks under test at CI's 200 examples.
def _schema(draw: st.DrawFn, *, depth: int) -> _Schema:
    """Draw a declared type at most ``depth`` containers deep, and its values."""
    kind = draw(_KINDS) if depth else "leaf"
    if kind in ("leaf", "set", "holder"):
        hint, values = (
            (Holder, _HOLDERS)
            if kind == "holder"
            else draw(_LEAVES if kind == "leaf" else _SETS)
        )
        return hint, partial(_one, values)
    item, items = _schema(draw, depth=depth - 1)
    many = partial(_many, items)
    tuples = partial(_tuple, items)
    entries = partial(_entries, items)
    proxies = partial(_proxy, items)
    if kind == "list":
        return GenericAlias(list, (item,)), many
    if kind == "tuple":
        return GenericAlias(tuple, (item, ...)), tuples
    if kind == "sequence":
        return GenericAlias(Sequence, (item,)), partial(_either, many, tuples)
    if kind == "mapping":
        return GenericAlias(Mapping, (str, item)), partial(_either, entries, proxies)
    if kind == "proxy":
        return GenericAlias(MappingProxyType, (str, item)), proxies
    if kind == "optional":
        optional = item | None
        assert isinstance(optional, type | GenericAlias | UnionType)
        return optional, partial(_either, _none, items)
    if kind == "object":
        return object, items
    if kind == "box":
        return Box, partial(_box, items)
    second, seconds = _schema(draw, depth=depth - 1)
    if kind == "fixed":
        return GenericAlias(tuple, (item, second)), partial(_pair, items, seconds)
    either = item | second
    assert isinstance(either, type | GenericAlias | UnionType)
    return either, partial(_either, items, seconds)


def _one(values: st.SearchStrategy[object], draw: st.DrawFn) -> object:
    return draw(values)


def _many(items: _Values, draw: st.DrawFn) -> list[object]:
    return [items(draw) for _ in range(draw(_SIZES))]


def _tuple(items: _Values, draw: st.DrawFn) -> tuple[object, ...]:
    return tuple(_many(items, draw))


def _entries(items: _Values, draw: st.DrawFn) -> dict[str, object]:
    return {draw(_KEYS): items(draw) for _ in range(draw(_SIZES))}


def _proxy(items: _Values, draw: st.DrawFn) -> MappingProxyType[str, object]:
    return MappingProxyType(_entries(items, draw))


def _either(first: _Values, second: _Values, draw: st.DrawFn) -> object:
    return (second if draw(_COIN) else first)(draw)


def _pair(first: _Values, second: _Values, draw: st.DrawFn) -> tuple[object, object]:
    return first(draw), second(draw)


def _none(draw: st.DrawFn) -> None:
    del draw


def _box(items: _Values, draw: st.DrawFn) -> Box:
    return Box(item=items(draw))


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
