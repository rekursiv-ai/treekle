"""Tests that the Rust core gives what the pure-Python functions give.

``serialize`` and ``deserialize`` keep their Python versions as
``_python_to_plain`` and ``_python_from_plain``; each case here runs both and
compares results, container types included, or the errors they raise.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import MISSING, Field, dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from functools import partial
from importlib import util
from pathlib import Path, PurePosixPath
from types import MappingProxyType, MemberDescriptorType, ModuleType, SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    NamedTuple,
    Protocol,
    TypedDict,
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import datetime as dt
import gc
import importlib
import inspect
import json
import math
import os
import random
import struct
import subprocess
import sys
import textwrap
import threading
import tracemalloc
import weakref

import pytest

from treekle import deserialize, serialize, wire
from treekle.types import ReadError


if TYPE_CHECKING:
    from _collections_abc import dict_items
    from abc import ABCMeta
    from collections.abc import Callable

    from treekle.types import Dialect, Hooks, PlainTree


# Without the Rust core each case compares the Python versions with
# themselves, which still runs this file; only Rust-specific limits skip.
needs_rust = pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain
    or deserialize.from_plain is deserialize._python_from_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)


class _Core(Protocol):
    def setup_decode(self, module: dict[str, object], /) -> None: ...


def _core() -> _Core:
    """Import the built core by name; CI type-checks without building it."""
    return cast("_Core", importlib.import_module("treekle._rust"))


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    LOW = 1


class Pair(NamedTuple):
    a: int
    b: list[int]


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- Built positionally, as the parity cases write it.
class Point:
    x: int = 0
    y: float = 0.0


@dataclass(kw_only=True, slots=True)
class WithClassVar:
    x: int
    y: int = 7
    shared: ClassVar[int] = 0


@dataclass(kw_only=True, slots=True)
class ChangingField:
    x: int = 1

    def __post_init__(self) -> None:
        self.x *= 10


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- Built positionally; nodes link into cycles.
class Node:
    name: str
    children: list[Node] = field(default_factory=list["Node"])
    parent: Node | None = None


class Slotted:
    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


@dataclass(slots=True, kw_only=True)
class AttributeOrder:
    z: int = 1
    a: int = 2


class Bare:
    def __init__(self, x: int) -> None:
        self.x = x


class Opaque:
    __slots__ = ()

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("no")


class MyDict(dict[str, int]):
    pass


class MyList(list[int]):
    pass


class Tensor:
    def __init__(self, data: list[float]) -> None:
        self.data = data


@dataclass(
    frozen=True,
    slots=True,
)  # house-ignore[dataclass] -- Built positionally inside generated deep trees.
class Frozen:
    x: int


type _Retried = list[_Retried] | tuple[_Retried, ...] | UUID


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- Built positionally beside ChangingDict.
class ChangingSlot:
    a: Tensor | int
    b: int


@dataclass  # house-ignore[dataclass] -- The dict-backed twin of ChangingSlot: its fields live in __dict__; built positionally.
class ChangingDict:
    a: Tensor
    b: int


def module_function() -> None:
    pass


def _tensor_data(tensor: Tensor) -> list[float]:
    return tensor.data


def _tensor_tagged(tensor: Tensor) -> dict[str, list[float]]:
    return {"py/x": tensor.data}


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- Built positionally in the parity cases.
class SlotPoint:
    x: int = 0
    y: list[int] = field(default_factory=list[int])


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- Built positionally in the parity cases.
class SlotDynamic:
    a: int = 0
    b: int = 0

    def __getattr__(self, name: str) -> object:
        if name == "b":
            return "from getattr"
        raise AttributeError(name)


def _ninety_nine(self: object) -> int:
    del self
    return 99


# Built with ``type`` because a checker rejects a property overriding a field.
def _slot_shadow() -> type[SlotPoint]:
    """Return a class whose property shadows a slot: ``getattr`` reads the property."""
    return type(
        "SlotShadow",
        (SlotPoint,),
        {"__slots__": (), "__module__": __name__, "x": property(_ninety_nine)},
    )


SlotShadow = _slot_shadow()


class ForeignSlotOwner:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class ForeignSlotSource:
    __slots__ = ("a", "b", "c")

    def __init__(self) -> None:
        self.a = 0
        self.b = 0
        self.c = 0


def _unset_point() -> SlotPoint:
    point = SlotPoint()
    del point.x
    return point


def _unset_dynamic() -> SlotDynamic:
    dynamic = SlotDynamic()
    del dynamic.b
    return dynamic


def _shadow() -> SlotPoint:
    shadow = SlotShadow.__new__(SlotShadow)
    cast("MemberDescriptorType", vars(SlotPoint)["x"]).__set__(shadow, 1)
    cast("MemberDescriptorType", vars(SlotPoint)["y"]).__set__(shadow, [2])
    return shadow


def _cycle() -> list[object]:
    cycle: list[object] = []
    cycle.append(cycle)
    return cycle


def _dict_cycle() -> dict[str, object]:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    return cycle


def _tree() -> Node:
    root = Node("root")
    root.children.append(Node("kid", parent=root))
    return root


class Key:
    """A dict key reduced to an instance of a class made afresh each time.

    A key's own encoding is dropped once its text is made, so each state
    object and its class die before the next key, whose class may then take
    the freed address.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        gc.collect()
        name = f"slot{self.n}"
        state = type("Temp", (), {"__slots__": (name,)})()
        setattr(state, name, self.n)
        return (Key, (self.n,), state)


class Liar(Sequence[object]):
    """A sequence whose ``len`` disagrees with how many items it yields."""

    def __init__(self, length: int, items: list[object]) -> None:
        self.length = length
        self.items = items

    @override
    def __len__(self) -> int:
        return self.length

    @override
    def __getitem__(self, index: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride] -- Accepts any index to raise IndexError as a misbehaving sequence would.  # ty: ignore[invalid-method-override] -- Same.
        if not isinstance(index, int) or index >= len(self.items):
            raise IndexError(index)
        return self.items[index]


class Prefixed(str):
    """A str that claims every prefix."""

    __slots__ = ()

    @override
    def startswith(self, *args: object) -> bool:
        return True


class Unaskable:
    """An option whose truth value raises, to show when it is asked."""

    def __bool__(self) -> bool:
        raise RuntimeError("asked")


class LowersToOne(str):
    """Text whose ``lower()`` is the int ``1`` rather than text."""

    __slots__ = ()

    @override
    def lower(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride] -- The test needs a str whose lower() is not text.  # ty: ignore[invalid-method-override] -- Same.
        return 1


class SaysFalse(str):
    """Text whose ``str()`` is ``"false"`` whatever it holds."""

    __slots__ = ()

    @override
    def __str__(self) -> str:
        return "false"


class LowersToSaysFalse(str):
    """Text whose ``lower()`` is ``SaysFalse("true")``."""

    __slots__ = ()

    @override
    def lower(self) -> str:
        return SaysFalse("true")


class RefusesHash(str):
    """A key whose hash raises."""

    __slots__ = ()

    @override
    def __hash__(self) -> int:
        raise RuntimeError("hashed")


class RefusesEq(str):
    """A key whose ``==`` raises against equal text."""

    __slots__ = ()
    __hash__ = str.__hash__

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and str.__eq__(self, other):
            raise RuntimeError("compared with ==")
        return False


class RefusesNe(str):
    """A key whose ``!=`` raises."""

    __slots__ = ()
    __hash__ = str.__hash__

    @override
    def __ne__(self, other: object) -> bool:
        raise RuntimeError("compared with !=")


class ClaimsTag(str):
    """A key that says it equals ``"py/object"``."""

    __slots__ = ()
    __hash__ = str.__hash__

    @override
    def __eq__(self, other: object) -> bool:
        return other == "py/object" or str.__eq__(self, other)

    @override
    def __ne__(self, other: object) -> bool:
        return not self == other


class Pairs(Mapping[object, object]):
    """A mapping over a list of pairs, so it never hashes its keys."""

    def __init__(self, *pairs: tuple[object, object]) -> None:
        self.pairs = pairs

    @override
    def __getitem__(self, key: object) -> object:
        for have, value in self.pairs:
            if have is key or have == key:
                return value
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[object]:
        return (key for key, _ in self.pairs)

    @override
    def __len__(self) -> int:
        return len(self.pairs)


# Local classes are not cached, and ``get_type_hints`` strips ``Annotated`` into new
# objects each time, which die as soon as a read moves on.
def _typeddict_targets() -> tuple[object, object]:
    """Return a union and a tuple of ``TypedDict`` classes read afresh each call."""

    class Listed(TypedDict):
        x: list[Annotated[int, "unit"]]

    class Mapped(TypedDict):
        y: dict[str, Annotated[int, "unit"]]

    return list[Listed | Mapped], tuple[Listed, Mapped]


TYPEDDICT_UNION, TYPEDDICT_TUPLE = _typeddict_targets()


def _nested(depth: int) -> list[object]:
    root: list[object] = []
    current = root
    for _ in range(depth):
        child: list[object] = []
        current.append(child)
        current = child
    return root


SHARED = [1, 2]
PROXY = MappingProxyType({"a": SHARED, "b": SHARED})

# (id, value, dialect, hooks); every case runs with ``mutable`` on and off.
ENCODE_CASES: list[tuple[str, object, object, Hooks | None]] = [
    ("leaves", [None, True, 1, "s", 1.5, -0.0], "json", None),
    ("nonfinite json", [math.inf, -math.inf, math.nan], "json", None),
    ("nonfinite python", [math.inf, math.nan], "python", None),
    ("complex", [1 + 2j, complex(math.inf, math.nan)], "json", None),
    ("bytes", [b"abc", bytearray(b"xy")], "json", None),
    ("tuple", (1, (2, 3), []), "json", None),
    ("set ints", {3, 10, -1, 2}, "json", None),
    ("set mixed", {"b", "a", 1, (1, 2), None, 2.5}, "json", None),
    ("frozenset", frozenset({frozenset({1}), "x"}), "json", None),
    ("set and shared list", [SHARED, {(1, 2), (3,)}], "json", None),
    (
        "dict keys",
        {1: "a", None: "b", "py/x": 1, "json://y": 2, (1, 2): 3, 2.5: 5},
        "json",
        None,
    ),
    ("bool keys", {True: "a", False: "b"}, "json", None),
    ("int keys", {-12: "a", 2**100: "b", 0: "c"}, "json", None),
    ("shared", [SHARED, SHARED, {"k": SHARED}], "json", None),
    ("cycle", _cycle(), "json", None),
    ("dict cycle", _dict_cycle(), "json", None),
    ("dataclass tree", _tree(), "json", None),
    ("proxy", [PROXY, PROXY], "json", None),
    ("enum", [Color.RED, Color.RED, Level.LOW], "json", None),
    ("decimal fraction", [Decimal("1.5"), Fraction(1, 3)], "json", None),
    ("namedtuple", Pair(1, [2]), "json", None),
    ("partial", partial(int, base=2), "json", None),
    ("path uuid", [Path("/a/b"), PurePosixPath("x/y"), UUID(int=5)], "json", None),
    (
        "datetime",
        [
            dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 2, tzinfo=ZoneInfo("Europe/Paris")),
            dt.date(2026, 1, 2),
        ],
        "json",
        None,
    ),
    (
        "types and functions",
        [int, Point, module_function, max, len, math.floor],
        "json",
        None,
    ),
    ("bound method", [list[int]().append], "json", None),
    ("slotted with dict", Slotted(1, 2), "json", None),
    ("plain objects", [Bare(1), Bare(2)], "json", None),
    ("subclasses", [MyDict(a=1), MyList([1, 2]), OrderedDict(b=1, a=2)], "json", None),
    (
        "hook",
        [Tensor([1.0, math.inf]), Tensor([2.0])],
        "json",
        {Tensor: (_tensor_data, Tensor)},
    ),
    (
        "hook in proxy",
        Tensor([1.0]),
        "json",
        MappingProxyType({Tensor: (_tensor_tagged, Tensor)}),
    ),
    ("opaque", Opaque(), "json", None),
    ("local class", type("Local", (), {})(), "json", None),
    ("lambda", lambda: 0, "json", None),
    ("bad dialect", 1, "yaml", None),
    ("bad dialect type", 1, 3, None),
    ("object", object(), "json", None),
    ("range slice", [range(3), slice(1, 2)], "json", None),
    ("slots", [SlotPoint(1, [2]), _unset_point(), SlotPoint(3)], "json", None),
    ("slots and getattr", [SlotDynamic(1, 2), _unset_dynamic()], "json", None),
    ("slot shadowed by property", _shadow(), "json", None),
    ("lone surrogate tag keys", {"py/\ud800": 1, "json://\udfff": 2}, "json", None),
    (
        "keys reduced to short-lived classes",
        {Key(n): n for n in range(20)},
        "json",
        None,
    ),
]

# (id, data, target, keyword arguments).
DECODE_CASES: list[tuple[str, object, object, dict[str, object]]] = [
    ("untagged dataclass", {"x": 1, "y": 2.0}, Point, {}),
    ("bad field", {"x": "bad"}, Point, {}),
    ("list of int", [1, 2, 3], list[int], {}),
    ("bad list items", [True, "bad", 3], list[int], {}),
    ("default on none", None, str, {"default": ""}),
    ("not strict", {"x": 1, "extra": 2}, Point, {"strict": False}),
    ("strict extra", {"x": 1, "extra": 2}, Point, {}),
    ("nested containers", {"a": [1, 2]}, dict[str, tuple[int, ...]], {}),
    ("int keys", {"json://1": "a"}, dict[int, str], {}),
    ("union", [1, "a", None], list[int | str | None], {}),
    (
        "import without permission",
        {"py/type": "treekle.serialize.to_plain"},
        object,
        {},
    ),
    ("unknown tag", {"py/nope": 1}, object, {}),
    ("unhashable set member", {"py/set": [[1]]}, object, {}),
    ("lone surrogate tag key", {"py/\ud800": 1}, object, {}),
    ("lone surrogate escaped key", {"json://\ud800": 1}, dict[str, int], {}),
    ("str subclass key", {Prefixed("ab"): 1}, object, {}),
    ("long value in a message", list(range(300)), int, {}),
    ("deep value in a message", _nested(20), int, {}),
    ("long str in a message", "x" * 20_000, int, {}),
    ("complex from too few items", Liar(2, [1.0]), complex, {}),
    ("complex from too many items", Liar(2, [1.0, 2.0, 3.0]), complex, {}),
    ("complex with a bad third item", Liar(2, [1.0, 2.0, "x"]), complex, {}),
    (
        "hook entry that is not a pair",
        {"py/hook": [f"{__name__}.Tensor", {"py/b64": "!"}]},
        object,
        {"hooks": {Tensor: 5}},
    ),
    ("strict never asked", 1, int, {"strict": Unaskable()}),
    ("strict asked", "3", int, {"strict": Unaskable()}),
    ("lax bool whose lower() is not text", LowersToOne("x"), bool, {"strict": False}),
    (
        "lax bool whose lower() prints otherwise",
        LowersToSaysFalse("x"),
        bool,
        {"strict": False},
    ),
    ("record key whose != raises", {RefusesNe("x"): 1}, Point, {}),
    ("record key that claims to be the tag", {ClaimsTag("x"): 1}, Point, {}),
    (
        "escaped key whose hash raises",
        Pairs((RefusesHash("json://1"), [1])),
        dict[int, list[int]],
        {},
    ),
    (
        "py/ref through a key whose == raises",
        {RefusesEq("a"): [1], "r": {"py/ref": ["a"]}},
        object,
        {},
    ),
    (
        "allow_imports never asked",
        {"x": 1},
        dict[str, int],
        {"allow_imports": Unaskable()},
    ),
    (
        "allow_imports asked",
        {"py/type": "builtins.int"},
        object,
        {"allow_imports": Unaskable()},
    ),
    (
        "typeddict union with fresh hints",
        [{"x": [1]}, {"y": {"k": 1}}] * 50,
        TYPEDDICT_UNION,
        {},
    ),
    (
        "typeddict tuple with fresh hints",
        ({"x": [1]}, {"y": {"k": 1}}),
        TYPEDDICT_TUPLE,
        {},
    ),
]


def _shape(tree: object) -> object:
    """Return ``tree`` with each container's exact type made visible."""
    name = type(tree).__name__
    if isinstance(tree, (dict, MappingProxyType)):
        items = cast("dict[object, object]", tree).items()
        return (name, [(key, _shape(value)) for key, value in items])
    if isinstance(tree, (list, tuple)):
        return (
            name,
            [_shape(item) for item in cast("list[object]", tree)],
        )
    if isinstance(tree, float) and math.isnan(tree):
        return ("nan",)
    return (type(tree).__name__, tree)


# Each place links to its parent's, so placing a node costs one tuple.
type _Place = tuple[_Place, object] | None


def _path(place: _Place) -> list[object]:
    """Return the keys leading to ``place``."""
    keys: list[object] = []
    while place is not None:
        place, key = place
        keys.append(key)
    return keys[::-1]


def _assert_same_shape(actual: object, expected: object) -> None:
    """Assert ``_shape(actual) == _shape(expected)`` for trees of any depth."""
    pending: list[tuple[object, object, _Place]] = [(actual, expected, None)]
    while pending:
        left, right, at = pending.pop()
        assert type(left) is type(right), _path(at)
        if isinstance(left, (dict, MappingProxyType)):
            left_items = list(cast("Mapping[object, object]", left).items())
            right_items = list(cast("Mapping[object, object]", right).items())
            assert [key for key, _ in left_items] == [key for key, _ in right_items], (
                _path(at)
            )
            pending.extend(
                (one, other, (at, key))
                for (key, one), (_, other) in zip(left_items, right_items, strict=True)
            )
        elif isinstance(left, (list, tuple)):
            left_list = cast("Sequence[object]", left)
            right_list = cast("Sequence[object]", right)
            assert len(left_list) == len(right_list), _path(at)
            pending.extend(
                (one, other, (at, index))
                for index, (one, other) in enumerate(
                    zip(left_list, right_list, strict=True),
                )
            )
        elif isinstance(left, float) and math.isnan(left):
            assert isinstance(right, float), _path(at)
            assert math.isnan(right), _path(at)
        else:
            assert left == right, _path(at)


# A ``ReadError`` also gives its ``_failure``.
def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result shape, or the type and text of what it raised."""
    try:
        return ("ok", _shape(call()))
    except ReadError as error:
        return ("raise", "ReadError", _shape(_failure(error)))
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either walker raises.
        return ("raise", type(error).__name__, str(error))


def _failure(error: ReadError) -> tuple[object, ...]:
    """Return a read's error as plain data: its text, each bad part's place and reason, its partial value as Python encodes it, and the errors that caused it."""
    bad = [(list(at), invalid.reason) for at, invalid in error.bad.items()]
    causes: list[object] = []
    cause = error.__cause__
    while cause is not None:
        causes.append(
            (type(cause).__name__, str(cause), cause.__context__ is cause.__cause__),
        )
        cause = cause.__cause__
    return (
        str(error),
        bad,
        serialize._python_to_plain(error.partial),
        error.__context__ is error.__cause__,
        causes,
    )


def _encoder(
    function: Callable[..., object],
    value: object,
    *,
    mutable: bool,
    dialect: object,
    hooks: Hooks | None,
) -> Callable[[], object]:
    options: dict[str, object] = {"mutable": mutable, "dialect": dialect}
    if hooks is not None:
        options["hooks"] = hooks
    return lambda: function(value, **options)


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
@pytest.mark.parametrize(
    ("value", "dialect", "hooks"),
    [case[1:] for case in ENCODE_CASES],
    ids=[case[0] for case in ENCODE_CASES],
)
def test_to_plain_matches_python(
    value: object,
    dialect: object,
    hooks: Hooks | None,
    mutable: bool,
) -> None:
    python = _outcome(
        _encoder(
            serialize._python_to_plain,
            value,
            mutable=mutable,
            dialect=dialect,
            hooks=hooks,
        ),
    )
    rust = _outcome(
        _encoder(
            serialize.to_plain,
            value,
            mutable=mutable,
            dialect=dialect,
            hooks=hooks,
        ),
    )
    assert rust == python


@pytest.mark.parametrize(
    ("sort", "names"),
    [(True, ["py/object", "a", "z"]), (False, ["py/object", "z", "a"])],
)
def test_object_attribute_sort_matches_python(sort: bool, names: list[str]) -> None:
    value = AttributeOrder()
    expected = serialize._python_to_plain(value, sort=sort)

    assert isinstance(expected, dict)
    assert list(expected) == names
    assert serialize.to_plain(value, sort=sort) == expected


class _Scrambled(str):
    __slots__ = ()

    @override
    def __lt__(self, other: object) -> bool:
        return (sum(map(ord, self)) * 31 + sum(map(ord, str(other)))) % 3 == 0


class _Unordered(str):
    __slots__ = ()

    @override
    def __lt__(self, other: object) -> bool:
        raise ValueError("unordered")


@dataclass(
    frozen=True,
    repr=False,
    slots=True,
)  # house-ignore[dataclass] -- Built positionally by its test.
class _ShownScrambled:
    n: int

    @override
    def __repr__(self) -> str:
        return _Scrambled(f"m{self.n}")


@dataclass(
    frozen=True,
    repr=False,
    slots=True,
)  # house-ignore[dataclass] -- Built positionally by its test.
class _ShownUnordered:
    n: int

    @override
    def __repr__(self) -> str:
        return _Unordered(f"m{self.n}")


@pytest.mark.parametrize("kind", [_ShownScrambled, _ShownUnordered])
def test_set_members_with_inconsistent_repr_order_match_python(
    kind: Callable[[int], object],
) -> None:
    # A repr may be a str subclass whose ``<`` is no total order, or raises.
    value = {kind(n) for n in range(64)}

    assert _outcome(lambda: serialize.to_plain(value)) == _outcome(
        lambda: serialize._python_to_plain(value),
    )


@dataclass(slots=True)  # house-ignore[dataclass] -- Built positionally.
class _SlottedPair:
    a: int
    b: int


@needs_rust
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("planted", ["import path", "stale import path", "slots"])
def test_planted_cache_entries_encode_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    planted: str,
    sort: bool,
) -> None:
    from treekle import reflect  # noqa: PLC0415 -- Only this test reads it.

    value = _SlottedPair(1, 2)
    serialize._python_to_plain(value, sort=sort)
    # The native walk keeps what it read from these caches across calls.
    assert serialize.to_plain(value, sort=sort) == serialize._python_to_plain(
        value,
        sort=sort,
    )
    if planted == "import path":
        monkeypatch.setitem(
            serialize._IMPORT_PATHS,
            id(_SlottedPair),
            (_SlottedPair, "a.b"),
        )
    elif planted == "stale import path":
        monkeypatch.setitem(
            serialize._IMPORT_PATHS,
            id(_SlottedPair),
            (object(), "a.b"),
        )
    else:
        monkeypatch.setitem(reflect._SLOT_NAMES, _SlottedPair, ("b",))
        monkeypatch.setitem(reflect._SORTED_SLOT_NAMES, _SlottedPair, ("b",))

    expected = serialize._python_to_plain(value, sort=sort)

    assert serialize.to_plain(value, sort=sort) == expected
    assert isinstance(expected, dict)
    assert (expected["py/object"] == "a.b") == (planted == "import path")
    assert ("a" in expected) == (planted != "slots")


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("planted", ["import path", "slots", "sorted slots"])
def test_cache_entries_planted_before_a_first_encode_win(
    monkeypatch: pytest.MonkeyPatch,
    planted: str,
    sort: bool,
) -> None:
    from treekle import reflect  # noqa: PLC0415 -- Only this test reads it.

    # A class the core never met, so nothing native was kept for it.
    module = ModuleType(f"treekle_parity_planted_{planted[0]}{sort:d}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pair = dataclass(slots=True)(_class(module.__name__, "Pair", a=int, b=int))
    vars(module)["Pair"] = pair
    if planted == "import path":
        monkeypatch.setitem(serialize._IMPORT_PATHS, id(pair), (pair, "a.b"))
    if planted == "slots":
        monkeypatch.setitem(reflect._SLOT_NAMES, pair, ("b",))
    if planted != "import path":
        # Read alone when sorting: ``attribute_names`` asks it first.
        monkeypatch.setitem(reflect._SORTED_SLOT_NAMES, pair, ("b",))
    value = cast("Callable[..., object]", pair)(a=1, b=2)

    # Native first: Python's walk would fill the other caches for ``pair``.
    actual = serialize.to_plain(value, sort=sort)
    expected = serialize._python_to_plain(value, sort=sort)

    assert actual == expected
    assert isinstance(expected, dict)
    assert (expected["py/object"] == "a.b") == (planted == "import path")
    dropped = planted == "slots" or (planted == "sorted slots" and sort)
    assert ("a" in expected) != dropped


@needs_rust
@pytest.mark.parametrize("source", ["cache", "mapping"])
def test_encoder_changes_during_walk_match_python(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    original = serialize._ENCODERS

    def replaced(obj: object, state: serialize._Encoding) -> str:
        del obj, state
        return "replaced"

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        monkeypatch.setattr(serialize, "_ENCODERS", original)
        monkeypatch.setattr(
            serialize,
            "_ENCODER_CACHE",
            weakref.WeakKeyDictionary[type, serialize._Encoder](),
        )

        def replace(value: Tensor) -> str:
            del value
            if source == "cache":
                serialize._ENCODER_CACHE[Point] = replaced
            else:
                monkeypatch.setattr(
                    serialize,
                    "_ENCODERS",
                    {**original, Point: replaced},
                )
            return "triggered"

        return function(
            [Point(1), Tensor([]), Point(2)],
            hooks={Tensor: (replace, decoded)},
        )

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[object]", expected)[2] == "replaced"


@needs_rust
@pytest.mark.parametrize("proxy", [False, True])
def test_hook_added_to_empty_mapping_during_walk_matches_python(
    proxy: bool,
) -> None:
    hooks: dict[type, tuple[object, object]] = {}

    def encode_tensor(value: Tensor) -> list[float]:
        return value.data

    class InstallHook:
        @override
        def __reduce_ex__(self, protocol: object) -> tuple[object, tuple[()]]:
            del protocol
            hooks[Tensor] = (encode_tensor, Tensor)
            return (list, ())

    def run(function: Callable[..., object]) -> object:
        hooks.clear()
        supplied = MappingProxyType(hooks) if proxy else hooks
        return function([InstallHook(), Tensor([7.0])], hooks=supplied)

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[dict[str, object]]", expected)[1]["py/hook"]


@needs_rust
def test_same_size_dict_key_mutation_matches_python() -> None:
    def run(function: Callable[..., object]) -> object:
        value: dict[str, object] = {}

        class ChangeKeys:
            @override
            def __reduce_ex__(self, protocol: object) -> tuple[object, tuple[()]]:
                del protocol
                del value["a"]
                value["c"] = 3
                return (list, ())

        value.update(a=ChangeKeys(), b=2)
        return _outcome(lambda: function(value))

    expected = run(serialize._python_to_plain)
    assert cast("tuple[object, ...]", expected)[0] == "ok"
    assert run(serialize.to_plain) == expected


@dataclass(kw_only=True, slots=True)
class ThreeFields:
    a: int
    b: str
    c: float = 0.5


class LoudKey(str):
    __slots__ = ()


@needs_rust
@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": "x", "c": 2.0},
        {"c": 2.0, "b": "x", "a": 1},
        {"b": "x", "a": 1},
        {"a": 1, "c": 2.0, "b": "x"},
        {LoudKey("a"): 1, "b": "x", LoudKey("c"): 2.0},
        {"a": 1, "b": "x", "d": 3},
        {"a": 1, "b": "x", "\ud800": 3},
        {"a": 1, "b": "x", 7: 3},
    ],
    ids=[
        "in_order",
        "reversed",
        "missing",
        "skips",
        "str_subclass",
        "unknown",
        "lone_surrogate",
        "int_key",
    ],
)
def test_record_keys_match_fields_as_python_does(data: dict[object, object]) -> None:
    _assert_reads_alike(data, ThreeFields)
    _assert_reads_alike([data, data], list[ThreeFields])


_EDIT_WHILE_READ: list[Callable[[], None]] = []


@dataclass(kw_only=True, slots=True)
class EditsWhileRead:
    x: int

    def __post_init__(self) -> None:
        if self.x == 1:
            for edit in _EDIT_WHILE_READ:
                edit()


@needs_rust
@pytest.mark.parametrize("target", [dict[str, EditsWhileRead], object])
@pytest.mark.parametrize(
    "change",
    ["grow", "shrink", "swap_a_later_key", "swap_the_read_key", "values_only"],
)
def test_dict_changed_while_read_matches_python(
    target: object,
    change: str,
) -> None:
    # The dict is iterated live while each value reads, so these follow
    # CPython's item-iterator checks: a size change, then more items than
    # the dict began with.
    def run(function: Callable[..., object]) -> object:
        data: dict[str, object] = {}

        def edit() -> None:
            if change == "grow":
                data["z"] = {"x": 9}
            elif change == "shrink":
                del data["c"]
            elif change == "swap_a_later_key":
                del data["b"]
                data["d"] = {"x": 4}
            elif change == "swap_the_read_key":
                del data["a"]
                data["d"] = {"x": 4}
            else:
                data["b"] = {"x": 5}

        name = f"{EditsWhileRead.__module__}.{EditsWhileRead.__qualname__}"
        tagged = target is object

        def entry(x: int) -> dict[str, object]:
            if tagged:
                return {"py/reduce": [{"py/type": name}, {"py/tuple": [x]}]}
            return {"x": x}

        data.update(a=entry(1), b=entry(2), c=entry(3))
        _EDIT_WHILE_READ[:] = [edit]
        try:
            return _outcome(
                lambda: serialize._python_to_plain(
                    function(data, target, allow_imports=tagged),
                ),
            )
        finally:
            _EDIT_WHILE_READ.clear()

    expected = run(deserialize._python_from_plain)
    assert run(deserialize.from_plain) == expected


@needs_rust
def test_custom_mapping_items_as_lists_match_python() -> None:
    class ListPairs:
        def __iter__(self) -> Iterator[str]:
            return iter(("x",))

        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: str) -> int:
            if key != "x":
                raise KeyError(key)
            return 1

        def items(self) -> Iterator[list[str | int]]:
            return ([key, self[key]] for key in self)

        def values(self) -> Iterator[int]:
            return (self[key] for key in self)

        def get(self, key: str, default: object = None) -> object:
            return self[key] if key == "x" else default

    cast("ABCMeta", Mapping).register(ListPairs)
    data = ListPairs()
    _assert_reads_alike(data, dict[str, int])
    assert deserialize.from_plain(data, dict[str, int]) == {"x": 1}


@needs_rust
@pytest.mark.parametrize("change", ["slot", "dict_value"])
def test_attribute_changes_during_walk_match_python(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    original_slot = cast("object", ChangingSlot.__dict__["b"])

    def replaced_b(value: ChangingSlot) -> int:
        del value
        return 99

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        if change == "slot":
            monkeypatch.setattr(ChangingSlot, "b", original_slot)
            target: ChangingSlot | ChangingDict = ChangingSlot(Tensor([]), 2)
        else:
            target = ChangingDict(Tensor([]), 2)

        def replace(value: Tensor) -> str:
            del value
            if change == "slot":
                monkeypatch.setattr(ChangingSlot, "b", property(replaced_b))
            else:
                target.b = 99
            return "triggered"

        return function(target, hooks={Tensor: (replace, decoded)})

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("dict[str, object]", expected)["b"] == 2


@needs_rust
def test_slot_changes_between_objects_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_slot = cast("object", ChangingSlot.__dict__["b"])

    def replaced_b(value: ChangingSlot) -> int:
        del value
        return 99

    def replace(value: Tensor) -> str:
        del value
        monkeypatch.setattr(ChangingSlot, "b", property(replaced_b))
        return "triggered"

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        monkeypatch.setattr(ChangingSlot, "b", original_slot)
        return function(
            [ChangingSlot(0, 2), Tensor([]), ChangingSlot(0, 3)],
            hooks={Tensor: (replace, decoded)},
        )

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("dict[str, object]", cast("list[object]", expected)[2])["b"] == 99


@needs_rust
def test_foreign_slot_descriptor_raises_like_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = ForeignSlotOwner(11, 22)
    monkeypatch.setattr(
        ForeignSlotOwner,
        "x",
        cast("MemberDescriptorType", vars(ForeignSlotSource)["b"]),
    )

    with pytest.raises(TypeError) as expected:
        serialize._python_to_plain(value)
    with pytest.raises(TypeError) as actual:
        serialize.to_plain(value)
    assert str(actual.value) == str(expected.value)


class _HidesHash(type):
    """Reports ``__hash__`` as ``None`` while the class's hash slot still hashes."""

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__hash__":
            return None
        return cast("object", super().__getattribute__(name))


class _ReportsHash(type):
    """Reports a ``__hash__`` for a class whose hash slot refuses hashing."""

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__hash__":
            return object.__hash__
        return cast("object", super().__getattribute__(name))


class HashHidden(metaclass=_HidesHash):
    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (HashHidden, (self.n,))


class HashReported(metaclass=_ReportsHash):
    __hash__ = None  # pyright: ignore[reportAssignmentType] -- An unhashable class is the case under test.

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (HashReported, (self.n,))


@pytest.mark.parametrize("kind", [HashHidden, HashReported])
def test_hashability_is_judged_by_the_class_attribute_as_python_does(
    kind: type[HashHidden | HashReported],
) -> None:
    # Python's ``is_mutable`` reads ``type(value).__hash__``, which these
    # metaclasses make disagree with the hash slot; it decides whether a
    # repeat is a ``py/ref`` or written again.
    value = kind(1)
    tree = [value, value]
    assert serialize.to_plain(tree) == serialize._python_to_plain(tree)


@pytest.mark.parametrize(
    ("value", "hooks"),
    [(case[1], case[3]) for case in ENCODE_CASES],
    ids=[case[0] for case in ENCODE_CASES],
)
def test_from_plain_reads_what_to_plain_wrote(
    value: object,
    hooks: Hooks | None,
) -> None:
    try:
        tree = serialize._python_to_plain(value, hooks=hooks or {})
    except (TypeError, ValueError):
        pytest.skip("the value does not encode")

    def read(function: Callable[..., object]) -> Callable[[], object]:
        def call() -> object:
            decoded = function(tree, object, hooks=hooks or {}, allow_imports=True)
            return serialize._python_to_plain(decoded, hooks=hooks or {})

        return call

    assert _outcome(read(deserialize.from_plain)) == _outcome(
        read(deserialize._python_from_plain),
    )


@pytest.mark.parametrize(
    ("data", "target", "options"),
    [case[1:] for case in DECODE_CASES],
    ids=[case[0] for case in DECODE_CASES],
)
def test_typed_from_plain_matches_python(
    data: object,
    target: object,
    options: dict[str, object],
) -> None:
    _assert_reads_alike(data, target, **options)


def _assert_reads_alike(data: object, target: object, **options: object) -> None:
    def read(function: Callable[..., object]) -> Callable[[], object]:
        return lambda: serialize._python_to_plain(function(data, target, **options))

    assert _outcome(read(deserialize.from_plain)) == _outcome(
        read(deserialize._python_from_plain),
    )


def _class(module: str, name: str, **annotations: object) -> type[object]:
    return type(name, (), {"__module__": module, "__annotations__": annotations})


def test_a_class_made_a_dataclass_after_a_read_reads_as_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A module-level owner has its field types cached, so what is learned
    # about them is kept across calls too, and must notice the change.
    module = ModuleType("treekle_parity_changed")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    inner = _class(module.__name__, "Inner", x=int)
    outer = dataclass(_class(module.__name__, "Outer", inner=inner))
    vars(module).update(Inner=inner, Outer=outer)
    data = {"inner": {"x": 1}}

    _assert_reads_alike(data, outer)
    _assert_reads_alike(data["inner"], inner)
    assert outer in deserialize._FIELD_TYPES
    dataclass(inner)

    _assert_reads_alike(data, outer)
    _assert_reads_alike(data["inner"], inner)


def test_cached_field_types_are_read_before_the_class_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Python's ``_field_types`` answers from ``_FIELD_TYPES`` first, so an
    # entry already there decides the read, whatever the class declares.
    # Compiled apart from this file, whose postponed annotations ``exec``
    # would inherit: a hint left a string is never read natively.
    module = ModuleType("treekle_parity_cached_types")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = (
        "import dataclasses\n"
        "@dataclasses.dataclass(slots=True)\n"
        "class Cached:\n"
        "    x: int = 0\n"
    )
    exec(  # noqa: S102 -- Fixed source, for evaluated annotations.
        compile(source, module.__name__, "exec", dont_inherit=True),
        vars(module),
    )
    cached = cast("type[Point]", vars(module)["Cached"])
    deserialize._FIELD_TYPES[cached] = MappingProxyType({"x": str})

    assert deserialize.from_plain({"x": "text"}, cached).x == "text"
    assert deserialize._python_from_plain({"x": "text"}, cached).x == "text"
    _assert_reads_alike({"x": 1}, cached)


def test_a_renamed_dataclass_reads_by_its_new_name() -> None:
    point = dataclass(_class(__name__, "Before", x=int))
    data = {"py/object": f"{__name__}.After", "x": 1}

    _assert_reads_alike(data, point, allow_imports=True)
    point.__qualname__ = "After"

    _assert_reads_alike(data, point, allow_imports=True)


@needs_rust
def test_mutated_dataclass_field_facts_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tagged = {"py/object": f"{ChangingField.__module__}.{ChangingField.__qualname__}"}
    _assert_reads_alike(tagged, ChangingField)
    field = cast("Field[object]", ChangingField.__dataclass_fields__["x"])
    monkeypatch.setattr(field, "default", 2)
    _assert_reads_alike(tagged, ChangingField)
    assert deserialize.from_plain(tagged, ChangingField).x == 2

    untagged = {"x": 3}
    _assert_reads_alike(untagged, ChangingField)
    monkeypatch.setattr(field, "init", False)
    _assert_reads_alike(untagged, ChangingField)
    assert deserialize.from_plain(untagged, ChangingField).x == 3
    monkeypatch.setattr(field, "default", MISSING)
    monkeypatch.setattr(field, "default_factory", lambda: 4)
    _assert_reads_alike(tagged, ChangingField)
    assert deserialize.from_plain(tagged, ChangingField).x == 4


def test_a_field_named_like_the_object_tag_reads_as_the_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("treekle_parity_tag_field")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    named = dataclass(init=False, eq=False)(
        _class(module.__name__, "Named", **{"py/object": int}),
    )
    vars(module).update(Named=named)
    data = {"py/object": f"{module.__name__}.Named"}
    _assert_reads_alike(data, named)
    _assert_reads_alike(data, named, allow_imports=True)


class _LoggedName(str):
    """A ``py/object`` path that logs each time it is compared."""

    __slots__ = ()
    log: ClassVar[list[str]] = []

    @override
    def __eq__(self, other: object) -> bool:
        _LoggedName.log.append("eq")
        return str.__eq__(self, other)

    __hash__ = str.__hash__


class _LoggedRecord(dict[str, object]):
    """A record that logs each lookup and item read."""

    log: ClassVar[list[str]] = []

    @override
    def items(self) -> dict_items[str, object]:
        _LoggedRecord.log.append("items")
        return super().items()

    @override
    def __contains__(self, key: object) -> bool:
        _LoggedRecord.log.append(f"contains {key}")
        return super().__contains__(key)


class _LoggedFlag:
    """An ``allow_imports`` that logs each time it is asked."""

    log: ClassVar[list[str]] = []

    def __bool__(self) -> bool:
        _LoggedFlag.log.append("bool")
        return True


@needs_rust
def test_a_stored_record_runs_the_same_user_code_in_both() -> None:
    name = f"{Point.__module__}.{Point.__qualname__}"
    logs: list[tuple[list[str], ...]] = []
    for function in (deserialize.from_plain, deserialize._python_from_plain):
        _LoggedName.log.clear()
        _LoggedFlag.log.clear()
        _LoggedRecord.log.clear()
        function({"py/object": _LoggedName(name), "x": 2}, Point, allow_imports=True)
        flag = cast("bool", cast("object", _LoggedFlag()))
        function({"py/object": name, "x": 2}, Point, allow_imports=flag)
        record = _LoggedRecord({"py/object": name, "x": 2})
        function(record, Point, allow_imports=True)
        logs.append((_LoggedName.log[:], _LoggedFlag.log[:], _LoggedRecord.log[:]))
    assert logs[0] == logs[1]
    assert all(logs[1])


def test_a_class_variable_does_not_count_as_a_stored_field() -> None:
    # ``__dataclass_fields__`` holds the ``ClassVar`` too, so a record with
    # one key per entry there still lacks ``y``; one naming it is refused.
    path = f"{WithClassVar.__module__}.{WithClassVar.__qualname__}"
    _assert_reads_alike({"py/object": path, "x": 1}, WithClassVar)
    assert deserialize.from_plain({"py/object": path, "x": 1}, WithClassVar).y == 7
    _assert_reads_alike({"py/object": path, "x": 1, "shared": 2}, WithClassVar)


def test_deep_nesting_round_trips_without_recursing() -> None:
    deep = _nested(20_000)
    encoded = serialize.to_plain(deep)
    decoded = deserialize.from_plain(encoded, object)

    for tree in (encoded, decoded):
        cursor = tree
        for _ in range(20_000):
            assert isinstance(cursor, list)
            children = cast("list[object]", cursor)
            assert len(children) == 1
            cursor = children[0]
        assert cursor == []


@needs_rust
def test_nesting_deeper_than_the_old_python_limit_matches_python() -> None:
    value = _nested(900)
    expected = serialize._python_to_plain(value)
    actual = serialize.to_plain(value)
    for tree in (expected, actual):
        cursor = tree
        for _ in range(900):
            assert isinstance(cursor, list)
            children = cast("list[object]", cursor)
            assert len(children) == 1
            cursor = children[0]
        assert cursor == []
    assert deserialize.from_plain(actual, object) is actual


def test_shallow_nesting_still_round_trips() -> None:
    value = _nested(200)
    assert serialize.to_plain(value) == serialize._python_to_plain(value)
    assert deserialize.from_plain(value, object) == value


def _deep(value: object, depth: int) -> object:
    """Return ``value`` nested ``depth`` lists down."""
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_references_deep_in_the_tree_match_python(mutable: bool) -> None:
    shared = {"value": 1}
    first_seen_deep = {"value": 2}
    deep = _deep({"earlier": shared, "new": first_seen_deep}, 5_000)
    value = {"first": shared, "deep": deep, "later": first_seen_deep}

    expected = serialize._python_to_plain(value, mutable=mutable)
    actual = serialize.to_plain(value, mutable=mutable)
    _assert_same_shape(actual, expected)
    _assert_same_shape(
        deserialize.from_plain(actual, object),
        deserialize._python_from_plain(expected, object),
    )


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_many_deep_subtrees_match_python(mutable: bool) -> None:
    shared = [0]
    value = _deep([[shared, [index]] for index in range(500)], 47)

    expected = serialize._python_to_plain(value, mutable=mutable)
    assert _shape(serialize.to_plain(value, mutable=mutable)) == _shape(expected)


class _Recipe:
    """Reduces to a recipe whose state holds whatever it was given."""

    def __init__(self, state: object) -> None:
        self.state = state

    @override
    def __reduce_ex__(self, protocol: object) -> tuple[object, ...]:
        del protocol
        return (_Recipe, (None,), {"state": self.state})


def _every_native_kind(depth: int) -> object:
    """Return a tree that nests each kind the native walk encodes ``depth`` deep."""
    shared = Point(1, 2.0)
    wrappers: list[Callable[[object, int], object]] = [
        lambda tree, level: {"plain": tree, "py/tagged": level, 3: shared},
        lambda tree, level: (tree, Color.RED, Level.LOW, Frozen(level)),
        lambda tree, level: [tree, {frozenset({level, "x"}), Frozen(level), "y"}],
        lambda tree, _: _Recipe(tree),
        lambda tree, level: Node(f"n{level}", [Node("leaf")], cast("Node", tree)),
        lambda tree, _: MappingProxyType({"m": tree, "f": math.inf, "s": shared}),
    ]
    tree: object = shared
    for level in range(depth):
        tree = wrappers[level % len(wrappers)](tree, level)
    return [shared, tree]


@needs_rust
@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_deep_trees_of_every_kind_encode_natively_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    mutable: bool,
) -> None:
    # Python's walk is what the native one hands a subtree to when it cannot
    # encode it itself; none of this needs that, however deep.
    value = _every_native_kind(3_000)
    expected = serialize._python_to_plain(value, mutable=mutable)

    def refused(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the native walk ran Python's")

    monkeypatch.setattr(serialize, "_encode", refused)
    monkeypatch.setattr(serialize, "_Encoding", refused)
    _assert_same_shape(serialize.to_plain(value, mutable=mutable), expected)


@needs_rust
def test_each_python_encoder_shares_one_reference_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh map per Python encoder would copy every registration so far each
    # time, which is quadratic in how many values one encodes.
    received: list[object] = []

    class Holder:
        def __init__(self, held: object) -> None:
            self.held = held

    def encode_holder(value: object, state: serialize._Encoding) -> object:
        received.append(state.seen)
        serialize._register(value, state)
        node: dict[str, object] = {"held": cast("Holder", value).held}
        serialize._push_values(node, state)
        return node

    monkeypatch.setitem(serialize._ENCODER_CACHE, Holder, encode_holder)
    shared = [0]
    value = [[Holder(index), shared, Holder(shared)] for index in range(50)]
    expected = serialize._python_to_plain(value)
    received.clear()

    assert serialize.to_plain(value) == expected
    assert len(received) == 100
    assert all(seen is received[0] for seen in received)


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_rolled_back_registrations_deep_in_the_tree_match_python(
    mutable: bool,
) -> None:
    # A set member and an escaped key are encoded once for their text, and
    # what that registers is forgotten.
    inner = Frozen(1)
    deep: object = inner
    for _ in range(60):
        deep = (deep,)
    value = [{deep}, {deep: 1}, inner, [inner]]

    expected = serialize._python_to_plain(value, mutable=mutable)
    assert _shape(serialize.to_plain(value, mutable=mutable)) == _shape(expected)


@dataclass(eq=False, kw_only=True, slots=True)
class KeyBox:
    """A key hashed by identity, holding whatever it is given."""

    held: object


def _key_cases() -> dict[str, object]:
    point = Frozen(1)
    looped = KeyBox(held=None)
    looped.held = [looped]
    return {
        "repeat": {(point, point): 1},
        "earlier": {"first": [point], (point, point): 1},
        "nested": {KeyBox(held={(point, point): 1, "p": point}): point},
        "cycle": {looped: 1},
    }


@pytest.mark.parametrize("case", list(_key_cases()))
def test_repeats_inside_an_escaped_key_match_python(case: str) -> None:
    value = _key_cases()[case]

    expected = _outcome(lambda: serialize._python_to_plain(value))
    assert _outcome(lambda: serialize.to_plain(value)) == expected


@needs_rust
def test_a_python_encoder_inside_a_key_sees_it_is_inside_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Python walk the Rust one calls is placed under the same sentinel, so
    # a repeat on either side of the boundary is written in full.
    def encode_box(value: object, state: serialize._Encoding) -> object:
        serialize._register(value, state)
        node: dict[str, object] = {"held": cast("KeyBox", value).held}
        serialize._push_values(node, state)
        return node

    monkeypatch.setitem(serialize._ENCODER_CACHE, KeyBox, encode_box)
    point = Frozen(1)
    box = KeyBox(held=point)
    value = {(point, box, box): 1}

    expected = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == expected
    assert isinstance(expected, dict)
    assert "py/ref" not in next(iter(expected))


@dataclass(kw_only=True, slots=True)
class RegisteredUnhashable:
    """Unhashable, as an ``eq`` dataclass is, yet registered as ``Hashable``."""


Hashable.register(RegisteredUnhashable)


@pytest.mark.parametrize(
    "key",
    [
        'json://{"py/tuple": [[1]]}',
        f'json://{{"py/object": "{__name__}.RegisteredUnhashable"}}',
    ],
)
def test_an_unhashable_escaped_key_matches_python(key: str) -> None:
    def read(function: Callable[..., object]) -> object:
        return function({key: 1}, dict[object, int], allow_imports=True)

    expected = _outcome(lambda: read(deserialize._python_from_plain))
    assert cast("tuple[object, ...]", expected)[:2] == ("raise", "ReadError")
    assert _outcome(lambda: read(deserialize.from_plain)) == expected


@needs_rust
@pytest.mark.parametrize("change", ["class", "module"])
def test_enum_reduce_changed_between_calls_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    # What a walk learns about a class is kept for the next call; changing
    # the class or the module function it stands in for must be seen.
    value = [Color.RED, Color.BLUE]
    before = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == before

    def custom(self: Color, protocol: object) -> tuple[object, ...]:
        del protocol
        return (Color, (self.value,), {"note": "custom"})

    if change == "class":
        monkeypatch.setattr(Color, "__reduce_ex__", custom, raising=False)
    else:
        original = serialize._canonical_reduce

        def canonical(value: object) -> object:
            if value is Color.RED:
                return (Color, ("red",), {"note": "module"})
            return original(value)

        monkeypatch.setattr(serialize, "_canonical_reduce", canonical)
    expected = serialize._python_to_plain(value)
    assert expected != before
    assert serialize.to_plain(value) == expected


@needs_rust
def test_list_changes_during_walk_match_python() -> None:
    def run(function: Callable[..., object]) -> object:
        value: list[object] = []

        def grow(tensor: Tensor) -> str:
            del tensor
            value.append("added")
            return "triggered"

        value.extend([Tensor([]), "tail"])
        return function(value, hooks={Tensor: (grow, Tensor)})

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[object]", expected)[1:] == ["tail"]


@needs_rust
def test_a_hooked_value_deeper_than_the_stack_is_judged_without_recursing() -> None:
    # The data is shallow, so the native decoder reads it; only the value the
    # hook returns is deep, and judging it against ``Tree`` once crashed.
    printed = _run_python(
        """
        import treekle.deserialize as deserialize

        class Deep:
            pass

        def rebuild(payload):
            value = 0
            for _ in range(100_000):
                value = [value]
            return value

        type Tree = list[Tree] | int
        hooks = {Deep: (lambda value: 0, rebuild)}
        data = {"py/hook": ["__main__.Deep", 0]}
        value = deserialize.from_plain(data, Tree, hooks=hooks)
        depth = 0
        while isinstance(value, list):
            value = value[0]
            depth += 1
        print(depth, value)
        """,
    )
    assert printed == "100000 0"


type _Tree = list[_Tree] | int


def _refuse_python_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if Python's decoding walk runs."""

    def refused(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the native walk ran Python's")

    monkeypatch.setattr(deserialize, "_decode", refused)
    monkeypatch.setattr(deserialize, "_Decoding", refused)


def _deep_read(call: Callable[[], object]) -> object:
    """Return a read's value as Python encodes it, else its ``_failure``."""
    try:
        return ("ok", serialize._python_to_plain(call()))
    except ReadError as error:
        return ("raise", _failure(error))


def _cycle_deep_down(depth: int) -> list[object]:
    """Return a tagged list chain whose deepest list holds its top."""
    top: list[object] = [{"py/float": "inf"}]
    cursor = top
    for _ in range(depth):
        child: list[object] = []
        cursor.append(child)
        cursor = child
    cursor.append(top)
    return top


def _bad_leaf_chain(depth: int) -> dict[str, object]:
    """Return ``_plain_chain(depth)`` whose deepest name is not a str."""
    chain = _plain_chain(depth)
    cursor = chain
    while "parent" in cursor:
        cursor = cast("dict[str, object]", cursor["parent"])
    cursor["name"] = 5
    return chain


_DEEP_READS: dict[str, tuple[Callable[[], object], object, dict[str, object]]] = {
    "union of lists": (lambda: _deep(0, 5_000), _Tree, {}),
    "union refusing its deepest leaf": (lambda: _deep("x", 5_000), _Tree, {}),
    "dataclass chain": (lambda: _plain_chain(5_000), Node, {}),
    "dataclass chain with a bad deepest field": (
        lambda: _bad_leaf_chain(5_000),
        Node,
        {},
    ),
    "every tagged kind": (
        lambda: serialize._python_to_plain(_every_native_kind(3_000)),
        object,
        {"allow_imports": True},
    ),
    "every tagged kind, frozen": (
        lambda: serialize._python_to_plain(_every_native_kind(3_000), mutable=False),
        object,
        {"allow_imports": True},
    ),
    "a cycle deep down": (lambda: _cycle_deep_down(3_000), object, {}),
}


@needs_rust
@pytest.mark.parametrize(
    ("build", "target", "options"),
    list(_DEEP_READS.values()),
    ids=list(_DEEP_READS),
)
def test_deep_reads_decode_natively_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], object],
    target: object,
    options: dict[str, object],
) -> None:
    # The native decoder once handed any tree past 128 levels to Python's walk.
    data = build()

    def read(function: Callable[..., object]) -> object:
        return _deep_read(lambda: function(data, target, **options))

    expected = read(deserialize._python_from_plain)
    _refuse_python_decoding(monkeypatch)
    _assert_same_shape(read(deserialize.from_plain), expected)


def test_a_cycle_is_refused_after_its_container_is_numbered() -> None:
    # Python refuses a cycle once the container's frame exists, so the dict
    # that closes it already holds a ``py/id`` number, and later numbers count it.
    cycle: dict[str, object] = {"n": 1}
    cycle["self"] = cycle
    data = {
        "x": cycle,
        "later": [{"k": 1}],
        "ref": {"py/id": 3},
        "tag": {"py/float": "inf"},
    }
    expected = _deep_read(lambda: deserialize._python_from_plain(data, object))
    assert _deep_read(lambda: deserialize.from_plain(data, object)) == expected
    with pytest.raises(ReadError) as caught:
        deserialize.from_plain(data, object)
    partial = cast("dict[str, object]", caught.value.partial)
    assert partial["ref"] is partial["later"]


@needs_rust
def test_a_nested_union_reads_its_innermost_node_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rust = _core()

    depth = 12
    data: object = "bad"
    for _ in range(depth):
        data = [data]
    original = deserialize._decode_uuid
    calls: list[object] = []

    def counted(data: PlainTree, target: object, at: object, state: object) -> object:
        calls.append(data)
        return original(data, target, at, state)  # pyright: ignore[reportArgumentType] -- Forwards the spied decoder's arguments unchanged.  # ty: ignore[invalid-argument-type] -- Same.

    counts: list[int] = []
    try:
        monkeypatch.setattr(deserialize, "_decode_uuid", counted)
        monkeypatch.setattr(
            deserialize,
            "_DECODERS",
            MappingProxyType({**deserialize._DECODERS, UUID: counted}),
        )
        _rust.setup_decode(deserialize.__dict__)
        for function in (deserialize.from_plain, deserialize._python_from_plain):
            calls.clear()
            with pytest.raises(ReadError):
                function(data, _Retried)
            counts.append(len(calls))
    finally:
        monkeypatch.undo()
        _rust.setup_decode(deserialize.__dict__)
    # Each union tries ``UUID`` once on its own node: linear, not 2 ** depth.
    assert counts[0] == counts[1] == depth + 1


@needs_rust
def test_repeated_top_level_union_reuses_its_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rust = _core()

    original = deserialize._union_candidates
    calls = 0

    def counted(data: PlainTree, members: tuple[object, ...]) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return original(data, members)

    monkeypatch.setattr(deserialize, "_union_candidates", counted)
    try:
        _rust.setup_decode(deserialize.__dict__)
        target = int | None
        # Before 3.14, ``X | Y`` cannot be weakly referenced, so its plan
        # lasts one call rather than across calls.
        per_call = 1 if sys.version_info >= (3, 14) else 3
        for _ in range(3):
            assert deserialize.from_plain(3, target) == 3
        # Each kind of data asks for its member order once, on first arrival.
        assert calls == per_call
        for _ in range(3):
            assert deserialize.from_plain(None, target) is None
        assert calls == 2 * per_call

        # Within one call, every container sharing a union shares its plan.
        calls = 0
        nested = [["a", None] for _ in range(50)]
        assert deserialize.from_plain(nested, list[list[str | None]]) == nested
        assert calls == 2
    finally:
        monkeypatch.setattr(deserialize, "_union_candidates", original)
        _rust.setup_decode(deserialize.__dict__)


@needs_rust
def test_nonclass_type_plan_does_not_keep_its_target_alive() -> None:
    target = list[int]
    weak = weakref.ref(target)
    assert deserialize.from_plain([1], target) == [1]
    del target
    gc.collect()
    assert weak() is None


@needs_rust
def test_a_dead_targets_plan_leaves_the_cache() -> None:
    # ``list[int]`` is a new alias each time it is written, so a caller that
    # writes it in place plans a new target on every call. The cache kept each
    # dead target's entry until it filled, so memory grew with the number of
    # calls rather than with the targets still alive.
    _core().setup_decode(deserialize.__dict__)
    targets = [list[int] for _ in range(500)]
    gc.collect()
    tracemalloc.start()
    try:
        while targets:
            assert deserialize.from_plain([1], targets.pop()) == [1]
        gc.collect()
        kept, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert kept < 4_096


@needs_rust
def test_cached_field_plan_does_not_keep_its_hint_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("treekle_parity_short_lived_hint")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    hint = list[int]
    weak = weakref.ref(hint)
    owner = dataclass(_class(module.__name__, "Owner", value=hint))
    vars(module)["Owner"] = owner

    assert vars(deserialize.from_plain({"value": [1]}, owner))["value"] == [1]
    assert owner in deserialize._FIELD_TYPES
    del vars(module)["Owner"]
    del owner, hint
    gc.collect()
    assert weak() is None


def _nested_dataclass(module: ModuleType, **annotations: object) -> type[object]:
    inner = _class(module.__name__, "Inner", **annotations)
    inner.__qualname__ = "Outer.Inner"
    dataclass(inner)
    vars(module)["Outer"] = type(
        "Outer",
        (),
        {"__module__": module.__name__, "Inner": inner},
    )
    return inner


@needs_rust
def test_a_nested_dataclass_keeps_its_field_types_across_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rust = _core()

    module = ModuleType("treekle_parity_nested")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    inner = _nested_dataclass(module, x=int)
    original = deserialize._field_types
    calls: list[object] = []

    def counted(kind: type) -> Mapping[str, object]:
        calls.append(kind)
        return original(kind)

    try:
        monkeypatch.setattr(deserialize, "_field_types", counted)
        _rust.setup_decode(deserialize.__dict__)
        assert vars(deserialize.from_plain({"x": 1}, inner)) == {"x": 1}
        first = len(calls)
        for _ in range(3):
            assert vars(deserialize.from_plain({"x": 2}, inner)) == {"x": 2}
        assert inner in deserialize._FIELD_TYPES
        assert len(calls) == first
        monkeypatch.setattr(inner, "changed", True, raising=False)
        assert vars(deserialize.from_plain({"x": 3}, inner)) == {"x": 3}
        assert len(calls) > first
    finally:
        monkeypatch.undo()
        _rust.setup_decode(deserialize.__dict__)


@needs_rust
def test_an_unresolved_nested_dataclass_resolves_its_hints_once_per_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unresolved hint is resolved afresh on each read, since its import may
    # only be pending: on the first read with the whole class, then alone; on
    # later reads alone. The native read once added a whole-class try of its
    # own to learn whether to keep a nested class's facts.
    evaluated: list[str] = []

    class Pending:
        def __getattr__(self, name: str) -> object:
            evaluated.append(name)
            raise AttributeError(name)

    module = ModuleType("treekle_parity_pending")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vars(module)["pending"] = Pending()
    inner = _nested_dataclass(module, x=int, later="pending.Later")
    data = {"x": 1, "later": [1]}

    counts: list[int] = []
    for read in (deserialize._python_from_plain, deserialize.from_plain):
        deserialize._PARTIAL_HINTS.pop(inner, None)
        evaluated.clear()
        for _ in range(3):
            read(data, inner)
        counts.append(len(evaluated))
    assert counts == [4, 4]
    _assert_reads_alike(data, inner)

    vars(module)["pending"] = SimpleNamespace(Later=tuple[int, ...])
    _assert_reads_alike(data, inner)
    assert vars(deserialize.from_plain(data, inner))["later"] == (1,)


@needs_rust
def test_a_kept_nested_dataclass_can_still_be_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("treekle_parity_short_lived_nested")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    inner = _nested_dataclass(module, x=int)
    weak = weakref.ref(inner)

    assert vars(deserialize.from_plain({"x": 1}, inner)) == {"x": 1}
    del vars(module)["Outer"]
    del inner
    gc.collect()
    assert weak() is None


def test_a_nested_dataclass_does_not_see_a_rebound_hint_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A nested class's field types are kept as a module-level class's are, so
    # rebinding a name a string annotation uses is not seen by either read.
    module = ModuleType("treekle_parity_rebound")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    vars(module)["Alias"] = tuple[int, ...]
    inner = _nested_dataclass(module, x="Alias")
    data = {"x": [1, 2]}

    _assert_reads_alike(data, inner)
    vars(module)["Alias"] = list[int]

    _assert_reads_alike(data, inner)
    assert vars(deserialize._python_from_plain(data, inner))["x"] == (1, 2)


class _Areal(Protocol):
    def area(self) -> float: ...


@dataclass(kw_only=True, slots=True)
class _Plot:
    side: float = 1.0

    def area(self) -> float:
        return self.side**2


class _NoArea:
    @property
    def area(self) -> float:
        msg = "no area"
        raise ValueError(msg)


_PLOT = _Plot()


@pytest.mark.parametrize(
    "tree",
    [
        serialize._python_to_plain([_PLOT, _PLOT]),
        serialize._python_to_plain([Color.RED]),
        serialize._python_to_plain([_NoArea()]),
        [1, {"py/ref": []}],
    ],
    ids=["fits", "lacks a member", "member raises", "reference lacks a member"],
)
def test_a_protocol_target_judges_alike(tree: PlainTree) -> None:
    _assert_reads_alike(tree, list[_Areal], allow_imports=True)


def test_a_protocol_naming_a_non_str_member_judges_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_Areal, "__protocol_attrs__", frozenset({1}))

    _assert_reads_alike(
        serialize._python_to_plain([_PLOT]),
        list[_Areal],
        allow_imports=True,
    )


# The native walks keep what they learn about a class or a key across values
# and calls, so each case below reads or writes more than once: a wrong cached
# answer shows only after the first.
_TIMES = 3


def _assert_reads_alike_each_time(
    data: object,
    target: object,
    *,
    allow_imports: bool = False,
) -> None:
    """Assert ``from_plain`` and ``json_loads`` read ``data`` as Python does, call after call."""
    text = json.dumps(data)

    def python() -> object:
        tree = cast("PlainTree", json.loads(text))
        read = deserialize._python_from_plain(tree, target, allow_imports=allow_imports)
        return serialize._python_to_plain(read)

    def fused() -> object:
        read = wire.json_loads(text, target, allow_imports=allow_imports)
        return serialize._python_to_plain(read)

    for _ in range(_TIMES):
        _assert_reads_alike(data, target, allow_imports=allow_imports)
        assert _outcome(fused) == _outcome(python)


def _assert_writes_alike_each_time(value: object, *, sort: bool = True) -> None:
    """Assert ``to_plain`` and ``json_dumps`` write ``value`` as Python does, call after call."""
    tree = _outcome(lambda: serialize._python_to_plain(value, sort=sort))
    text = _outcome(lambda: json.dumps(serialize._python_to_plain(value, sort=sort)))
    for _ in range(_TIMES):
        assert _outcome(lambda: serialize.to_plain(value, sort=sort)) == tree
        assert _outcome(lambda: wire.json_dumps(value, sort=sort)) == text


# Any ``py/`` key makes its dict a tag (``_has_tag_key``), even under a target
# whose values are plain, so the dict is a reference, or an unknown tag's error.
@pytest.mark.parametrize(
    ("data", "target"),
    [
        ([{"a": 1}, {"py/id": 1}], list[dict[str, int]]),
        ([{"a": 1}, {"py/ref": [0]}], list[dict[str, int]]),
        ({"k": {"a": 1}, "j": {"py/id": 1}}, dict[str, dict[str, object]]),
        ([{"a": "x"}, {"py/x": "y"}], list[dict[str, str]]),
        ([{"a": "x"}, {"py/": "y"}], list[dict[str, object]]),
        ([{"a": "x"}, {"b": "y", "py/id": 1}], list[dict[str, object]]),
    ],
    ids=[
        "py/id",
        "py/ref",
        "py/id under str keys",
        "unknown tag",
        "bare prefix",
        "tag key last",
    ],
)
def test_a_tag_key_in_a_plain_dict_reads_as_a_tag(data: object, target: object) -> None:
    _assert_reads_alike_each_time(data, target)


class _MaybeArea:
    """Has ``area`` or not, by instance: its property raises ``AttributeError`` when off."""

    def __init__(self, *, has: bool) -> None:
        self.has = has

    # The error names the value; its default repr holds an address.
    @override
    def __repr__(self) -> str:
        return f"_MaybeArea(has={self.has})"

    @property
    def area(self) -> float:
        if self.has:
            return 1.0
        raise AttributeError("area")


# ``hasattr`` runs the property for each value, so one value's answer says
# nothing about the next one of its class.
@pytest.mark.parametrize(
    "order",
    [[True, False], [False, True], [True, True, False], [False, False, True]],
    ids=[
        "fits, then not",
        "not, then fits",
        "fits twice, then not",
        "not twice, then fits",
    ],
)
def test_a_protocol_member_answering_per_value_judges_each_value(
    order: list[bool],
) -> None:
    for has in order:
        tree = serialize._python_to_plain([_MaybeArea(has=has)])
        _assert_reads_alike_each_time(tree, list[_Areal], allow_imports=True)
    mixed = serialize._python_to_plain([_MaybeArea(has=has) for has in order])
    _assert_reads_alike_each_time(mixed, list[_Areal], allow_imports=True)


@dataclass(kw_only=True, slots=True)
class _DictFromGetattr:
    """A slotted class with no ``__dict__`` slot whose ``__getattr__`` supplies one."""

    a: int = 1

    def __getattr__(self, name: str) -> object:
        if name == "__dict__":
            return {"extra": 2}
        if name == "extra":
            return 2
        raise AttributeError(name)


class _SlotsDictFromGetattr:
    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a = 1

    def __getattr__(self, name: str) -> object:
        if name == "__dict__":
            return {"extra": 2}
        raise AttributeError(name)


# ``attribute_names`` asks ``getattr(value, "__dict__", None)``; with no dict
# storage that reaches ``__getattr__``, whose answer joins the attributes.
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize(
    "kind",
    [_DictFromGetattr, _SlotsDictFromGetattr],
    ids=["slots dataclass", "slotted class"],
)
def test_a_dict_from_getattr_writes_as_python_does(
    kind: Callable[[], object],
    sort: bool,
) -> None:
    _assert_writes_alike_each_time(kind(), sort=sort)


class _ReduceBase:
    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a = 1


class _DictReduceBase:
    def __init__(self) -> None:
        self.a = 1


def _reduce_property(error: type[Exception] | None) -> property:
    """Return a ``__reduce__`` property that raises ``error``, or gives a recipe."""

    def reduce(self: object) -> object:
        if error is None:
            return lambda: (type(self), ())
        raise error("no reduce")

    return property(reduce)


# Built with ``type`` because a checker rejects a property overriding a method.
def _reduce_class(
    name: str,
    error: type[Exception] | None,
    *,
    slotted: bool,
) -> type[object]:
    reduce = _reduce_property(error)
    if slotted:
        return type(
            name,
            (_ReduceBase,),
            {"__module__": __name__, "__slots__": (), "__reduce__": reduce},
        )
    return type(
        name,
        (_DictReduceBase,),
        {"__module__": __name__, "__reduce__": reduce},
    )


SlottedReduceRaisesAttributeError = _reduce_class(
    "SlottedReduceRaisesAttributeError",
    AttributeError,
    slotted=True,
)
SlottedReduceRaisesTypeError = _reduce_class(
    "SlottedReduceRaisesTypeError",
    TypeError,
    slotted=True,
)
SlottedReduceRaisesKeyError = _reduce_class(
    "SlottedReduceRaisesKeyError",
    KeyError,
    slotted=True,
)
SlottedReduceFromProperty = _reduce_class(
    "SlottedReduceFromProperty",
    None,
    slotted=True,
)
DictReduceRaisesAttributeError = _reduce_class(
    "DictReduceRaisesAttributeError",
    AttributeError,
    slotted=False,
)


# ``object.__reduce_ex__`` reads ``__reduce__`` off the instance, so a property
# runs there: an ``AttributeError`` falls back to copyreg's recipe, other
# errors propagate, and what it returns is called for the recipe.
@pytest.mark.parametrize(
    "kind",
    [
        SlottedReduceRaisesAttributeError,
        SlottedReduceRaisesTypeError,
        SlottedReduceRaisesKeyError,
        SlottedReduceFromProperty,
        DictReduceRaisesAttributeError,
    ],
    ids=[
        "slotted, AttributeError",
        "slotted, TypeError",
        "slotted, KeyError",
        "slotted, gives a recipe",
        "dict-backed, AttributeError",
    ],
)
def test_a_reduce_property_writes_as_python_does(kind: Callable[[], object]) -> None:
    _assert_writes_alike_each_time(kind())


class _HidesReduce(type):
    """Answers ``__reduce__`` with ``object``'s, hiding the class's own."""

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__reduce__":
            return object.__reduce__
        return cast("object", super().__getattribute__(name))


class ReduceHiddenByMetaclass(metaclass=_HidesReduce):
    __slots__ = ()

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (ReduceHiddenByMetaclass, ())


# ``object.__reduce_ex__`` reads ``__reduce__`` through the metaclass, so one
# that hides the class's own leaves copyreg's recipe.
def test_a_reduce_the_metaclass_hides_writes_as_python_does() -> None:
    _assert_writes_alike_each_time(ReduceHiddenByMetaclass())


class _CountsModuleReads(type):
    """Counts each read of ``__module__``."""

    reads: ClassVar[int] = 0

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__module__":
            _CountsModuleReads.reads += 1
        return cast("object", super().__getattribute__(name))


class ReducesToAnUnknownName(metaclass=_CountsModuleReads):
    __slots__ = ()

    @override
    def __reduce__(self) -> str:
        return "unknown_name"


@pytest.mark.parametrize(
    "function",
    [serialize._python_to_plain, serialize.to_plain, wire.json_dumps],
    ids=["python", "to_plain", "json_dumps"],
)
def test_a_bare_name_reduce_reads_the_module_once_as_python_does(
    function: Callable[[object], object],
) -> None:
    for _ in range(_TIMES):
        _CountsModuleReads.reads = 0
        with pytest.raises(TypeError, match="does not resolve"):
            function(ReducesToAnUnknownName())
        assert _CountsModuleReads.reads == 1


@needs_rust
@pytest.mark.parametrize("slots", [False, True], ids=["dict-backed", "slotted"])
def test_a_member_that_drops_a_later_member_writes_as_python_does(
    slots: bool,
) -> None:
    # Python reads every member before encoding any, so the dropped list is
    # still written; read lazily, it would be gone, or freed, by its turn.
    printed = _run_python(
        f"""
        from dataclasses import dataclass
        import gc
        import json
        from treekle import serialize, wire

        class Drops:
            def __init__(self, owner):
                self.owner = owner

            def __reduce__(self):
                if self.owner is not None and hasattr(self.owner, "b"):
                    del self.owner.b
                    gc.collect()
                    [bytearray(64) for _ in range(1000)]
                return (Drops, (None,))

        @dataclass(slots={slots})
        class Owner:
            a: object
            b: object

        def built():
            owner = Owner(None, None)
            owner.a, owner.b = Drops(owner), ["only", ["held", "here"]]
            return [owner]

        tree = serialize._python_to_plain(built())
        text = json.dumps(tree)
        print(*(serialize.to_plain(built()) == tree for _ in range(3)))
        print(*(wire.json_dumps(built()) == text for _ in range(3)))
        """,
    )
    assert printed == "True True True\nTrue True True"


@needs_rust
def test_a_resolved_cache_replaced_many_times_is_still_read() -> None:
    # Each replacement is another dict for the reader to watch.
    printed = _run_python(
        """
        import os
        import treekle.deserialize as deserialize

        path = "collections.OrderedDict"
        kept = []
        for _ in range(6):
            deserialize._RESOLVED = {}
            kept.append(deserialize._RESOLVED)
            for _ in range(2):
                deserialize.from_plain({"py/type": path}, object, allow_imports=True)
        deserialize._RESOLVED[path] = os.PathLike
        print(deserialize.from_plain({"py/type": path}, object, allow_imports=True))
        """,
    )
    assert printed == "<class 'os.PathLike'>"


class _GcProbe:
    pass


# A hook may read or switch the cyclic collector; the walk leaves it alone.
@pytest.mark.parametrize(
    "enabled",
    [True, False],
    ids=["collector on", "collector off"],
)
def test_a_hook_sees_and_keeps_the_collector_state_as_python_does(
    enabled: bool,
) -> None:
    def encode(probe: _GcProbe) -> PlainTree:
        del probe
        seen = gc.isenabled()
        if enabled:
            gc.disable()
        else:
            gc.enable()
        return {"enabled": seen}

    def decode(tree: PlainTree) -> _GcProbe:
        del tree
        return _GcProbe()

    hooks: Hooks = {_GcProbe: (encode, decode)}
    writers: list[Callable[[], object]] = [
        lambda: serialize._python_to_plain(_GcProbe(), hooks=hooks),
        lambda: serialize.to_plain(_GcProbe(), hooks=hooks),
        lambda: cast("object", json.loads(wire.json_dumps(_GcProbe(), hooks=hooks))),
    ]
    was = gc.isenabled()
    try:
        seen: list[tuple[object, bool]] = []
        for write in writers:
            for _ in range(_TIMES):
                if enabled:
                    gc.enable()
                else:
                    gc.disable()
                seen.append((_outcome(write), gc.isenabled()))
    finally:
        if was:
            gc.enable()
        else:
            gc.disable()
    assert seen == [seen[0]] * len(seen)
    assert seen[0] == (
        ("ok", _shape({"py/hook": [f"{__name__}._GcProbe", {"enabled": enabled}]})),
        not enabled,
    )


# With the core, ``json_loads`` and ``json_dumps`` parse and write the text
# themselves and leave to ``json`` only what they would not answer exactly as
# it does. Each case below must give what ``json`` gives: the same value, down
# to float bits and container types, or the same error with the same text.


def _exact(tree: object) -> object:
    """Return ``tree`` as data that tells JSON values apart, ``-0.0`` from ``0.0`` too."""
    if isinstance(tree, dict):
        items = cast("dict[object, object]", tree).items()
        return ("dict", [(_exact(key), _exact(value)) for key, value in items])
    if isinstance(tree, list):
        return ("list", [_exact(item) for item in cast("list[object]", tree)])
    if isinstance(tree, float):
        return ("float", repr(tree))
    return (type(tree).__name__, tree)


def _text_outcome(call: Callable[[], object]) -> object:
    """Return the call's value as ``_exact`` shows it, or the type and text of what it raised."""
    try:
        return ("ok", _exact(call()))
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either side raises.
        return ("raise", type(error).__name__, str(error))


def _read_outcome(call: Callable[[], object]) -> object:
    """Return what the call read, as Python writes it and ``_exact`` shows it, or its error."""
    try:
        return ("ok", _exact(serialize._python_to_plain(call())))
    except ReadError as error:
        return ("raise", "ReadError", _exact(_failure(error)))
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either side raises.
        return ("raise", type(error).__name__, str(error))


def _parses_alike(text: object) -> bool:
    """Return whether ``json_loads(text)`` gives what ``json.loads(text)`` gives."""
    document = cast("str", text)
    return _text_outcome(lambda: wire.json_loads(document)) == _text_outcome(
        lambda: cast("object", json.loads(document)),
    )


_JSON_TEXTS: dict[str, list[object]] = {
    "structure": [
        "",
        " ",
        "{",
        "}",
        "[",
        "]",
        "[1,]",
        '{"a":1,}',
        '{"a" 1}',
        "{1:2}",
        "[1 2]",
        "{} {}",
        "{}  x",
        '{"a": [1, {"b": null}], "c": {}}',
        "[[], {}, [{}]]",
    ],
    "numbers": [
        "0",
        "-0",
        "-0.0",
        "01",
        "-",
        "1.",
        ".5",
        "1e",
        "1e+",
        "1E+2",
        "0e0",
        "+1",
        "0x10",
        "1e400",
        "-1e400",
        "1e-400",
        "4.9e-324",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "0.1",
        "999999999999999999",
        "-999999999999999999",
        "9223372036854775807",
        "9223372036854775808",
        "-9223372036854775808",
        "-9223372036854775809",
        "9999999999999999999",
        "18446744073709551616",
        "123456789012345678901234567890",
        "9" * 4300,
        "9" * 4301,
        "-" + "9" * 5000,
    ],
    "constants": [
        "NaN",
        "-NaN",
        "nan",
        "Infinity",
        "-Infinity",
        "infinity",
        "[NaN, Infinity, -Infinity]",
        "true",
        "false",
        "null",
        "True",
        "nul",
        "t",
    ],
    "strings": [
        '"a"',
        '"\\u00e9"',
        '"\\ud83d\\ude00"',
        '"\\ud83d"',
        '"\\ude00"',
        '"\\ud83dx"',
        '"\\ud83d\\u0041"',
        '"\\x41"',
        '"\\u12"',
        '"\\u12g4"',
        '"\\/\\b\\f\\n\\r\\t\\"\\\\"',
        '"a\nb"',
        '"a\tb"',
        '"a\x00b"',
        '"\x7f"',
        '"\u00e9"',
        '"\U0001f600"',
        '"\ud83d"',
        '"unterminated',
        '"\\',
    ],
    "framing": [
        "\ufeff{}",
        "{}\ufeff",
        "\x0c{}",
        "{}\x0c",
        "\u00a0{}",
        " \t\r\n{} \t\r\n",
    ],
    "keys": [
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1}, "a": [2]}',
        '{"": 0}',
        '{"\\u0061": 1, "a": 2}',
        '{"py/x": 1}',
        '{"json://1": 2}',
    ],
    "nesting": [
        "[" * 199 + "]" * 199,
        "[" * 200 + "]" * 200,
        "[" * 201 + "]" * 201,
        '{"a": ' * 300 + "1" + "}" * 300,
        "[" * 5_000 + "]" * 5_000,
    ],
    "bytes": [
        b"{}",
        b'{"a": "\xc3\xa9"}',
        b'"\xff"',
        b'"\xc3"',
        b'"\xed\xa0\xbd"',
        b"\xef\xbb\xbf{}",
        '{"a": 1}'.encode("utf-16"),
        '{"a": 1}'.encode("utf-16-le"),
        '{"a": 1}'.encode("utf-32"),
        bytearray(b"[1]"),
        memoryview(b"[1]"),
    ],
}


@pytest.mark.parametrize("group", list(_JSON_TEXTS))
def test_json_loads_parses_as_json_does(group: str) -> None:
    assert [text for text in _JSON_TEXTS[group] if not _parses_alike(text)] == []


@dataclass(kw_only=True, slots=True)
class _Row:
    id: int
    name: str
    score: float
    ok: bool
    tags: list[str]
    note: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class _FrozenRow:
    id: int
    score: float


@dataclass(
    kw_only=True,
)  # house-ignore[dataclass] -- The dict-backed row: its fields live in __dict__.
class _DictRow:
    id: int
    score: float


@dataclass(kw_only=True, slots=True)
class _Batch:
    rows: list[_Row]
    by_name: dict[str, _FrozenRow]
    pair: tuple[int, str]
    counts: dict[int, float]
    either: _DictRow | int


def _tag(name: str) -> str:
    return f'"py/object": "{__name__}.{name}"'


# Records carry their tags, as ``to_plain`` writes them: the typed reader
# takes only tagged records straight from the text.
_ROW_FIELDS = '"id": 1, "name": "a", "score": 0.5, "ok": true, "tags": ["x"]'
_ROW = f"{{{_tag('_Row')}, {_ROW_FIELDS}}}"
_FROZEN = f'{{{_tag("_FrozenRow")}, "id": 2, "score": -0.0}}'
_DICT_ROW = f'{{{_tag("_DictRow")}, "id": 4, "score": 1e400}}'
_BATCH = (
    f'{{{_tag("_Batch")}, "rows": [{_ROW}], "by_name": {{"a": {_FROZEN}}},'
    f' "pair": [3, "b"], "counts": {{"json://1": 0.5}}, "either": {_DICT_ROW}}}'
)

# (id, text, target, strict): documents the typed reader takes straight from
# the text, and ones where it must give up part way and leave them to the tree.
_TYPED_TEXTS: list[tuple[str, str, object, bool]] = [
    ("rows", f"[{_ROW}, {_ROW}]", list[_Row], True),
    ("untagged rows", f"[{{{_ROW_FIELDS}}}]", list[_Row], True),
    ("tag last", f"[{{{_ROW_FIELDS}, {_tag('_Row')}}}]", list[_Row], True),
    ("tag twice", f"[{_ROW[:-1]}, {_tag('_Row')}}}]", list[_Row], True),
    ("tag of another class", f"[{_FROZEN}]", list[_Row], True),
    ("tag that is not text", f'[{{"py/object": 1, {_ROW_FIELDS}}}]', list[_Row], True),
    ("one record", _ROW, _Row, True),
    ("empty list", "[]", list[_Row], True),
    (
        "default left out",
        f'[{{{_tag("_Row")}, "id": 1, "name": "a", "score": 1, "ok": false, "tags": []}}]',
        list[_Row],
        True,
    ),
    (
        "required field left out",
        f'[{{{_tag("_Row")}, "id": 1, "name": "a"}}]',
        list[_Row],
        True,
    ),
    ("extra key", f'[{_ROW[:-1]}, "extra": 1}}]', list[_Row], True),
    ("extra key, not strict", f'[{_ROW[:-1]}, "extra": 1}}]', list[_Row], False),
    ("duplicate key", f'[{_ROW[:-1]}, "id": 2}}]', list[_Row], True),
    ("escaped key", _ROW.replace('"id"', '"\\u0069d"'), _Row, True),
    ("escaped text", _ROW.replace('"a"', '"\\u00e9\\ud83d\\ude00\\n"'), _Row, True),
    ("lone surrogate text", _ROW.replace('"a"', '"\\ud800"'), _Row, True),
    ("float text in an int field", _ROW.replace('"id": 1', '"id": 1.0'), _Row, True),
    ("int text in a float field", _ROW.replace("0.5", "7"), _Row, True),
    (
        "huge int",
        _ROW.replace('"id": 1', '"id": 123456789012345678901234567890'),
        _Row,
        True,
    ),
    (
        "non-finite floats",
        f"[{_FROZEN.replace('-0.0', 'NaN')}, {_FROZEN.replace('-0.0', '-Infinity')}]",
        list[_FrozenRow],
        True,
    ),
    (
        "float extremes",
        f"[{_FROZEN.replace('-0.0', '5e-324')}, {_FROZEN.replace('-0.0', '1e400')}]",
        list[_FrozenRow],
        True,
    ),
    ("int text in a float", _FROZEN.replace("-0.0", "-0"), _FrozenRow, True),
    ("float text in an int", _FROZEN.replace('"id": 2', '"id": 2.0'), _FrozenRow, True),
    ("text in an int", _FROZEN.replace('"id": 2', '"id": "2"'), _FrozenRow, False),
    ("bool in an int", _FROZEN.replace('"id": 2', '"id": true'), _FrozenRow, True),
    ("bool from 0", _ROW.replace("true", "0"), _Row, True),
    ("lax bool from text", _ROW.replace("true", '"yes"'), _Row, False),
    ("null for an optional", f'[{_ROW[:-1]}, "note": null}}]', list[_Row], True),
    ("null for a required field", _ROW.replace("0.5", "null"), _Row, True),
    (
        "bad record late in the list",
        "[" + ", ".join([_ROW] * 20 + [_ROW.replace('"id": 1', '"id": "x"')]) + "]",
        list[_Row],
        True,
    ),
    ("unclosed list", f"[{_ROW}, {_ROW}", list[_Row], True),
    ("trailing comma", f"[{_ROW},]", list[_Row], True),
    ("trailing text", f"[{_ROW}] x", list[_Row], True),
    (
        "space everywhere",
        f" \n[\t{_ROW.replace(': ', ' :  ')} ,\r\n{_ROW} ]\n ",
        list[_Row],
        True,
    ),
    ("tag key in a record", f'[{_ROW[:-1]}, "py/id": 1}}]', list[_Row], True),
    ("a reference to a record", f'[{_ROW}, {{"py/id": 1}}]', list[_Row], True),
    (
        "frozen rows by name",
        f'{{"a": {_FROZEN}, "b": {_FROZEN}}}',
        dict[str, _FrozenRow],
        True,
    ),
    ("dict-backed rows", f"[{_DICT_ROW}, {_DICT_ROW}]", list[_DictRow], True),
    ("tag key in a plain dict", '{"a": 1, "py/x": 2}', dict[str, int], True),
    (
        "reference in a plain dict",
        '[{"a": 1}, {"py/id": 1}]',
        list[dict[str, int]],
        True,
    ),
    ("records in records", _BATCH, _Batch, True),
    (
        "tuple as an envelope",
        _BATCH.replace('[3, "b"]', '{"py/tuple": [3, "b"]}'),
        _Batch,
        True,
    ),
    (
        "union member as an int",
        _BATCH.replace(_DICT_ROW, "5"),
        _Batch,
        True,
    ),
    (
        "escaped key not canonical",
        _BATCH.replace("json://1", "json://01"),
        _Batch,
        True,
    ),
    ("escaped key not an int", _BATCH.replace("json://1", "json://true"), _Batch, True),
]


@pytest.mark.parametrize(
    ("text", "target", "strict"),
    [case[1:] for case in _TYPED_TEXTS],
    ids=[case[0] for case in _TYPED_TEXTS],
)
def test_typed_json_loads_reads_as_from_plain_of_json_loads(
    text: str,
    target: object,
    strict: bool,
) -> None:
    def python() -> object:
        tree = cast("PlainTree", json.loads(text))
        return deserialize._python_from_plain(
            tree,
            target,
            strict=strict,
            allow_imports=True,
        )

    expected = _read_outcome(python)
    for _ in range(_TIMES):
        assert (
            _read_outcome(
                lambda: wire.json_loads(
                    text,
                    target,
                    strict=strict,
                    allow_imports=True,
                ),
            )
            == expected
        )


class _Layout(NamedTuple):
    sort: bool = True
    ensure_ascii: bool = True
    separators: tuple[str, str] | None = None
    indent: int | str | None = None
    sort_keys: bool = False


_LAYOUTS = {
    "default": _Layout(),
    "unsorted attributes": _Layout(sort=False),
    "compact": _Layout(separators=(",", ":")),
    "indented": _Layout(indent=2),
    "tab indent and sorted keys": _Layout(indent="\t", sort_keys=True),
    "non-ASCII kept": _Layout(ensure_ascii=False),
}


def _dumps_outcomes(
    value: object,
    *,
    dialect: object,
    hooks: Hooks | None,
    layout: _Layout,
) -> tuple[object, object]:
    """Return what ``json_dumps`` gives for ``value``, and ``json.dumps`` of Python's tree."""
    style = cast("Dialect", dialect)

    def python() -> str:
        tree = serialize._python_to_plain(
            value,
            dialect=style,
            hooks=hooks or {},
            sort=layout.sort,
        )
        return json.dumps(
            tree,
            ensure_ascii=layout.ensure_ascii,
            separators=layout.separators,
            indent=layout.indent,
            sort_keys=layout.sort_keys,
        )

    def written() -> str:
        return wire.json_dumps(
            value,
            dialect=style,
            hooks=hooks or {},
            sort=layout.sort,
            ensure_ascii=layout.ensure_ascii,
            separators=layout.separators,
            indent=layout.indent,
            sort_keys=layout.sort_keys,
        )

    return _text_outcome(written), _text_outcome(python)


@pytest.mark.parametrize("layout", list(_LAYOUTS))
@pytest.mark.parametrize(
    ("value", "dialect", "hooks"),
    [case[1:] for case in ENCODE_CASES],
    ids=[case[0] for case in ENCODE_CASES],
)
def test_json_dumps_writes_json_dumps_of_to_plain(
    value: object,
    dialect: object,
    hooks: Hooks | None,
    layout: str,
) -> None:
    written, expected = _dumps_outcomes(
        value,
        dialect=dialect,
        hooks=hooks,
        layout=_LAYOUTS[layout],
    )
    assert written == expected


@pytest.mark.parametrize(
    ("value", "hooks"),
    [(case[1], case[3]) for case in ENCODE_CASES],
    ids=[case[0] for case in ENCODE_CASES],
)
def test_json_loads_reads_back_what_json_dumps_wrote(
    value: object,
    hooks: Hooks | None,
) -> None:
    try:
        text = json.dumps(serialize._python_to_plain(value, hooks=hooks or {}))
    except (TypeError, ValueError):
        pytest.skip("the value does not encode")

    def python() -> object:
        tree = cast("PlainTree", json.loads(text))
        return deserialize._python_from_plain(
            tree,
            object,
            hooks=hooks or {},
            allow_imports=True,
        )

    assert _read_outcome(
        lambda: wire.json_loads(text, object, hooks=hooks or {}, allow_imports=True),
    ) == _read_outcome(python)


# Options ``json.dumps`` takes but would not take from a type checker: the
# core leaves each one it cannot read exactly to ``json``.
@pytest.mark.parametrize(
    ("ensure_ascii", "separators", "indent", "sort_keys"),
    [
        (0, None, None, False),
        (True, [",", ":"], None, False),
        (True, (1, 2), None, False),
        (True, ("\u00b7", "\u2192"), None, False),
        (True, None, 0, False),
        (True, None, -1, False),
        (True, None, 2.5, False),
        (True, None, "\u2192", True),
        (True, None, None, "yes"),
    ],
)
def test_json_dumps_takes_odd_options_as_json_dumps_does(
    ensure_ascii: object,
    separators: object,
    indent: object,
    sort_keys: object,
) -> None:
    layout = _Layout(
        ensure_ascii=cast("bool", ensure_ascii),
        separators=cast("tuple[str, str] | None", separators),
        indent=cast("int | str | None", indent),
        sort_keys=cast("bool", sort_keys),
    )
    value = [_tree(), {"\u00e9": [1.5, None], "b": (1, 2)}]
    written, expected = _dumps_outcomes(
        value,
        dialect="json",
        hooks=None,
        layout=layout,
    )
    assert written == expected


# What the core cannot write as text the way ``json`` would (an int past the
# digit limit, two keys that escape to the same text, a tree too deep) it
# leaves to ``json`` without encoding anything again: user code runs as many
# times as in Python, and errors carry the same text and notes.
_BIG = 10**5000


class _Counted:
    """A value written by a hook that counts its calls."""


class _Opaque:
    """A key hashed by identity, so two of them escape to the same text."""


class _Members:
    def __init__(self, members: Mapping[object, object]) -> None:
        for name, member in members.items():
            # Names that are not text too: ``__dict__`` takes any key.
            vars(self)[cast("str", name)] = member


class _Pair:
    __slots__ = ("a", "b")

    def __init__(self, a: object, b: object) -> None:
        self.a = a
        self.b = b


class _Float(float):
    __slots__ = ()


class _OddComplex(complex):
    __slots__ = ()

    @property
    @override
    def real(self) -> float:
        return _BIG


class _Repeating(Mapping[object, object]):
    """A mapping that lists one key twice."""

    @override
    def __iter__(self) -> Iterator[object]:
        return iter([1, 1, _Counted()])

    @override
    def __getitem__(self, key: object) -> object:
        return [_Counted(), 0 if isinstance(key, _Counted) else key]

    @override
    def __len__(self) -> int:
        return 3


class _Unreducible:
    @override
    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("cannot reduce")


class _Foreign:
    """A class whose encoder is a Python function, as one outside the core is."""

    def __init__(self, held: object = None) -> None:
        self.held = held


def _encode_foreign(value: object, state: serialize._Encoding) -> object:
    # An int key and a tuple: a tree ``json`` writes and the core's writer
    # leaves to it.
    del state
    return {1: "\u00e9", "held": cast("_Foreign", value).held, "pair": (1, 2)}


class _LateFailingReduce:
    """Reduces to ``first``, then a part that cannot be encoded at all.

    The walk then cuts back to the reduce and writes it by its slots.
    """

    __slots__ = ("first",)

    def __init__(self, first: object) -> None:
        self.first = first

    @override
    def __reduce__(self) -> str | tuple[object, ...]:
        return (_LateFailingReduce, (self.first, _Unreducible()))


class _Itemized(dict[str, object]):
    """A dict that writes how often its items were read into itself."""

    @override
    def items(self) -> dict_items[str, object]:
        self["asked"] = cast("int", self.get("asked", 0)) + 1
        return super().items()


def _shared_itemized() -> object:
    # Its two encoders' trees are left to ``json``, which reads its items
    # once for each.
    held = _Itemized(a=1)
    return [_Counted(), _Foreign(held), _Foreign(held)]


def _wrapped(depth: int, leaf: object, wrap: Callable[[object], object]) -> object:
    for _ in range(depth):
        leaf = wrap(leaf)
    return leaf


def _decode_counted(raw: object) -> _Counted:
    del raw
    return _Counted()


def _shared_under_an_int_name() -> object:
    inner = [1]
    return [_Members({5: inner}), inner, _Counted()]


_HOLE_CASES: dict[str, Callable[[], object]] = {
    "big in a list": lambda: [_Counted(), [1, _BIG], _Counted()],
    "big in a dict": lambda: {"a": _Counted(), "b": {"c": _BIG}, "d": _Counted()},
    "big in a tuple": lambda: (_Counted(), (1, _BIG)),
    "big in sets": lambda: [_Counted(), {1, _BIG}, frozenset({_BIG})],
    "big alone": lambda: _BIG,
    "big among leaves": lambda: [_Counted(), [1, 2, _BIG, 3], {"a": 1, "b": _BIG}],
    "big in objects": lambda: [
        _Counted(),
        _Members({"x": [_BIG]}),
        _Members({"x": _BIG}),
    ],
    "big in slots": lambda: [_Counted(), _Pair(1, _BIG), _Pair(_BIG, _Counted())],
    "big twice": lambda: [_Counted(), _BIG, _Counted(), [_BIG + 1]],
    "hook after big": lambda: [_BIG, _Counted(), {1}],
    "big key": lambda: [_Counted(), {_BIG: 1}],
    "nan keys": lambda: [
        _Counted(),
        {float("nan"): _Counted(), float("nan"): _Counted()},
    ],
    "object keys": lambda: [{_Opaque(): _Counted(), _Opaque(): _Counted()}, _Counted()],
    "tuple keys": lambda: [{(1, 2): _Counted(), (3,): [_BIG]}, _Counted()],
    "leaf keys": lambda: [
        {1: _Counted(), 2: [_BIG], 3.5: _Counted(), None: 1, False: 2},
        _Counted(),
    ],
    "tag-like keys": lambda: [{"py/a": _Counted(), "json://x": _Counted()}, _Counted()],
    "mapping repeating a key": lambda: [_Counted(), _Repeating()],
    "member named py/object": lambda: [_Counted(), _Members({"py/object": _Counted()})],
    "int member name": lambda: [_Counted(), _Members({5: _Counted()})],
    "str subclass member name": lambda: [_Counted(), _Members({_Text("y"): [_BIG]})],
    "reference to an int-named member": _shared_under_an_int_name,
    "float subclass": lambda: [_Counted(), _Float(1.5)],
    "complex": lambda: [_Counted(), 1 + 2j, complex(math.nan, 1), _OddComplex(1, 2)],
    "non-finite floats": lambda: [_Counted(), math.nan, math.inf, _Float("-inf")],
    "surrogate keys": lambda: [
        _Counted(),
        {"\ud800": {"\udc00x": _BIG}, "\ud800\udc00": [_BIG]},
    ],
    "deep lists": lambda: [_Counted(), _wrapped(150, [_BIG], lambda inner: [inner])],
    "deep tuples": lambda: _wrapped(120, [_Counted()], lambda inner: (inner,)),
    "deep dicts": lambda: _wrapped(
        300,
        {"k": _Counted()},
        lambda inner: {"k": inner, "o": _Counted()},
    ),
    "nan keys, non-ASCII": lambda: [
        _Counted(),
        {float("nan"): "\u00e9", float("nan"): _Counted()},
    ],
    "python encoder": lambda: [_Counted(), _Foreign(), _Foreign([_Counted()])],
    "python encoder, big": lambda: [_Counted(), _Foreign(_BIG)],
    "python encoders sharing a dict subclass": _shared_itemized,
    "reduce failing late": lambda: [
        _Counted(),
        _LateFailingReduce(_Foreign()),
        _LateFailingReduce(_BIG),
        _Counted(),
    ],
    "reduce failing late, after a python encoder": lambda: [
        _Foreign(),
        _LateFailingReduce(_Foreign()),
    ],
    "reduce failing late, after big": lambda: [_BIG, _LateFailingReduce(1)],
}


class _HoleLayout(NamedTuple):
    dialect: str = "json"
    separators: tuple[str, str] | None = None
    ensure_ascii: bool = True


_HOLE_LAYOUTS = {
    "default": _HoleLayout(),
    "compact": _HoleLayout(separators=(",", ":")),
    "no separators": _HoleLayout(separators=("", "")),
    "non-ASCII kept": _HoleLayout(ensure_ascii=False),
    "python dialect": _HoleLayout(dialect="python"),
}


@pytest.fixture
def digit_limit() -> Iterator[None]:
    """Hold ``int`` to the default digit limit, as a test before may have changed it."""
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(limit)


def _counted_dumps(
    make: Callable[[], object],
    write: Callable[[object, Hooks], str],
    payload: Callable[[int], object],
) -> tuple[object, int]:
    """Return what ``write`` gives for a fresh value, with its notes, and how often hooks ran."""
    calls: list[object] = []

    def encode(value: object) -> object:
        calls.append(value)
        return payload(len(calls))

    hooks: Hooks = {_Counted: (encode, _decode_counted)}
    value = make()
    try:
        result: object = ("ok", write(value, hooks))
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either side raises.
        notes = getattr(error, "__notes__", None)
        result = ("raise", type(error).__name__, str(error), notes)
    return result, len(calls)


def _hole_writers(
    layout: _HoleLayout,
) -> tuple[Callable[[object, Hooks], str], Callable[[object, Hooks], str]]:
    """Return ``json_dumps`` and Python's ``json.dumps(to_plain(...))`` under ``layout``."""
    style = cast("Dialect", layout.dialect)

    def written(value: object, hooks: Hooks) -> str:
        return wire.json_dumps(
            value,
            dialect=style,
            hooks=hooks,
            separators=layout.separators,
            ensure_ascii=layout.ensure_ascii,
        )

    def python(value: object, hooks: Hooks) -> str:
        # Every option, as ``json_dumps`` passes them: a replaced
        # ``json.dumps`` sees what it is passed.
        return json.dumps(
            serialize._python_to_plain(value, dialect=style, hooks=hooks),
            ensure_ascii=layout.ensure_ascii,
            separators=layout.separators,
            indent=None,
            sort_keys=False,
        )

    return written, python


def _assert_dumps_with_hooks_alike(
    make: Callable[[], object],
    layout: _HoleLayout,
    payload: Callable[[int], object] = lambda count: count,
) -> None:
    written, python = _hole_writers(layout)
    assert _counted_dumps(make, written, payload) == _counted_dumps(
        make,
        python,
        payload,
    )


@pytest.mark.usefixtures("digit_limit")
@pytest.mark.parametrize("layout", list(_HOLE_LAYOUTS))
@pytest.mark.parametrize("case", list(_HOLE_CASES))
def test_json_dumps_runs_hooks_as_often_as_python_where_text_falls_short(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    layout: str,
) -> None:
    monkeypatch.setitem(serialize._ENCODER_CACHE, _Foreign, _encode_foreign)
    for _ in range(_TIMES):
        _assert_dumps_with_hooks_alike(_HOLE_CASES[case], _HOLE_LAYOUTS[layout])


def test_json_dumps_gives_up_where_json_does_on_a_deep_tree() -> None:
    # The text walk could write any depth; ``json`` stops where the C stack
    # runs short, and so must ``json_dumps``. Notes differ with the stack.
    value = _wrapped(100_000, [_Counted()], lambda inner: [inner])

    def kind(write: Callable[[object, Hooks], str]) -> tuple[object, int]:
        outcome, calls = _counted_dumps(lambda: value, write, lambda count: count)
        return cast("tuple[object, ...]", outcome)[:2], calls

    assert kind(lambda tree, hooks: wire.json_dumps(tree, hooks=hooks)) == kind(
        lambda tree, hooks: json.dumps(serialize._python_to_plain(tree, hooks=hooks)),
    )


_JSON_DUMPS = cast("Callable[..., str]", json.dumps)
_JSON_LOADS = cast("Callable[[str | bytes], object]", json.loads)


class _ReplacedDumps:
    """A ``json.dumps`` put in place after import.

    Each text it writes starts with its call's place in line and keeps
    non-ASCII text, so a call Python does not make, or one on part of a tree,
    shows in the document.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, tree: object, **options: object) -> str:
        self.calls.append(options)
        text = _JSON_DUMPS(tree, **{**options, "ensure_ascii": False})
        return f"{len(self.calls)}:{text}"


class _ReplacedLoads:
    """A ``json.loads`` put in place after import that lists what it parses.

    A list it reads gains its call's place in line, so a document parsed
    without it reads otherwise.
    """

    def __init__(self) -> None:
        self.calls: list[str | bytes] = []

    def __call__(self, text: str | bytes) -> object:
        self.calls.append(text)
        tree = _JSON_LOADS(text)
        if isinstance(tree, list):
            return [*cast("list[object]", tree), len(self.calls)]
        return tree


@pytest.mark.usefixtures("digit_limit")
@pytest.mark.parametrize("layout", ["default", "compact", "python dialect"])
@pytest.mark.parametrize("case", [*_HOLE_CASES, "no part left to json"])
def test_json_dumps_calls_a_replaced_json_dumps_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    layout: str,
) -> None:
    # Python calls ``json.dumps`` once, on the whole tree, and ``to_plain``
    # calls it for set orders and escaped keys; the core must call one
    # replaced since import just so, not on parts of the tree or twice.
    monkeypatch.setitem(serialize._ENCODER_CACHE, _Foreign, _encode_foreign)
    make = _HOLE_CASES.get(case, lambda: [_Counted(), {"\u00e9": [1.5, None]}])
    written, python = _hole_writers(_HOLE_LAYOUTS[layout])

    def run(
        write: Callable[[object, Hooks], str],
    ) -> tuple[object, int, list[dict[str, object]]]:
        replaced = _ReplacedDumps()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(json, "dumps", replaced)
            outcome, hooks = _counted_dumps(make, write, lambda count: count)
        return outcome, hooks, replaced.calls

    assert run(written) == run(python)


_NO_TARGET = object()

_REPLACED_LOADS_CASES: dict[str, tuple[str | bytes, object]] = {
    "plain": ('[1, {"a": 2.5}]', _NO_TARGET),
    "plain bytes": (b"[1, 2]", _NO_TARGET),
    "typed": ("[1, 2]", list[int]),
    "typed records": ('{"a": [1], "b": [2, 3]}', dict[str, list[int]]),
    "escaped keys": ('{"json://1": [3], "json://2.5": [4]}', dict[float, list[int]]),
    "malformed": ("[1,", list[int]),
}


@pytest.mark.parametrize("case", list(_REPLACED_LOADS_CASES))
def test_json_loads_calls_a_replaced_json_loads_as_python_does(case: str) -> None:
    # Python calls ``json.loads`` once on the document, and ``from_plain``
    # once for each escaped key.
    text, target = _REPLACED_LOADS_CASES[case]

    def read() -> object:
        if target is _NO_TARGET:
            return wire.json_loads(text)
        return wire.json_loads(text, target)

    def python() -> object:
        tree = cast("PlainTree", json.loads(text))
        if target is _NO_TARGET:
            return tree
        return deserialize._python_from_plain(tree, target)

    def run(call: Callable[[], object]) -> tuple[object, list[str | bytes]]:
        replaced = _ReplacedLoads()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(json, "loads", replaced)
            outcome = _read_outcome(call)
        return outcome, replaced.calls

    for _ in range(_TIMES):
        assert run(read) == run(python)


@pytest.mark.usefixtures("digit_limit")
def test_json_dumps_runs_a_hook_once_whose_payload_json_cannot_write() -> None:
    _assert_dumps_with_hooks_alike(
        lambda: [_Counted(), {"k": _Counted()}],
        _HoleLayout(),
        lambda count: {"x": [_BIG + count], 1: 2},
    )


@pytest.mark.usefixtures("digit_limit")
@pytest.mark.parametrize("limit", [4300, 0], ids=["digit limit", "no digit limit"])
def test_json_dumps_writes_big_ints_as_json_does_under_the_digit_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    monkeypatch.setitem(serialize._ENCODER_CACHE, _Foreign, _encode_foreign)
    sys.set_int_max_str_digits(limit)
    _assert_dumps_with_hooks_alike(
        lambda: {"a": [_Counted(), _Foreign(_BIG)], "b": _BIG, "c": {_BIG: 1}},
        _HoleLayout(separators=(";", "=")),
    )


class _Flag:
    """A ``json.dumps`` flag that counts how often its truth is asked, and may act."""

    def __init__(self, value: bool, action: Callable[[], None] | None = None) -> None:
        self.value = value
        self.action = action
        self.asked = 0

    def __bool__(self) -> bool:
        self.asked += 1
        if self.action is not None:
            self.action()
        return self.value


def _raise_from_a_flag() -> None:
    raise RuntimeError("asked")


class _FlagCase(NamedTuple):
    """A value, and a flag for ``ensure_ascii``, or ``sort_keys`` if ``sorts``."""

    value: list[object]
    truth: bool = True
    sorts: bool = False
    grows: bool = False
    raises: bool = False
    indent: int | None = None


def _flag_cases() -> dict[str, _FlagCase]:
    return {
        "ensure_ascii": _FlagCase(["\u00e9"]),
        "sort_keys": _FlagCase([{"b": 1, "a": 2}], sorts=True),
        "ensure_ascii that grows the value": _FlagCase([1], grows=True),
        "sort_keys that grows the value": _FlagCase(
            [1],
            truth=False,
            sorts=True,
            grows=True,
        ),
        "ensure_ascii that grows an indented value": _FlagCase(
            [1],
            grows=True,
            indent=1,
        ),
        "raising, with a value that cannot be encoded": _FlagCase(
            [_Unreducible()],
            raises=True,
        ),
        "raising, with a value that can": _FlagCase([1], raises=True),
    }


@pytest.mark.parametrize("case", list(_flag_cases()))
def test_json_dumps_asks_a_flags_truth_when_json_dumps_does(case: str) -> None:
    # ``json`` asks only once the tree is built, so a flag that acts sees,
    # or changes, what Python's walk left.
    def run(write: Callable[[list[object], object, object], str]) -> tuple[object, int]:
        value, truth, sorts, grows, raises, _ = _flag_cases()[case]
        action = (
            (lambda: value.append(2))
            if grows
            else _raise_from_a_flag
            if raises
            else None
        )
        flag = _Flag(truth, action)
        outcome = _text_outcome(
            lambda: write(value, True, flag) if sorts else write(value, flag, False),
        )
        return outcome, flag.asked

    indent = _flag_cases()[case].indent

    def written(value: list[object], ensure_ascii: object, sort: object) -> str:
        return wire.json_dumps(
            value,
            ensure_ascii=cast("bool", ensure_ascii),
            sort_keys=cast("bool", sort),
            indent=indent,
        )

    def python(value: list[object], ensure_ascii: object, sort: object) -> str:
        return json.dumps(
            serialize._python_to_plain(value),
            ensure_ascii=cast("bool", ensure_ascii),
            sort_keys=cast("bool", sort),
            indent=indent,
        )

    assert run(written) == run(python)


@pytest.mark.parametrize("kind", [list, tuple])
def test_json_dumps_reads_separators_after_hooks_run(kind: type) -> None:
    # A hook may change a list of separators, and json reads them only once
    # the tree is built.
    def run(write: Callable[[object, Hooks, object], str]) -> object:
        separators = [", ", ": "]

        def encode(value: object) -> object:
            del value
            separators[0] = ";"
            return 0

        hooks: Hooks = {_Counted: (encode, _decode_counted)}
        given = separators if kind is list else tuple(separators)
        return _text_outcome(lambda: write([_Counted(), {"a": 1}], hooks, given))

    assert run(
        lambda value, hooks, separators: wire.json_dumps(
            value,
            hooks=hooks,
            separators=cast("tuple[str, str]", separators),
        ),
    ) == run(
        lambda value, hooks, separators: json.dumps(
            serialize._python_to_plain(value, hooks=hooks),
            separators=cast("tuple[str, str]", separators),
        ),
    )


def _float_samples() -> list[float]:
    """Return float edges, then random bit patterns and random magnitudes, all finite."""
    rng = random.Random(20_261_018)  # noqa: S311 -- Seeded test data, not secrets.
    samples = [
        0.0,
        -0.0,
        5e-324,
        -5e-324,
        2.2250738585072009e-308,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        0.1,
        0.2,
        0.3,
        1 / 3,
        1e16,
        1e-7,
        1e21,
        1e22,
        1e23,
        9007199254740993.0,
        123456789.123456789,
    ]
    while len(samples) < 4_000:
        (value,) = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))
        if math.isfinite(value):
            samples.append(value)
    samples.extend(rng.uniform(-1e6, 1e6) for _ in range(1_000))
    # Quarters near 2**50 to 2**53 lie halfway between two shortest spellings
    # (``x.25`` between ``x.2`` and ``x.3``), which ``repr`` breaks to the even.
    samples.extend(
        sign * (base + odd) / 4
        for base in (2**50, 2**51, 2**52, 2**53 - 64)
        for odd in range(1, 64, 2)
        for sign in (1, -1)
    )
    return samples


def _float_texts(samples: list[float]) -> list[str]:
    """Return several spellings of each sample, and long decimals no double holds exactly."""
    rng = random.Random(7)  # noqa: S311 -- Seeded test data, not secrets.
    texts: list[str] = []
    for value in samples:
        texts.extend((repr(value), f"{value:.17e}", f"{value:.{rng.randint(1, 16)}g}"))
    for _ in range(1_000):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(18, 40)))
        texts.append(f"{digits[0]}.{digits[1:]}e{rng.randint(-330, 310)}")
    return texts


def test_floats_parse_and_write_as_json_does() -> None:
    samples = _float_samples()
    text = "[" + ", ".join(_float_texts(samples)) + "]"
    assert _text_outcome(lambda: wire.json_loads(text)) == _text_outcome(
        lambda: cast("object", json.loads(text)),
    )

    def python() -> object:
        tree = cast("PlainTree", json.loads(text))
        return deserialize._python_from_plain(tree, list[float])

    assert _read_outcome(lambda: wire.json_loads(text, list[float])) == _read_outcome(
        python,
    )
    rows = (
        "[" + ", ".join(f'{{"id": 0, "score": {value!r}}}' for value in samples) + "]"
    )

    def python_rows() -> object:
        tree = cast("PlainTree", json.loads(rows))
        return deserialize._python_from_plain(tree, list[_FrozenRow])

    assert _read_outcome(
        lambda: wire.json_loads(rows, list[_FrozenRow]),
    ) == _read_outcome(python_rows)
    assert wire.json_dumps(samples) == json.dumps(serialize._python_to_plain(samples))


def _string_samples() -> list[str]:
    """Return strings built from escapes, controls, surrogates and wide characters."""
    rng = random.Random(11)  # noqa: S311 -- Seeded test data, not secrets.
    alphabet = [
        "a",
        "\x00",
        "\x1f",
        "\x7f",
        "\x80",
        "\u00e9",
        "\u20ac",
        "\u2028",
        "\uffff",
        "\ud800",
        "\udfff",
        "\U0001f600",
        '"',
        "\\",
        "/",
        "\n",
        "\t",
        "\b",
        "\f",
        "\r",
    ]
    samples = ["", *alphabet, "x" * 5_000]
    samples.extend(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
        for _ in range(300)
    )
    # ASCII text is checked for escapes eight bytes at a time, then byte by
    # byte: put each character that needs one at every place in both.
    samples.extend(
        "a" * at + special + "a" * (size - at - 1)
        for special in ('"', "\\", "\x7f", "\x00", "\x1f", "\n")
        for size in range(1, 26)
        for at in range(size)
    )
    return samples


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_strings_parse_and_write_as_json_does(ensure_ascii: bool) -> None:
    samples = _string_samples()
    expected = json.dumps(
        serialize._python_to_plain(samples),
        ensure_ascii=ensure_ascii,
    )
    assert wire.json_dumps(samples, ensure_ascii=ensure_ascii) == expected
    keyed = {text: index for index, text in enumerate(samples)}
    assert wire.json_dumps(keyed, ensure_ascii=ensure_ascii) == json.dumps(
        serialize._python_to_plain(keyed),
        ensure_ascii=ensure_ascii,
    )
    assert _parses_alike(expected)

    def python() -> object:
        tree = cast("PlainTree", json.loads(expected))
        return deserialize._python_from_plain(tree, list[str])

    assert _read_outcome(lambda: wire.json_loads(expected, list[str])) == _read_outcome(
        python,
    )


# The plain-tree writer takes any tree, not only ``to_plain``'s, so it meets
# what ``to_plain`` never writes: cycles, non-text keys and leaf subclasses.
class _ShownInt(int):
    @override
    def __repr__(self) -> str:
        return "shown"


class _ShownFloat(float):
    @override
    def __repr__(self) -> str:
        return "shown"


class _Text(str):
    __slots__ = ()


def _plain_trees() -> dict[str, object]:
    cycle: list[object] = []
    cycle.append(cycle)
    looped: dict[str, object] = {}
    looped["self"] = looped
    return {
        "list cycle": cycle,
        "dict cycle": looped,
        "non-text keys": {1: "a", 2.5: "b", None: "c", False: "d"},
        "mixed keys sorted": {"b": 1, 2: 3},
        "leaf subclasses": [_ShownInt(1), _ShownFloat(2.5), _Text("t"), True],
        "subclass key": {_Text("k"): 1},
        "tuple": (1, [2, (3,)]),
        "not plain": [object()],
        "nested": {"b": [1, {"d": 1.5, "c": None}], "a": "\u00e9"},
    }


@needs_rust
@pytest.mark.parametrize("sort_keys", [False, True])
@pytest.mark.parametrize("indent", [None, 1])
@pytest.mark.parametrize("name", list(_plain_trees()))
def test_the_plain_tree_writer_writes_as_json_dumps_does(
    name: str,
    indent: int | None,
    sort_keys: bool,
) -> None:
    from treekle import _rust  # noqa: PLC0415 -- Absent in a pure-Python run.

    tree = _plain_trees()[name]
    assert _text_outcome(
        lambda: _rust.json_dumps_plain(tree, indent=indent, sort_keys=sort_keys),
    ) == _text_outcome(lambda: json.dumps(tree, indent=indent, sort_keys=sort_keys))


@needs_rust
def test_a_cyclic_untagged_tree_reads_natively_as_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    _refuse_python_decoding(monkeypatch)
    assert deserialize.from_plain(cycle, object) is cycle


def _chain(depth: int) -> Node:
    node = Node("leaf")
    for _ in range(depth):
        node = Node("link", parent=node)
    return node


def _plain_chain(depth: int) -> dict[str, object]:
    node: dict[str, object] = {"name": "leaf"}
    for _ in range(depth):
        node = {"name": "link", "parent": node}
    return node


_DEEP_CALLS: dict[str, tuple[Callable[[], object], Callable[[object], object]]] = {
    "list to_plain": (lambda: _nested(3000), serialize.to_plain),
    "list from_plain": (
        lambda: _nested(3000),
        lambda tree: deserialize.from_plain(tree, object),
    ),
    "dataclass to_plain": (lambda: _chain(3000), serialize.to_plain),
    "dataclass from_plain": (
        lambda: _plain_chain(3000),
        lambda tree: deserialize.from_plain(tree, Node),
    ),
    "union from_plain": (
        lambda: _deep(0, 3000),
        lambda tree: deserialize.from_plain(tree, _Tree),
    ),
}


@pytest.mark.parametrize(
    ("build", "call"),
    list(_DEEP_CALLS.values()),
    ids=list(_DEEP_CALLS),
)
def test_deep_nesting_on_a_small_thread_stack_round_trips(
    build: Callable[[], object],
    call: Callable[[object], object],
) -> None:
    # A 512 KiB stack runs out long before the recursion limit is reached. The
    # tree is built and freed on this thread: CPython 3.13 itself overflows a
    # small stack when freeing a few thousand nested dataclasses.
    tree = build()
    raised: list[type[BaseException] | None] = []

    def run() -> None:
        try:
            call(tree)
        except RecursionError as error:
            raised.append(type(error))
        else:
            raised.append(None)

    previous = threading.stack_size(512 * 1024)
    try:
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
    finally:
        threading.stack_size(previous)
    assert raised == [None]


def _run_python(script: str, **env: str) -> str:
    """Run ``script`` in a fresh interpreter; return what it printed."""
    completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        env={**os.environ, **env},
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_a_hook_that_reloads_the_module_does_not_break_the_walk() -> None:
    # Reloading installs fresh helpers while a walk still uses the old ones.
    printed = _run_python(
        """
        import importlib
        import treekle.deserialize as deserialize
        import treekle.serialize as serialize

        class Leaf:
            pass

        def encode(value):
            importlib.reload(serialize)
            return [1.5, (1, 2), {3, 4}]

        def decode(data):
            importlib.reload(deserialize)
            return Leaf()

        hooks = {Leaf: (encode, decode)}
        for _ in range(20):
            tree = serialize.to_plain([Leaf(), [1.0, (2, 3)] * 100, Leaf()], hooks=hooks)
            value = deserialize.from_plain(tree, object, hooks=hooks)
        print(type(value[0]).__name__, len(value[1]))
        """,
    )
    assert printed == "Leaf 200"


@needs_rust
def test_reentrant_setup_does_not_restore_old_decoder_plans() -> None:
    printed = _run_python(
        """
        from types import MappingProxyType
        from treekle import _rust
        import treekle.deserialize as deserialize

        class Leaf:
            pass

        # Keep an old bytes plan in the cache taken by the outer hook read.
        assert deserialize.from_plain("YQ==", bytes) == b"a"

        def decode(_payload):
            def changed(_data, _target, _at, _state):
                return b"changed"

            deserialize._decode_bytes = changed
            deserialize._DECODERS = MappingProxyType({
                **deserialize._DECODERS, bytes: changed,
            })
            _rust.setup_decode(vars(deserialize))
            return Leaf()

        deserialize.from_plain(
            {"py/hook": ["__main__.Leaf", 0]},
            object,
            hooks={Leaf: (lambda _value: 0, decode)},
        )
        print(
            deserialize.from_plain("YQ==", bytes),
            deserialize._python_from_plain("YQ==", bytes),
        )
        """,
    )
    assert printed == "b'changed' b'changed'"


_WHICH = """
import treekle.deserialize as deserialize
import treekle.serialize as serialize

print(
    serialize.to_plain is serialize._python_to_plain,
    deserialize.from_plain is deserialize._python_from_plain,
)
"""


def test_the_environment_variable_selects_python() -> None:
    assert _run_python(_WHICH, TREEKLE_PURE_PYTHON="1") == "True True"


def test_a_missing_core_falls_back_to_python() -> None:
    hidden = "import sys\nsys.modules['treekle._rust'] = None\n" + _WHICH
    assert _run_python(hidden, TREEKLE_PURE_PYTHON="") == "True True"


def test_a_core_that_cannot_set_up_falls_back_to_python_with_a_warning() -> None:
    # A core built from older sources imports, then refuses this module.
    mismatched = """
        import sys
        import types
        import warnings

        def refuse(module):
            raise AttributeError("treekle.serialize has no OLD_HELPER")

        core = types.ModuleType("treekle._rust")
        core.setup_encode = core.setup_decode = refuse
        core.to_plain = core.from_plain = refuse
        sys.modules["treekle._rust"] = core
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import treekle.deserialize as deserialize
            import treekle.serialize as serialize

        print(
            serialize.to_plain is serialize._python_to_plain,
            deserialize.from_plain is deserialize._python_from_plain,
            sum(
                warning.category is RuntimeWarning
                and "OLD_HELPER" in str(warning.message)
                for warning in caught
            ),
        )
        """
    assert _run_python(mismatched, TREEKLE_PURE_PYTHON="") == "True True 2"


def test_a_core_without_the_decoder_reads_with_python_and_warns() -> None:
    # A core built before the decoder existed has only the encoder.
    encoder_only = """
        import sys
        import types
        import warnings

        core = types.ModuleType("treekle._rust")
        core.setup_encode = lambda module: None
        core.to_plain = lambda obj, **options: obj
        sys.modules["treekle._rust"] = core
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import treekle.deserialize as deserialize
            import treekle.serialize as serialize

        print(
            serialize.to_plain is core.to_plain,
            deserialize.from_plain is deserialize._python_from_plain,
            sum(
                warning.category is RuntimeWarning
                and "from_plain" in str(warning.message)
                for warning in caught
            ),
        )
        """
    assert _run_python(encoder_only, TREEKLE_PURE_PYTHON="") == "True True 1"


# Runs whenever a core is built, even in a TREEKLE_PURE_PYTHON run, so a core
# that installs only one direction fails here instead of skipping.
@pytest.mark.skipif(
    util.find_spec("treekle._rust") is None,
    reason="the Rust core is not built",
)
@pytest.mark.parametrize("setting", ["", "0"])
def test_a_built_core_is_installed(setting: str) -> None:
    assert _run_python(_WHICH, TREEKLE_PURE_PYTHON=setting) == "False False"


@needs_rust
@pytest.mark.parametrize(
    ("rust", "python"),
    [
        (serialize.to_plain, serialize._python_to_plain),
        (deserialize.from_plain, deserialize._python_from_plain),
    ],
    ids=["to_plain", "from_plain"],
)
def test_the_rust_functions_document_themselves_as_python_does(
    rust: Callable[..., object],
    python: Callable[..., object],
) -> None:
    assert inspect.getdoc(rust) == inspect.getdoc(python)
    ours = inspect.signature(rust).parameters
    theirs = inspect.signature(python).parameters
    assert [(each.name, each.kind) for each in ours.values()] == [
        (each.name, each.kind) for each in theirs.values()
    ]
    # A text signature shows a default only when it is a literal.
    literal = {
        name: default
        for name, each in ours.items()
        if (default := cast("object", each.default)) is not Ellipsis
    }
    assert literal == {name: cast("object", theirs[name].default) for name in literal}


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
