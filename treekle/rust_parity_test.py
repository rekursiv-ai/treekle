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
import math
import os
import subprocess
import sys
import textwrap
import threading
import weakref

import pytest

from treekle import deserialize, serialize
from treekle.types import ReadError


if TYPE_CHECKING:
    from _collections_abc import dict_items
    from abc import ABCMeta
    from collections.abc import Callable

    from treekle.types import Hooks, PlainTree


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
