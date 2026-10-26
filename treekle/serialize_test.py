"""Tests for ``treekle.serialize``."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import (
    Mapping,
    MutableMapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal
from enum import Enum, IntEnum
from functools import partial
from pathlib import Path, PurePosixPath
from types import GenericAlias, MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    Protocol,
    SupportsIndex,
    TypedDict,
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import datetime as dt
import json
import math
import re
import threading
import typing
import weakref

import pytest

from treekle import serialize
from treekle.deserialize import from_plain
from treekle.lib.testing.resources import recursion_headroom
from treekle.plain import immutable
from treekle.serialize import to_plain
from treekle.types import PlainTree, ReadError


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle.types import Dialect, Hooks, MutablePlainTree


pytestmark = pytest.mark.usefixtures("implementation")


@pytest.fixture(autouse=True)
def fresh_encoder_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test choose its encoders anew, not read an earlier test's."""
    monkeypatch.setattr(
        serialize,
        "_ENCODER_CACHE",
        weakref.WeakKeyDictionary[type, object](),
    )


def _round_trip(value: object) -> object:
    return from_plain(to_plain(value), object, allow_imports=True)


def _unresolved(value: object, path: str) -> str:
    """Return the anchored pattern of ``_verified_path``'s whole error message."""
    return re.escape(
        f"Cannot serialize {value!r}: import path {path!r} does not resolve to "
        "the same object.",
    ).join("^$")


def _no_leaf(name: str) -> str:
    """Return the anchored pattern of the unknown-leaf error for class ``name``."""
    return re.escape(
        f"Cannot serialize leaf of type {name!r}. "
        f"Pass hooks={{{name}: (encode, decode)}} to to_plain().",
    ).join("^$")


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Bare:
    """A class with no reduce hooks of its own."""

    def __init__(self, x: int) -> None:
        self.x = x


class Slotted:
    """Slots plus a ``__dict__`` member, so both state paths are walked."""

    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


class Stateful:
    """Owns its pickled state through ``__setstate__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def __reduce__(self) -> tuple[object, tuple[int], dict[str, int]]:
        return (Stateful, (0,), {"value": self.value})

    def __setstate__(self, state: dict[str, int]) -> None:
        self.value = state["value"] * 10


@dataclass(frozen=True, kw_only=True, slots=True)
class Point:
    x: int
    y: float = 0.0


@dataclass(kw_only=True, slots=True)
class Node:
    name: str
    children: list[Node] = field(default_factory=list["Node"])


class Shape(Protocol):
    def area(self) -> float: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class Square:
    side: float

    def area(self) -> float:
        return self.side**2


@dataclass(frozen=True, kw_only=True, slots=True)
class Holder:
    shape: Shape


@dataclass(kw_only=True, slots=True)
class Flagged:
    """A dataclass with a slot that is state but not an ``__init__`` field."""

    name: str
    _done: bool = field(default=False, init=False)


class Tensor:
    """A leaf only a hook can encode."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    __slots__ = ("values",)


def _tensor_values(tensor: Tensor) -> list[float]:
    return tensor.values


_HOOKS: Hooks = MappingProxyType({Tensor: (_tensor_values, Tensor)})


class Unreducible(dict[str, int]):
    """A dict subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class UnreducibleList(list[int]):
    """A list subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class UnreducibleSet(set[int]):
    """A set subclass whose reduce refuses."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class NamedReduce:
    """Reduces to a name, which has no plain form here."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        return "NamedReduce"


class LocalReduce:
    """Reduces through a local callable, so state encoding takes over."""

    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        def rebuild() -> None:
            pass

        return (rebuild, ())


class LocalOnly:
    """Reduces through a local callable and has no state to fall back on."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        def rebuild() -> None:
            pass

        return (rebuild, ())


class Carrier(list[int]):
    """A list whose reduce state holds what only its own reduce could write."""

    def __init__(self, items: list[int], *, cargo: object) -> None:
        super().__init__(items)
        self.cargo = cargo


class Level(IntEnum):
    LOW = 1


@dataclass(
    slots=True,
)  # house-ignore[dataclass] -- An Enum mixin is built positionally; frozen+slots breaks it before 3.14.
class Coord:
    x: int


class Compass(Coord, Enum):
    """A dataclass enum, which still encodes as an enum member."""

    N = 1


class Corner(Coord, Enum):
    """A dataclass enum whose reduce refuses, so its state is written instead."""

    A = 1

    @override
    def __reduce_ex__(self, proto: object) -> tuple[object, ...]:
        raise TypeError("no reduce")


@dataclass(frozen=True, kw_only=True, slots=True)
class Ranked:
    """A set member whose ``repr`` is its label alone."""

    z: int
    a: int
    label: str = ""

    @override
    def __repr__(self) -> str:
        return self.label


class Sentinel:
    """Reduces to the name of its one module-level instance."""

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        return "SENTINEL"


SENTINEL = Sentinel()


class Recipe:
    """Reduces to whatever recipe it holds."""

    def __init__(self, recipe: object) -> None:
        self.recipe: object = recipe

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        return cast("tuple[object, ...]", self.recipe)


class Boxed:
    """Reduces to a call taking the very list it holds."""

    def __init__(self, items: list[int]) -> None:
        self.items = items

    @override
    def __reduce__(self) -> tuple[object, tuple[list[int]]]:
        return (Boxed, (self.items,))


class Items(list[int]):
    """A list subclass, which reduces with list items."""


@dataclass(kw_only=True, slots=True)
class Lazy:
    """A dataclass whose non-init slot stays unset until first use."""

    name: str
    cache: int = field(init=False)


@dataclass(frozen=True, kw_only=True, slots=True)
class Zed:
    """Sorts after ``Alpha`` by class path, before it by state."""

    a: int

    @override
    def __repr__(self) -> str:
        return ""


@dataclass(frozen=True, kw_only=True, slots=True)
class Alpha:
    """Sorts before ``Zed`` by class path, after it by state."""

    a: int

    @override
    def __repr__(self) -> str:
        return ""


@dataclass(eq=False, kw_only=True, slots=True)
class Member:
    """A set member that holds the set it is in."""

    owner: set[object]


class Celsius(float):
    """A float subclass."""


class Shadowed:
    """A slot whose name also sits in the instance ``__dict__``."""

    __slots__ = ("__dict__", "items")

    def __init__(self, items: list[int]) -> None:
        self.items = items
        vars(self)["items"] = "shadow"

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class Private:
    """A private slot, which Python stores under its mangled name."""

    __slots__ = ("__secret", "shown")

    def __init__(self, secret: int, shown: int) -> None:
        self.__secret = secret
        self.shown = shown

    def secret(self) -> int:
        return self.__secret

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class Faceless:
    """A set member with the default, address-bearing ``repr``."""

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class Moved:
    """A class whose module name is later rebound to another class."""


class Nest:
    """Holds a dataclass and a function whose names under it are later rebound."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Inner:
        a: int = 1

    def method_like(self) -> None:
        """Do nothing; read through the class, it is a plain function."""
        del self


def _ranked(z: int, a: int, label: str = "") -> MutablePlainTree:
    return {"py/object": f"{__name__}.Ranked", "z": z, "a": a, "label": label}


# Members order by their encoded tree with sorted keys. Sorting by ``a`` (the
# first sorted key), by ``z`` (the first written key), and CPython's iteration
# order give three different orders for these members.
_RANKED = (Ranked(z=3, a=1), Ranked(z=2, a=2), Ranked(z=1, a=3))
_RANKED_TREES = [_ranked(3, 1), _ranked(2, 2), _ranked(1, 3)]


def _around(middle: object) -> list[object]:
    """Return ``middle`` between two occurrences of one list."""
    shared = [1]
    return [shared, middle, shared]


def _then[T](build: Callable[[T], object], part: T) -> list[object]:
    """Return a value built around ``part``, then ``part`` itself."""
    return [build(part), part]


def _twice(value: object) -> list[object]:
    return [value, value]


def _node(children: list[Node]) -> Node:
    return Node(name="n", children=children)


def _frozen(member: object) -> frozenset[object]:
    return frozenset({member})


def _fake_encoder(value: object, state: object) -> MutablePlainTree:
    del state
    return f"fake {type(value).__name__}"


_JSON_FLOATS: MutablePlainTree = [
    {"py/float": "inf"},
    1.5,
    {"py/complex": [{"py/float": "-inf"}, {"py/float": "nan"}]},
]


_STATE_TUPLE = Recipe((Recipe, (1,), ("x", dict[str, int]())))
_STATE_TUPLE_TREE: MutablePlainTree = {
    "py/reduce": [
        {"py/type": f"{__name__}.Recipe"},
        {"py/tuple": [1]},
        {"py/tuple": ["x", {}]},
    ],
}


_NODE_TREE: MutablePlainTree = {
    "py/object": f"{__name__}.Node",
    "name": "n",
    "children": [],
}


def _newobj(kind: type) -> MutablePlainTree:
    """Return the reduce recipe of a stateless instance of ``kind``."""
    return [
        {"py/function": "copyreg.__newobj__"},
        {"py/tuple": [{"py/type": f"{__name__}.{kind.__qualname__}"}]},
    ]


_BARE_TREE: MutablePlainTree = {
    "py/reduce": [
        {"py/function": "copyreg.__newobj__"},
        {"py/tuple": [{"py/type": f"{__name__}.Bare"}]},
        {"x": 1},
    ],
}


class TestToPlain:
    def test_tags_name_python_types(self) -> None:
        tree = to_plain(
            {
                "t": (1, 2),
                "s": {2, 1},
                "f": frozenset({3}),
                "b": b"\x00",
                "p": Path("/a"),
                "u": UUID(int=1),
                "c": 1 + 2j,
            },
        )

        assert tree == {
            "t": {"py/tuple": [1, 2]},
            "s": {"py/set": [1, 2]},
            "f": {"py/frozenset": [3]},
            "b": {"py/b64": "AA=="},
            "p": {"py/path": "/a"},
            "u": {"py/uuid": "00000000-0000-0000-0000-000000000001"},
            "c": {"py/complex": [1.0, 2.0]},
        }

    @pytest.mark.parametrize(
        ("dialect", "expected"),
        [
            (None, _JSON_FLOATS),
            ("json", _JSON_FLOATS),
            ("python", [math.inf, 1.5, {"py/complex": [-math.inf, math.nan]}]),
        ],
        ids=repr,
    )
    def test_the_dialect_decides_which_floats_are_tagged(
        self,
        dialect: Dialect | None,
        expected: MutablePlainTree,
    ) -> None:
        value = [math.inf, 1.5, complex(-math.inf, math.nan)]

        tree = to_plain(value) if dialect is None else to_plain(value, dialect=dialect)

        assert repr(tree) == repr(expected)

    @pytest.mark.parametrize(
        ("dialect", "expected"),
        [
            ("json", {"inf": {"py/float": "inf"}, "t": {"py/tuple": [1]}, "f": 1.5}),
            ("python", {"inf": math.inf, "t": {"py/tuple": [1]}, "f": 1.5}),
        ],
    )
    def test_each_dict_value_before_a_plain_float_is_encoded(
        self,
        dialect: Dialect,
        expected: MutablePlainTree,
    ) -> None:
        tree = to_plain({"inf": math.inf, "t": (1,), "f": 1.5}, dialect=dialect)

        assert tree == expected

    @pytest.mark.parametrize(
        ("dialect", "value"),
        [("json", 1.5), ("python", math.inf)],
    )
    def test_a_dict_of_plain_values_encodes_no_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dialect: Dialect,
        value: float,
    ) -> None:
        encoded: list[object] = []
        encode_node = serialize._encode_node

        def counted(item: object, state: serialize._Encoding) -> object:
            encoded.append(item)
            return encode_node(item, state)

        monkeypatch.setattr(serialize, "_encode_node", counted)
        root = {"n": None, "i": 1, "s": "s", "f": value}

        serialize._python_to_plain(root, dialect=dialect)

        assert encoded == [root]

    @pytest.mark.parametrize("dialect", ["yaml", "JSON"])
    def test_an_unknown_dialect_is_rejected(self, dialect: str) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^unknown dialect '{dialect}'; expected 'json' or 'python'$",
        ):
            to_plain(math.inf, dialect=cast("Dialect", dialect))

    def test_immutable_output_is_read_only(self) -> None:
        tree = to_plain({"a": [(1,)]}, mutable=False)

        assert isinstance(tree, MappingProxyType)
        assert tree["a"] == (MappingProxyType({"py/tuple": (1,)}),)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (_twice([1]), [[1], {"py/ref": [0]}]),
            (_twice({1}), [{"py/set": [1]}, {"py/ref": [0]}]),
            (_twice(_STATE_TUPLE), [_STATE_TUPLE_TREE, {"py/ref": [0]}]),
            (
                _then(_frozen, Bare(1)),
                [{"py/frozenset": [_BARE_TREE]}, {"py/ref": [0, 0]}],
            ),
            (
                _then(_node, list[Node]()),
                [_NODE_TREE, {"py/ref": [0, "children"]}],
            ),
            (
                _then(Boxed, [1]),
                [
                    {
                        "py/reduce": [
                            {"py/type": f"{__name__}.Boxed"},
                            {"py/tuple": [[1]]},
                        ],
                    },
                    {"py/ref": [0, 1, 0]},
                ],
            ),
        ],
    )
    def test_shared_objects_become_references(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    @pytest.mark.parametrize(
        ("middle", "expected"),
        [
            pytest.param({2: "x"}, {"json://2": "x"}, id="escaped-key"),
            pytest.param({1, 2}, {"py/set": [1, 2]}, id="set"),
            pytest.param(
                LocalReduce(1, 2),
                {"py/object": f"{__name__}.LocalReduce", "a": 1, "b": 2},
                id="failed-reduce",
            ),
        ],
    )
    def test_a_trial_encoding_forgets_only_what_it_registered(
        self,
        middle: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(_around(middle)) == [[1], expected, {"py/ref": [0]}]

    def test_a_hooked_value_and_its_payload_are_referenced(self) -> None:
        tensor = Tensor([1.0])

        tree = to_plain([tensor, tensor, tensor.values], hooks=_HOOKS)

        assert tree == [
            {"py/hook": [f"{__name__}.Tensor", [1.0]]},
            {"py/ref": [0]},
            {"py/ref": [0, 1]},
        ]

    def test_an_unset_slot_is_omitted(self) -> None:
        assert to_plain(Lazy(name="n")) == {
            "py/object": f"{__name__}.Lazy",
            "name": "n",
        }

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Point, {"py/type": f"{__name__}.Point"}),
            (
                Color.RED,
                {
                    "py/reduce": [
                        {"py/type": f"{__name__}.Color"},
                        {"py/tuple": ["red"]},
                    ],
                },
            ),
            (
                Compass.N,
                {
                    "py/reduce": [
                        {"py/type": f"{__name__}.Compass"},
                        {"py/tuple": [{"py/object": f"{__name__}.Coord", "x": 1}]},
                    ],
                },
            ),
            (Path("/a"), {"py/path": "/a"}),
            (UUID(int=1), {"py/uuid": "00000000-0000-0000-0000-000000000001"}),
            (
                dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
                {"py/datetime": "2026-01-02T00:00:00+00:00"},
            ),
            (Point(x=1), {"py/object": f"{__name__}.Point", "x": 1, "y": 0.0}),
            (Bare(1), _BARE_TREE),
        ],
    )
    def test_each_class_chooses_its_encoder_once(
        self,
        value: object,
        expected: MutablePlainTree,
        implementation: str,
    ) -> None:
        kind = type(value)

        tree = to_plain(value)

        assert tree == expected
        # The Rust core keeps what it chose with its own class facts.
        if implementation == "python":
            assert serialize._ENCODER_CACHE[kind] is serialize._encoder_for(kind)

    def test_object_attributes_sort_by_name_but_can_keep_declaration_order(
        self,
    ) -> None:
        node = Node(name="root")

        sorted_tree = to_plain(node)
        declared_tree = to_plain(node, sort=False)

        assert isinstance(sorted_tree, dict)
        assert isinstance(declared_tree, dict)
        assert list(sorted_tree) == ["py/object", "children", "name"]
        assert list(declared_tree) == ["py/object", "name", "children"]

    def test_a_cached_encoder_is_reused(self) -> None:
        serialize._ENCODER_CACHE[Bare] = _fake_encoder

        assert to_plain([Bare(1)]) == ["fake Bare"]

    def test_a_shared_mutable_reduced_value_becomes_a_reference(self) -> None:
        shared = bytearray(b"abc")

        assert to_plain([shared, shared]) == [
            {
                "py/reduce": [
                    {"py/type": "builtins.bytearray"},
                    {"py/tuple": ["abc", "latin-1"]},
                ],
            },
            {"py/ref": [0]},
        ]

    def test_a_repeated_value_typed_reduce_is_written_in_full(self) -> None:
        member = to_plain(Color.RED)

        assert to_plain([Color.RED, Color.RED]) == [member, member]

    def test_a_shared_mappingproxy_becomes_a_reference(self) -> None:
        shared = MappingProxyType({"a": 1})

        assert to_plain([shared, shared]) == [
            {"py/mappingproxy": {"a": 1}},
            {"py/ref": [0]},
        ]

    def test_a_value_inside_a_mappingproxy_is_named_through_its_tag(self) -> None:
        inner = [1]

        assert to_plain([MappingProxyType({"k": inner}), inner]) == [
            {"py/mappingproxy": {"k": [1]}},
            {"py/ref": [0, "py/mappingproxy", "k"]},
        ]

    def test_a_reference_names_the_path_of_the_first_occurrence(self) -> None:
        shared = {"k": 1}

        tree = to_plain({"a": {"b": (shared,)}, "c": shared})

        assert tree == {
            "a": {"b": {"py/tuple": [{"k": 1}]}},
            "c": {"py/ref": ["a", "b", 0]},
        }

    def test_cycles_terminate(self) -> None:
        cycle: list[object] = []
        cycle.append(cycle)

        assert to_plain(cycle) == [{"py/ref": []}]

    def test_tag_lookalike_and_non_str_keys_are_escaped(self) -> None:
        tree = to_plain({"py/x": 1, 2: 3, "json://y": 4})

        assert tree == {'json://"py/x"': 1, "json://2": 3, 'json://"json://y"': 4}

    def test_dataclass_names_its_class(self) -> None:
        assert to_plain(Point(x=1)) == {
            "py/object": f"{__name__}.Point",
            "x": 1,
            "y": 0.0,
        }

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                Level.LOW,
                {"py/reduce": [{"py/type": f"{__name__}.Level"}, {"py/tuple": [1]}]},
            ),
            (
                PurePosixPath("a", "b"),
                {
                    "py/reduce": [
                        {"py/type": "pathlib.PurePosixPath"},
                        {"py/tuple": ["a/b"]},
                    ],
                },
            ),
            (
                Items([1, 2]),
                {
                    "py/reduce": [
                        {"py/function": "copyreg.__newobj__"},
                        {"py/tuple": [{"py/type": f"{__name__}.Items"}]},
                        None,
                        [1, 2],
                    ],
                },
            ),
            (_STATE_TUPLE, _STATE_TUPLE_TREE),
            (SENTINEL, {"py/type": f"{__name__}.SENTINEL"}),
        ],
    )
    def test_reduce_writes_a_canonical_recipe(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    def test_a_stateless_hashable_reduce_is_written_in_full(self) -> None:
        value = Recipe((Recipe, (1,), None))
        tree = {"py/reduce": [{"py/type": f"{__name__}.Recipe"}, {"py/tuple": [1]}]}

        assert to_plain([value, value]) == [tree, tree]

    def test_a_partial_encodes_the_same_once_its_dict_was_read(self) -> None:
        # CPython creates a partial's ``__dict__``, empty, the first time it is read.
        bound = partial(max, 0)
        before = to_plain(bound)
        _ = bound.__dict__

        assert to_plain(bound) == before

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ({"b", "a", "c"}, {"py/set": ["a", "b", "c"]}),
            (set(_RANKED), {"py/set": _RANKED_TREES}),
            (frozenset(_RANKED), {"py/frozenset": _RANKED_TREES}),
            (
                {Ranked(z=1, a=1, label="b"), Ranked(z=2, a=2, label="a")},
                {"py/set": [_ranked(1, 1, "b"), _ranked(2, 2, "a")]},
            ),
        ],
    )
    def test_set_members_sort_by_encoding_then_repr(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    def test_leaf_set_members_order_by_their_json_text(self) -> None:
        # JSON escapes ``é`` to ``\u00e9``, which sorts before ``z``; its repr
        # sorts after.
        assert to_plain({"z", "\u00e9"}) == {"py/set": ["\u00e9", "z"]}

    def test_set_members_order_by_encoding_not_by_address(self) -> None:
        # A default ``repr`` holds the object's address, which differs per run.
        trees = {
            json.dumps(to_plain({Faceless(n) for n in (3, 1, 2)})) for _ in range(8)
        }

        assert len(trees) == 1
        assert to_plain({Faceless(2), Faceless(1)}) == {
            "py/set": [
                {"py/object": f"{__name__}.Faceless", "n": 1},
                {"py/object": f"{__name__}.Faceless", "n": 2},
            ],
        }

    def test_set_members_tie_on_their_encoding_with_sorted_keys(self) -> None:
        tree = to_plain({Alpha(a=2), Zed(a=1)})

        assert tree == {
            "py/set": [
                {"py/object": f"{__name__}.Zed", "a": 1},
                {"py/object": f"{__name__}.Alpha", "a": 2},
            ],
        }

    def test_a_cycle_through_a_set_member_is_a_reference(self) -> None:
        owner: set[object] = set()
        owner.add(Member(owner=owner))

        assert to_plain(owner) == {
            "py/set": [{"py/object": f"{__name__}.Member", "owner": {"py/ref": []}}],
        }

    def test_a_failed_reduce_is_referenced_where_it_sits(self) -> None:
        shared = LocalReduce(1, 2)

        tree = to_plain([[shared], shared])

        assert tree == [
            [{"py/object": f"{__name__}.LocalReduce", "a": 1, "b": 2}],
            {"py/ref": [0, 0]},
        ]

    def test_named_zone_is_kept(self) -> None:
        moment = dt.datetime(2026, 1, 2, 3, tzinfo=ZoneInfo("America/Chicago"))

        assert to_plain(moment) == {
            "py/datetime": "2026-01-02T03:00:00-06:00[America/Chicago]",
        }

    def test_a_local_class_cannot_be_named(self) -> None:
        class Local:
            pass

        with pytest.raises(TypeError, match="<locals>"):
            to_plain(Local)

    @pytest.mark.parametrize(
        ("value", "name"),
        [
            (threading.Lock(), "lock"),
            (Recipe((Recipe,)), "Recipe"),
            (Recipe((Recipe, (1,), None, None, None, None)), "Recipe"),
            (Recipe((1, ())), "Recipe"),
            (Recipe((Recipe, [1])), "Recipe"),
            (Recipe(None), "Recipe"),
        ],
    )
    def test_an_unknown_leaf_names_the_hook_to_add(
        self,
        value: object,
        name: str,
    ) -> None:
        with pytest.raises(TypeError, match=_no_leaf(name)):
            to_plain(value)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            1.5,
            math.inf,
            -math.inf,
            pytest.param(b"\x00\xffbytes", id="bytes"),
            PurePosixPath("/a/b"),
            Path("/a/b"),
            UUID(int=7),
            dt.datetime(2026, 9, 26, 12, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 2, 3, tzinfo=ZoneInfo("Europe/Paris")),
            (1, (2, 3)),
            {1, 2},
            frozenset({1, 2}),
            {"k": [1, {"j": 2}]},
            {(1, 2): "tuple key", 3: "int key", "py/x": "tag key"},
            MappingProxyType({"a": 1}),
            1 + 2j,
            Color.BLUE,
            Point(x=3, y=4.5),
            max,
            Path,
        ],
        ids=repr,
    )
    def test_values_survive(self, value: object) -> None:
        restored = _round_trip(value)

        assert restored == value
        assert type(restored) is type(value)

    def test_a_key_repeating_one_object_writes_it_in_full_each_time(self) -> None:
        point = Point(x=1)
        whole = to_plain(point)

        tree = to_plain({(point, point): 1})

        assert tree == {"json://" + json.dumps({"py/tuple": [whole, whole]}): 1}
        restored = cast(
            "dict[tuple[Point, Point], int]",
            _round_trip({(point, point): 1}),
        )
        assert restored == {(point, point): 1}
        first, second = next(iter(restored))
        assert first is not second

    def test_a_key_refers_to_a_value_written_before_it(self) -> None:
        point = Point(x=1)
        value = {"first": [point], (point, point): 1}
        ref = {"py/ref": ["first", 0]}

        tree = to_plain(value)

        assert isinstance(tree, dict)
        assert list(tree) == [
            "first",
            "json://" + json.dumps({"py/tuple": [ref, ref]}),
        ]
        restored = cast("dict[object, object]", _round_trip(value))
        assert restored == value
        key = next(key for key in restored if key != "first")
        first = cast("list[Point]", restored["first"])
        assert cast("tuple[Point, Point]", key)[0] is first[0]

    def test_a_key_holding_a_cycle_is_refused(self) -> None:
        looped = Member(owner=set[object]())
        looped.owner.add(looped)

        with pytest.raises(TypeError, match=r"^Cannot serialize a dict key holding"):
            to_plain({looped: 1})

    def test_nan_survives(self) -> None:
        restored = _round_trip(math.nan)

        assert isinstance(restored, float)
        assert math.isnan(restored)

    def test_an_ordered_mapping_keeps_its_order(self) -> None:
        restored = _round_trip(OrderedDict([("b", 1), ("a", 2)]))

        assert isinstance(restored, OrderedDict)
        items = cast("OrderedDict[str, int]", restored).items()
        assert list(items) == [("b", 1), ("a", 2)]

    def test_a_plain_object_restores_its_attributes(self) -> None:
        restored = _round_trip(Bare(3))

        assert isinstance(restored, Bare)
        assert restored.x == 3

    def test_slots_and_dict_members_both_restore(self) -> None:
        restored = _round_trip(Slotted(1, 2))

        assert isinstance(restored, Slotted)
        assert (restored.a, restored.b) == (1, 2)

    def test_setstate_owns_the_restored_state(self) -> None:
        restored = _round_trip(Stateful(4))

        assert isinstance(restored, Stateful)
        assert restored.value == 40

    def test_shared_objects_stay_shared(self) -> None:
        shared = {"a": 1}

        restored = _round_trip([shared, (shared,), Node(name="n")])

        assert isinstance(restored, list)
        assert restored[1][0] is restored[0]

    @pytest.mark.parametrize(
        ("shared", "target"),
        [
            (bytearray(b"abc"), list[bytearray]),
            (MappingProxyType({"a": 1}), list[MappingProxyType[str, int]]),
        ],
        ids=repr,
    )
    def test_a_shared_identity_value_stays_shared(
        self,
        shared: object,
        target: object,
    ) -> None:
        tree = to_plain([shared, shared])

        untyped = from_plain(tree, object, allow_imports=True)
        typed = from_plain(tree, target, allow_imports=True)

        assert isinstance(untyped, list)
        assert untyped[1] is untyped[0]
        assert isinstance(typed, list)
        assert typed[1] is typed[0]
        assert typed[0] == shared

    def test_a_mappingproxy_cycle_survives(self) -> None:
        inner: dict[str, object] = {}
        proxy = MappingProxyType(inner)
        inner["self"] = proxy

        restored = _round_trip(proxy)

        assert isinstance(restored, MappingProxyType)
        assert restored["self"] is restored

    def test_a_dataclass_cycle_survives(self) -> None:
        parent = Node(name="p")
        parent.children.append(parent)

        restored = _round_trip(parent)

        assert isinstance(restored, Node)
        assert restored.children[0] is restored

    def test_a_non_init_slot_is_state_too(self) -> None:
        value = Flagged(name="n")
        value._done = True

        tree = to_plain(value)
        restored = _round_trip(value)

        assert tree == {"py/object": f"{__name__}.Flagged", "name": "n", "_done": True}
        assert isinstance(restored, Flagged)
        assert restored._done is True

    def test_hooks_encode_and_decode_leaves(self) -> None:
        tree = to_plain([Tensor([1.0, math.inf])], hooks=_HOOKS)

        restored = from_plain(tree, object, hooks=_HOOKS, allow_imports=True)

        assert tree == [
            {"py/hook": [f"{__name__}.Tensor", [1.0, {"py/float": "inf"}]]},
        ]
        assert isinstance(restored, list)
        assert isinstance(restored[0], Tensor)
        assert restored[0].values == [1.0, math.inf]

    def test_a_hook_returning_the_type_it_encodes_is_rejected(self) -> None:
        calls: list[Tensor] = []

        def encode(tensor: Tensor) -> Tensor:
            calls.append(tensor)
            return Tensor(tensor.values)

        hooks: Hooks = {Tensor: (encode, Tensor)}

        with pytest.raises(
            TypeError,
            match=r"Tensor: its encode hook returned a Tensor\.$",
        ):
            to_plain([Tensor([1.0])], hooks=hooks)
        assert len(calls) == 1

    def test_a_private_slot_round_trips_under_its_mangled_name(self) -> None:
        tree = to_plain(Private(secret=7, shown=1))

        restored = _round_trip(Private(secret=7, shown=1))

        assert tree == {
            "py/object": f"{__name__}.Private",
            "_Private__secret": 7,
            "shown": 1,
        }
        assert isinstance(restored, Private)
        assert restored.secret() == 7

    def test_immutable_trees_decode_alike(self) -> None:
        value = {"a": (1, {2}), "b": Point(x=1)}

        tree = to_plain(value, mutable=False)

        assert from_plain(tree, object, allow_imports=True) == value

    def test_a_protocol_field_reads_by_tag(self) -> None:
        tree = to_plain(Holder(shape=Square(side=2.0)))

        restored = from_plain(tree, Holder, allow_imports=True)

        assert restored == Holder(shape=Square(side=2.0))


class TestEncodeEdges:
    def test_a_shared_set_member_object_stays_shared(self) -> None:
        first = Point(x=1)

        restored = _round_trip([{first, Point(x=2)}, first])

        assert isinstance(restored, list)
        assert restored[1] in restored[0]

    def test_a_bound_method_round_trips_by_reduce(self) -> None:
        restored = _round_trip([1].append)

        assert callable(restored)
        restored(2)
        assert getattr(restored, "__self__", None) == [1, 2]

    def test_a_failed_reduce_falls_back_to_state(self) -> None:
        restored = _round_trip(LocalReduce(1, 2))

        assert isinstance(restored, LocalReduce)
        assert (restored.a, restored.b) == (1, 2)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Unreducible(a=1), {"a": 1}),
            (UnreducibleList([1]), [1]),
            (UnreducibleSet({1}), {"py/set": [1]}),
            (
                Corner.A,
                {
                    "py/object": f"{__name__}.Corner",
                    "__objclass__": {"py/type": f"{__name__}.Corner"},
                    "_name_": "A",
                    "_sort_order_": 0,
                    "_value_": {"py/object": f"{__name__}.Coord", "x": 1},
                    "x": 1,
                },
            ),
            (Shadowed([1]), {"py/object": f"{__name__}.Shadowed", "items": [1]}),
        ],
    )
    def test_an_unreducible_value_encodes_by_kind(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    @pytest.mark.parametrize("dialect", ["json", "python"])
    def test_a_nested_float_subclass_encodes_as_it_does_alone(
        self,
        dialect: Dialect,
    ) -> None:
        value = Celsius(1.5)

        assert to_plain([value], dialect=dialect) == [to_plain(value, dialect=dialect)]
        assert to_plain({"t": value}, dialect=dialect) == {
            "t": to_plain(value, dialect=dialect),
        }

    def test_a_nested_float_subclass_takes_its_hook(self) -> None:
        hooks: Hooks = {Celsius: (float, Celsius)}

        assert to_plain([Celsius(1.5)], hooks=hooks) == [
            {"py/hook": [f"{__name__}.Celsius", 1.5]},
        ]

    def test_a_part_failing_both_ways_falls_back_to_the_outer_reduce(self) -> None:
        assert to_plain(Carrier([1], cargo=LocalOnly())) == [1]

    def test_a_named_reduce_has_no_plain_form(self) -> None:
        value = NamedReduce()

        with pytest.raises(
            TypeError,
            match=_unresolved(value, f"{__name__}.NamedReduce"),
        ):
            to_plain(value)

    @pytest.mark.parametrize(
        ("module", "path"),
        [
            ("builtins", "builtins.max"),
            ("no_such_module_xyz", "no_such_module_xyz.max"),
        ],
    )
    def test_an_import_path_not_naming_the_object_is_rejected(
        self,
        module: str,
        path: str,
    ) -> None:
        def shadow() -> None:
            pass

        shadow.__qualname__ = "max"
        shadow.__module__ = module

        with pytest.raises(TypeError, match=_unresolved(shadow, path)):
            to_plain(shadow)


class TestImportPath:
    @pytest.mark.parametrize(
        "method",
        [str.maketrans, bytes.maketrans, bytearray.maketrans],
        ids=["str", "bytes", "bytearray"],
    )
    def test_a_static_builtin_method_round_trips(
        self,
        method: Callable[..., object],
    ) -> None:
        assert _round_trip(method) == method

    def test_a_local_function_names_why_it_cannot_round_trip(self) -> None:
        def local() -> None:
            pass

        with pytest.raises(TypeError) as raised:
            serialize._import_path(local)

        assert str(raised.value) == (
            f"Cannot serialize {local!r}: it has no importable path "
            "(module-level __qualname__). Local/lambda callables and local "
            "classes/subclasses cannot be deserialized."
        )

    @pytest.mark.parametrize(
        "names",
        [
            {},
            {"__module__": "no_such_module_xyz", "__qualname__": 1},
            {"__module__": 1, "__qualname__": "x"},
        ],
        ids=repr,
    )
    def test_a_non_text_module_or_qualname_is_rejected(
        self,
        names: dict[str, object],
    ) -> None:
        callee = Callee()
        for name, value in names.items():
            object.__setattr__(callee, name, value)

        with pytest.raises(TypeError, match="no importable path"):
            serialize._import_path(callee)

    def test_a_value_with_no_module_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="no importable path"):
            serialize._import_path(cast("type", object()))

    def test_a_path_is_the_module_and_qualname(self) -> None:
        assert serialize._import_path(Point) == f"{__name__}.Point"
        assert serialize._import_path(Node.__init__) == f"{__name__}.Node.__init__"

    def test_a_path_is_cached_for_the_same_object(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache: dict[int, tuple[object, str]] = {}
        monkeypatch.setattr(serialize, "_IMPORT_PATHS", cache)

        path = serialize._import_path(Point)

        assert cache == {id(Point): (Point, path)}
        # An alias the module also binds names the same object, so it is kept.
        monkeypatch.setitem(globals(), "PointAlias", Point)
        cache[id(Point)] = (Point, f"{__name__}.PointAlias")
        assert serialize._import_path(Point) == f"{__name__}.PointAlias"

    def test_a_cached_path_no_longer_naming_its_object_is_recomputed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = {id(Point): (cast("object", Point), "cached.Path")}
        monkeypatch.setattr(serialize, "_IMPORT_PATHS", cache)

        assert serialize._import_path(Point) == f"{__name__}.Point"
        assert cache == {id(Point): (Point, f"{__name__}.Point")}

    def test_a_path_rebound_to_another_object_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        value = Moved()
        assert to_plain(value) == {"py/reduce": _newobj(Moved)}

        monkeypatch.setitem(globals(), "Moved", Bare)

        with pytest.raises(TypeError, match=_no_leaf("Moved")):
            to_plain(value)

    def test_a_rebound_dataclass_path_is_rejected_once_encoded_before(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Encoded several times first, so its path is kept with its facts.
        value = Point(x=1)
        for _ in range(3):
            assert to_plain(value) == {
                "py/object": f"{__name__}.Point",
                "x": 1,
                "y": 0.0,
            }

        monkeypatch.setitem(globals(), "Point", Node)

        with pytest.raises(
            TypeError,
            match=_unresolved(type(value), f"{__name__}.Point"),
        ):
            to_plain(value)

    def test_a_rebound_nested_class_path_is_rejected_once_encoded_before(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The name is rebound on the outer class, not on the module.
        value = Nest.Inner()
        for _ in range(3):
            assert to_plain(value) == {"py/object": f"{__name__}.Nest.Inner", "a": 1}

        monkeypatch.setattr(Nest, "Inner", Bare)

        with pytest.raises(
            TypeError,
            match=_unresolved(type(value), f"{__name__}.Nest.Inner"),
        ):
            to_plain(value)

    def test_a_rebound_nested_function_path_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        function = Nest.method_like
        for _ in range(3):
            assert to_plain(function) == {
                "py/function": f"{__name__}.Nest.method_like",
            }

        monkeypatch.setattr(Nest, "method_like", _fake_encoder)

        with pytest.raises(
            TypeError,
            match=_unresolved(function, f"{__name__}.Nest.method_like"),
        ):
            to_plain(function)

    def test_a_reused_id_holding_another_object_is_recomputed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(serialize, "_IMPORT_PATHS", {id(Point): (Node, "stale")})

        assert serialize._import_path(Point) == f"{__name__}.Point"


class TestKeyHelpers:
    def test_a_plain_key_is_written_as_itself(self) -> None:
        state = serialize._Encoding(dialect="json", hooks={})

        assert serialize._encode_key("name", state) == "name"
        assert serialize._encode_key("py/x", state) == 'json://"py/x"'

    def test_a_place_outside_every_key_encloses_nothing(self) -> None:
        place: serialize._Here = (None, "a")

        assert serialize._encloses(place, ((None, "b"), 0)) is False
        assert serialize._encloses(place, (place, 0)) is True


_FROZEN: Final = MappingProxyType[type, type](
    {Mapping: MappingProxyType, Sequence: tuple},
)


def _no_json() -> Mapping[str, PlainTree]:
    return MappingProxyType[str, PlainTree]({})


@dataclass(frozen=True, kw_only=True, slots=True)
class Record:
    """Provider JSON in ``extra``, beside fields declared otherwise."""

    extra: Mapping[str, PlainTree] = field(default_factory=_no_json)
    names: Sequence[str] = ()
    pairs: tuple[int, ...] = ()
    loose: object = None


@dataclass(frozen=True, kw_only=True, slots=True)
class Extended(Record):
    """A record read through its base's declared type, untyped."""

    more: Sequence[int] = ()


class Tags(TypedDict):
    tags: Sequence[str]


def _no_records() -> Mapping[str, Record]:
    return MappingProxyType[str, Record]({})


@dataclass(frozen=True, kw_only=True, slots=True)
class Shelf:
    """Fields whose unions, TypedDicts, and records declared types reach through."""

    either: Mapping[str, int] | Sequence[int] | None = None
    either_way: Mapping[str, Sequence[int]] | dict[str, Sequence[int]] | None = None
    split: dict[str, int] | Mapping[str, int] | None = None
    pair: tuple[Mapping[str, int], Sequence[int]] | None = None
    base: Record | None = None
    tags: Tags | None = None
    records: tuple[Record, ...] = ()
    by_name: Mapping[str, Record] = field(default_factory=_no_records)
    plain: dict[str, int] | dict[str, str] | None = None
    record_or_map: Mapping[str, object] | Record | None = None
    members: AbstractSet[Sequence[int] | bool] = frozenset()


def _record_tree(*, tagged: bool, **fields: MutablePlainTree) -> MutablePlainTree:
    """Return a ``Record``'s tree under ``_FROZEN``, ``fields`` replacing."""
    tag: dict[str, MutablePlainTree] = (
        {"py/object": f"{__name__}.Record"} if tagged else {}
    )
    defaults: dict[str, MutablePlainTree] = {
        "extra": {},
        "loose": None,
        "names": [],
        "pairs": [],
    }
    return {**tag, **defaults, **fields}


def _kinds(value: object) -> object:
    """Return ``value`` with each container shown as its type and its parts' kinds."""
    kind: type = type(value)
    if is_dataclass(value) and not isinstance(value, type):
        return (
            kind,
            {
                item.name: _kinds(cast("object", getattr(value, item.name)))
                for item in fields(value)
            },
        )
    if isinstance(value, Mapping):
        entries = cast("Mapping[object, object]", value)
        return kind, {key: _kinds(item) for key, item in entries.items()}
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        return kind, [_kinds(item) for item in items]
    return value


def _concrete_round_trip(value: object, target: object) -> object:
    """Return ``value`` written and read back as ``target`` under ``_FROZEN``."""
    tree = to_plain(value, target=target, default_concrete_types=_FROZEN)
    return from_plain(
        tree,
        target,
        allow_imports=True,
        default_concrete_types=_FROZEN,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class Pins:
    """Fields whose declared types imply a non-JSON container."""

    names: tuple[str, ...] = ()
    pair: tuple[int, str] = (0, "")
    tags: frozenset[str] = frozenset()
    loose: object = None


@dataclass(frozen=True, kw_only=True, slots=True)
class PinsHolder:
    """A record reached through fields that declare nothing about it."""

    inner: Pins | None = None
    loose: object = None


@dataclass(frozen=True, kw_only=True, slots=True)
class Loose:
    """A record that declares nothing about its fields."""

    held: object = None


# ``Hooks`` is imported for type checking only, so this alias's value raises
# ``NameError`` when evaluated, as one naming ``torch.nn`` there would.
type _LateHooks = list[Hooks]


@dataclass(frozen=True, kw_only=True, slots=True)
class LateHinted:
    """A field whose alias does not evaluate beside one whose type implies."""

    hooks: _LateHooks
    names: tuple[str, ...] = ()


class TestRecordImplies:
    def test_a_class_outside_classes_is_not_written(self) -> None:
        with pytest.raises(TypeError, match="not in classes"):
            to_plain(Point(x=1), classes={"square": Square})

    def test_a_field_whose_alias_does_not_evaluate_reads_by_its_tags(self) -> None:
        value = LateHinted(hooks=[], names=("a",))

        tree = to_plain(value)

        assert tree == {
            "py/object": f"{__name__}.LateHinted",
            "hooks": [],
            "names": ["a"],
        }
        assert from_plain(tree, object, allow_imports=True) == value


@dataclass(frozen=True, kw_only=True, slots=True)
class Doubles:
    """Stores twice what it is built with, so a rebuild shows as a second doubling."""

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value * 2)


@dataclass(frozen=True, kw_only=True, slots=True)
class Nest2:
    """Records reached through fields that declare them, or declare less."""

    record: Record | None = None
    either: Mapping[str, object] | Record | None = None
    loose: object = None


class TestADeclaredRecordIsUntagged:
    @pytest.mark.parametrize(
        ("value", "target", "tagged"),
        [
            (Point(x=1), Point, False),
            (Point(x=1), Point | None, False),
            (Point(x=1), object, True),
            (Extended(more=(1,)), Record, True),
            (Square(side=2.0), Shape, True),
        ],
        ids=repr,
    )
    def test_the_tag_is_written_only_where_the_declared_type_does_not_imply_it(
        self,
        value: object,
        target: object,
        tagged: bool,
    ) -> None:
        tree = to_plain(value, target=target)

        assert isinstance(tree, dict)
        assert ("py/object" in tree) is tagged
        assert from_plain(tree, target, allow_imports=True) == value

    @pytest.mark.parametrize("target", [int, int | str], ids=repr)
    def test_a_record_its_declared_type_cannot_read_keeps_its_tag(
        self,
        target: object,
    ) -> None:
        tree = to_plain(Point(x=1), target=target)

        assert tree == {"py/object": f"{__name__}.Point", "x": 1, "y": 0.0}

    def test_a_record_field_declaring_its_class_is_untagged(self) -> None:
        value = Nest2(
            record=Record(names=("a",)),
            either=Record(names=("b",)),
            loose=Record(names=("c",)),
        )

        tree = to_plain(value)

        assert isinstance(tree, dict)
        tags = {
            name: part.get("py/object")
            for name, part in tree.items()
            if isinstance(part, dict)
        }
        assert tags == {
            "record": None,
            "either": f"{__name__}.Record",
            "loose": f"{__name__}.Record",
        }
        assert from_plain(tree, object, allow_imports=True) == value

    def test_an_untagged_record_restores_its_state(self) -> None:
        value = Doubles(value=1)

        tree = to_plain(value, target=Doubles)

        assert tree == {"value": 2}
        assert from_plain(tree, Doubles) == value

    def test_an_untagged_record_cycle_survives(self) -> None:
        parent = Node(name="p")
        parent.children.append(parent)

        tree = to_plain(parent, target=Node)
        restored = from_plain(tree, Node)

        assert tree == {"children": [{"py/ref": []}], "name": "p"}
        assert restored.children[0] is restored


def _no_proxy() -> MappingProxyType[str, list[int]]:
    return MappingProxyType[str, list[int]]({})


@dataclass(frozen=True, kw_only=True, slots=True)
class Proxied:
    """A field declared exactly ``MappingProxyType``, beside one declaring nothing."""

    fixed: MappingProxyType[str, list[int]] = field(default_factory=_no_proxy)
    loose: object = None


_EXACT: Final[list[tuple[object, object]]] = [
    ({"k": [1]}, dict[str, list[int]]),
    ([[1]], list[list[int]]),
    (([1],), tuple[list[int], ...]),
    (([1],), tuple[list[int]]),
    ({(1,)}, set[tuple[int]]),
    (frozenset({(1,)}), frozenset[tuple[int]]),
    (MappingProxyType({"k": [1]}), MappingProxyType[str, list[int]]),
]
"""A value of each container class, and that class declared exactly."""


class TestADeclaredContainerIsUntagged:
    @pytest.mark.parametrize(("value", "target"), _EXACT, ids=repr)
    def test_a_container_of_its_declared_class_is_written_untagged(
        self,
        value: object,
        target: object,
    ) -> None:
        tree = to_plain(value, target=target)
        restored = from_plain(tree, target)

        assert isinstance(tree, (list, dict))
        assert not any(str(key).startswith("py/") for key in tree)
        assert restored == value
        assert type(restored) is type(value)

    # A set's members are hashable, so none is a value a reference names.
    @pytest.mark.parametrize(
        ("value", "target", "step"),
        [
            (value, target, "k" if isinstance(value, Mapping) else 0)
            for value, target in _EXACT
            if not isinstance(value, AbstractSet)
        ],
        ids=repr,
    )
    def test_a_reference_into_it_names_its_own_place(
        self,
        value: object,
        target: object,
        step: str | int,
    ) -> None:
        held = cast("Mapping[str, list[int]] | Sequence[list[int]]", value)
        inner = held["k"] if isinstance(held, Mapping) else held[0]
        declared = GenericAlias(tuple, (target, list[int]))

        tree = to_plain((value, inner), target=declared)
        restored = cast("tuple[object, ...]", from_plain(tree, declared))

        assert tree == [to_plain(value, target=target), {"py/ref": [0, step]}]
        first = cast("Mapping[str, list[int]] | Sequence[list[int]]", restored[0])
        assert restored[1] is (first["k"] if isinstance(first, Mapping) else first[0])


_MISDECLARED: Final[list[tuple[object, object, PlainTree]]] = [
    ((1,), tuple[int, str] | list[int], {"py/tuple": [1]}),
    ([1], Literal["a"] | tuple[int, ...], {"py/list": [1]}),
    ([1], type[int] | tuple[int, ...], {"py/list": [1]}),
    ([1], tuple[int, ...] | object, {"py/list": [1]}),
]
"""A container, a declared union it is no member of, and its tree."""


class TestAContainerNotOfItsDeclaredType:
    @pytest.mark.parametrize(("value", "target", "expected"), _MISDECLARED, ids=repr)
    def test_it_is_tagged(
        self,
        value: object,
        target: object,
        expected: PlainTree,
    ) -> None:
        tree = to_plain(value, target=target)
        restored = from_plain(tree, object)

        assert tree == expected
        assert restored == value
        assert type(restored) is type(value)

    def test_a_fitting_tuple_judged_whole_is_untagged(self) -> None:
        target = tuple[int, str] | set[int]

        tree = to_plain((1, "a"), target=target)

        assert tree == [1, "a"]
        assert from_plain(tree, target) == (1, "a")

    def test_an_unhashable_annotation_still_untags(self) -> None:
        target = Annotated[list[int], {"unit": "m"}]

        assert to_plain([1], target=target) == [1]

    def test_decisions_start_over_once_full(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        decisions: dict[tuple[object, ...], object] = {}
        monkeypatch.setattr(serialize, "_DECISIONS", decisions)
        monkeypatch.setattr(serialize, "_DECISION_LIMIT", 1)

        first = serialize._python_to_plain([1], target=list[int])
        second = serialize._python_to_plain((1,), target=tuple[int, ...])

        assert (first, second) == ([1], [1])
        assert len(decisions) == 1


class Ratio(Enum):
    HALF = 0.5
    PAIR = (1, 2)


_LEAVES: Final[list[tuple[object, object, PlainTree]]] = [
    (Path("/a/b"), Path, "/a/b"),
    (PurePosixPath("c/d"), PurePosixPath, "c/d"),
    (UUID(int=7), UUID, "00000000-0000-0000-0000-000000000007"),
    (dt.datetime(2026, 1, 2, tzinfo=dt.UTC), dt.datetime, "2026-01-02T00:00:00+00:00"),
    (
        dt.datetime(2026, 1, 2, tzinfo=ZoneInfo("America/New_York")),
        dt.datetime,
        "2026-01-02T00:00:00-05:00[America/New_York]",
    ),
    (Color.RED, Color, "red"),
    (Level.LOW, Level, 1),
    (Ratio.HALF, Ratio, 0.5),
]
"""A leaf of each class read from untagged text, that class, and its text."""


_TAGGED_LEAVES: Final[list[tuple[object, object]]] = [
    *((value, object) for value, _, _ in _LEAVES),
    (Path("/a/b"), str | Path),
    (PurePosixPath("c/d"), str | PurePosixPath),
    (UUID(int=7), str | UUID),
    (dt.datetime(2026, 1, 2, tzinfo=dt.UTC), str | dt.datetime),
    (Color.RED, str | Color),
    (Level.LOW, int | Level),
    (Ratio.HALF, float | Ratio),
    (Decimal("1.5"), Decimal),
    (Ratio.PAIR, Ratio),
]
"""A leaf and a declared type whose untagged reading is not the leaf."""


class TestADeclaredLeafIsUntagged:
    @pytest.mark.parametrize(("value", "target", "text"), _LEAVES, ids=repr)
    def test_a_leaf_of_its_declared_class_is_written_as_its_text(
        self,
        value: object,
        target: object,
        text: PlainTree,
    ) -> None:
        tree = to_plain(value, target=target)
        restored = from_plain(tree, target)

        assert tree == text
        assert restored == value
        assert type(restored) is type(value)

    @pytest.mark.parametrize(
        ("value", "declared", "text"),
        [
            (value, declared, text)
            for (value, _, text), declared in zip(
                _LEAVES,
                [
                    (Path | None, None | Path),
                    (PurePosixPath | None, None | PurePosixPath),
                    (UUID | None, None | UUID),
                    (dt.datetime | None, None | dt.datetime),
                    (dt.datetime | None, None | dt.datetime),
                    (Color | None, None | Color),
                    (Level | None, None | Level),
                    (Ratio | None, None | Ratio),
                ],
                strict=True,
            )
            for declared in declared
        ],
        ids=repr,
    )
    def test_an_optional_leaf_is_written_as_its_text(
        self,
        value: object,
        declared: object,
        text: PlainTree,
    ) -> None:
        assert to_plain(value, target=declared) == text
        assert from_plain(text, declared) == value

    @pytest.mark.parametrize(("value", "target"), _TAGGED_LEAVES, ids=repr)
    def test_a_leaf_another_member_would_read_keeps_its_tag(
        self,
        value: object,
        target: object,
    ) -> None:
        tree = to_plain(value, target=target)
        restored = from_plain(tree, target, allow_imports=True)

        assert isinstance(tree, dict)
        assert restored == value
        assert type(restored) is type(value)

    @pytest.mark.parametrize(("value", "target", "text"), _LEAVES, ids=repr)
    def test_a_key_of_its_declared_class_is_escaped_as_its_text(
        self,
        value: object,
        target: object,
        text: PlainTree,
    ) -> None:
        declared = GenericAlias(dict, (target, int))

        tree = to_plain({value: 1}, target=declared)
        restored = cast("dict[object, int]", from_plain(tree, declared))

        assert tree == {"json://" + json.dumps(text): 1}
        assert restored == {value: 1}
        assert type(next(iter(restored))) is type(value)


class TestRecordsImplyTheirFields:
    def test_a_field_declared_a_mappingproxy_holds_one_untagged(self) -> None:
        inner = [1]
        value = Proxied(fixed=MappingProxyType({"k": inner}), loose=inner)

        tree = to_plain(value)
        restored = from_plain(tree, object, allow_imports=True)

        assert tree == {
            "py/object": f"{__name__}.Proxied",
            "fixed": {"k": [1]},
            "loose": {"py/ref": ["fixed", "k"]},
        }
        assert isinstance(restored, Proxied)
        assert type(restored.fixed) is MappingProxyType
        assert restored.loose is restored.fixed["k"]

    def test_an_int_in_a_float_field_reads_back_as_an_int(self) -> None:
        value = Point(x=1, y=3)

        tree = to_plain(value)
        restored = from_plain(tree, object, allow_imports=True)

        assert tree == {"py/object": f"{__name__}.Point", "x": 1, "y": 3}
        assert isinstance(restored, Point)
        assert (restored.y, type(restored.y)) == (3, int)

    def test_a_fixed_tuple_not_of_its_arity_is_tagged(self) -> None:
        value = Pins(pair=cast("tuple[int, str]", cast("object", (1, "b", (2,)))))

        tree = to_plain(value)

        assert isinstance(tree, dict)
        assert tree["pair"] == {"py/tuple": [1, "b", {"py/tuple": [2]}]}
        assert from_plain(tree, Pins, allow_imports=True) == value

    def test_a_record_writes_what_its_fields_imply_untagged(self) -> None:
        value = Pins(names=("a",), pair=(1, "b"), tags=frozenset({"t"}), loose=(2,))

        assert to_plain(value) == {
            "py/object": f"{__name__}.Pins",
            "loose": {"py/tuple": [2]},
            "names": ["a"],
            "pair": [1, "b"],
            "tags": ["t"],
        }

    @pytest.mark.parametrize("field", ["inner", "loose"])
    def test_a_record_anywhere_writes_what_its_fields_imply(self, field: str) -> None:
        value = PinsHolder(**{field: Pins(names=("a",))})

        tree = to_plain(value)

        assert isinstance(tree, dict)
        inner = tree[field]
        assert isinstance(inner, dict)
        assert inner["names"] == ["a"]

    @pytest.mark.parametrize(
        "value",
        [
            Pins(names=("a",), pair=(1, "b"), tags=frozenset({"t"}), loose=(2,)),
            PinsHolder(inner=Pins(names=("a",)), loose=Pins(tags=frozenset({"t"}))),
            [Pins(names=("a",))],
        ],
        ids=repr,
    )
    def test_the_tag_alone_reads_it_back(self, value: object) -> None:
        restored = from_plain(to_plain(value), object, allow_imports=True)

        assert restored == value
        assert _kinds(restored) == _kinds(value)

    def test_concrete_types_apply_inside_a_record_with_no_target(self) -> None:
        value = Record(extra=immutable({"a": [1]}), names=("x",))

        tree = to_plain(value, default_concrete_types=_FROZEN)
        restored = from_plain(
            tree,
            object,
            allow_imports=True,
            default_concrete_types=_FROZEN,
        )

        assert tree == _record_tree(tagged=True, extra={"a": [1]}, names=["x"])
        assert _kinds(restored) == _kinds(value)


class TestDefaultConcreteTypes:
    def test_a_value_of_its_concrete_type_is_written_untagged(self) -> None:
        value = Record(extra=immutable({"a": [1, {"b": 2}]}), names=("x",))

        tree = to_plain(value, target=Record, default_concrete_types=_FROZEN)

        assert tree == _record_tree(
            tagged=False,
            extra={"a": [1, {"b": 2}]},
            names=["x"],
        )

    def test_a_dict_or_list_where_another_type_is_concrete_is_tagged(self) -> None:
        value = Record(extra={"a": [1, (2,)]}, names=["x"])

        tree = to_plain(value, target=Record, default_concrete_types=_FROZEN)

        assert tree == _record_tree(
            tagged=False,
            extra={"py/dict": {"a": {"py/list": [1, [2]]}}},
            names={"py/list": ["x"]},
        )

    @pytest.mark.parametrize(
        "value",
        [MappingProxyType({"a": (1,)}), [(1,), {"b": frozenset({2})}]],
        ids=repr,
    )
    def test_without_a_target_a_plain_root_declares_nothing(
        self,
        value: object,
    ) -> None:
        tree = to_plain(value, default_concrete_types=_FROZEN)

        assert tree == to_plain(value)
        assert _kinds(from_plain(tree, object, allow_imports=True)) == _kinds(value)

    def test_a_root_list_reads_back_as_its_target_declares(self) -> None:
        value = [Record(names=["x"])]

        restored = _concrete_round_trip(value, list[Record])

        assert _kinds(restored) == _kinds(value)

    @pytest.mark.parametrize(
        ("loose", "tree"),
        [
            (
                immutable({"a": [1]}),
                {"py/mappingproxy": {"a": {"py/tuple": [1]}}},
            ),
            ({"a": [1]}, {"a": [1]}),
        ],
        ids=["frozen", "mutable"],
    )
    def test_a_field_declared_otherwise_is_written_as_without_them(
        self,
        loose: object,
        tree: MutablePlainTree,
    ) -> None:
        value = Record(loose=loose)

        tree_written = to_plain(value, target=Record, default_concrete_types=_FROZEN)

        assert tree_written == _record_tree(tagged=False, loose=tree)
        assert to_plain(value) == {
            **cast("dict[str, MutablePlainTree]", to_plain(Record())),
            "loose": tree,
        }

    def test_without_them_a_frozen_field_keeps_its_tags(self) -> None:
        tree = to_plain(Record(extra=immutable({"a": [1]})), target=Record)

        assert isinstance(tree, dict)
        assert tree["extra"] == {"py/mappingproxy": {"a": {"py/tuple": [1]}}}

    @pytest.mark.parametrize(
        "concrete",
        [{Mapping: dict}, {Sequence: list}, {}],
        ids=repr,
    )
    def test_baseline_entries_write_as_without_them(
        self,
        concrete: Mapping[type, type],
    ) -> None:
        value = Record(extra=immutable({"a": [1]}), names=["x"], loose={"b": (2,)})

        tree = to_plain(value, target=Record, default_concrete_types=concrete)

        assert tree == to_plain(value, target=Record)

    def test_a_union_with_no_overridden_member_writes_as_without_them(self) -> None:
        value = Shelf(plain={"a": 1})

        tree = to_plain(value, target=Shelf, default_concrete_types=_FROZEN)

        assert isinstance(tree, dict)
        assert tree["plain"] == {"a": 1}

    def test_union_members_that_disagree_tag_only_what_they_disagree_on(
        self,
    ) -> None:
        value = Shelf(either_way={"a": [1], "b": (2,)})

        tree = to_plain(value, target=Shelf, default_concrete_types=_FROZEN)

        assert isinstance(tree, dict)
        assert tree["either_way"] == {"py/dict": {"a": {"py/list": [1]}, "b": [2]}}

    def test_a_record_under_another_declared_type_writes_its_own_fields(self) -> None:
        value = Shelf(either=cast("Mapping[str, int]", Record(names=["n"])))

        tree = to_plain(value, target=Shelf, default_concrete_types=_FROZEN)

        assert isinstance(tree, dict)
        assert tree["either"] == to_plain(
            Record(names=["n"]),
            default_concrete_types=_FROZEN,
        )

    def test_a_value_its_declared_type_cannot_hold_keeps_every_tag(self) -> None:
        value = Shelf(split=cast("dict[str, int]", cast("object", [1, (2,)])))

        tree = to_plain(value, target=Shelf, default_concrete_types=_FROZEN)

        assert isinstance(tree, dict)
        assert tree["split"] == {"py/list": [1, {"py/tuple": [2]}]}

    def test_a_set_orders_its_members_as_they_are_written(self) -> None:
        value = Shelf(members=frozenset({(1,), True}))

        tree = to_plain(value, target=Shelf, default_concrete_types=_FROZEN)

        assert isinstance(tree, dict)
        assert tree["members"] == [[1], True]

    def test_a_shared_untagged_value_is_one_value_read_back(self) -> None:
        shared = immutable({"a": [1]})

        restored = _concrete_round_trip(Record(extra=shared, loose=[shared]), Record)

        assert isinstance(restored, Record)
        assert isinstance(restored.extra, MappingProxyType)
        assert cast("list[object]", restored.loose)[0] is restored.extra

    @pytest.mark.parametrize(
        "value",
        [
            Record(
                extra=immutable({"a": [1, {"b": None}]}),
                names=("x",),
                pairs=(1, 2),
            ),
            Record(extra={"a": [1, {"b": 2}]}, names=["x"]),
            Record(extra=MappingProxyType({"a": [1], "b": {"c": (2,)}})),
            Record(extra=MappingProxyType({"py/x": (1,), "json://y": {}})),
            Record(loose=immutable({"a": [1]})),
            Record(loose={"a": [1], "b": (2,)}),
            Record(loose=Record(extra=immutable({"a": [1]}), names=["y"])),
            Shelf(either=MappingProxyType({"a": 1})),
            Shelf(either={"a": 1}),
            Shelf(either=(1, 2)),
            Shelf(either=[1, 2]),
            Shelf(either_way={"a": [1], "b": (2,)}),
            Shelf(either_way=MappingProxyType({"a": [1]})),
            Shelf(split=MappingProxyType({"a": 1})),
            Shelf(split={"a": 1}),
            Shelf(base=Extended(extra=immutable({"a": [1]}), more=(1,))),
            Shelf(base=Extended(extra={"a": (1,)}, more=[1])),
            Shelf(base=Record(extra=immutable({"a": [1]}), names=["n"])),
            Shelf(pair=(MappingProxyType({"a": 1}), (1,))),
            Shelf(pair=({"a": 1}, [1])),
            Shelf(tags={"tags": ("a",)}),
            Shelf(tags={"tags": ["a"]}),
            Shelf(
                records=(Record(extra=immutable({"a": [1]})),),
                by_name=MappingProxyType({"r": Record(names=["n"])}),
            ),
            Shelf(by_name={"r": Record(extra={"a": (1,)})}),
            Shelf(plain={"a": 1}),
            Shelf(
                tags=cast("Tags", cast("object", MappingProxyType({"tags": ("a",)}))),
            ),
            Shelf(record_or_map=Extended(more=(1,))),
            Shelf(record_or_map={"a": (1,)}),
            Shelf(members=frozenset({(1,), True})),
        ],
        ids=repr,
    )
    def test_each_container_reads_back_as_its_own_type(self, value: object) -> None:
        restored = _concrete_round_trip(value, type(value))

        assert restored == value
        assert _kinds(restored) == _kinds(value)

    def test_an_abstract_set_may_read_as_a_set(self) -> None:
        value = Shelf(members={(1,)})
        concrete = MappingProxyType[type, type]({AbstractSet: set, Sequence: tuple})

        tree = to_plain(value, target=Shelf, default_concrete_types=concrete)
        back = from_plain(tree, Shelf, default_concrete_types=concrete)

        assert isinstance(tree, dict)
        assert tree["members"] == [[1]]
        assert type(back.members) is set

    def test_a_typing_alias_names_its_abstract_class(self) -> None:
        value = Record(extra=immutable({"a": 1}))
        concrete = cast(
            "Mapping[type, type]",
            {typing.Mapping: MappingProxyType},  # pyright: ignore[reportDeprecated] -- The deprecated alias is the input under test.
        )

        tree = to_plain(value, default_concrete_types=concrete)

        assert isinstance(tree, dict)
        assert tree["extra"] == {"a": 1}

    @pytest.mark.parametrize(
        ("concrete", "match"),
        [
            (
                {Mapping: list},
                (
                    "default_concrete_types\\[Mapping\\] must be one of "
                    "dict, mappingproxy, not <class 'list'>"
                ),
            ),
            (
                {Sequence: dict},
                (
                    "default_concrete_types\\[Sequence\\] must be one of "
                    "list, tuple, not <class 'dict'>"
                ),
            ),
            (
                {MutableMapping: MappingProxyType},
                "default_concrete_types\\[MutableMapping\\] must be one of dict",
            ),
            (
                {dict: MappingProxyType},
                (
                    "default_concrete_types keys must be one of Mapping, "
                    "MutableMapping, Sequence, MutableSequence, Set, MutableSet"
                ),
            ),
        ],
        ids=repr,
    )
    def test_a_concrete_type_it_cannot_apply_is_refused(
        self,
        concrete: Mapping[type, type],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            to_plain(1, default_concrete_types=concrete)


class Callee:
    """A callable instance: it has no ``__qualname__`` of its own."""

    def __call__(self) -> None:
        pass


@dataclass(kw_only=True, slots=True)
class Link:
    next: Link | None = None


type _Chain = list[_Chain] | int


# Deeper than ``recursion_headroom`` allows, so recursing once per level raises.
_DEPTH = 200
_HEADROOM = 100


def _deep(wrap: Callable[[object], object]) -> tuple[object, int]:
    """Return ``0`` wrapped ``_DEPTH`` deep, and that depth."""
    depth = _DEPTH
    value: object = 0
    for _ in range(depth):
        value = wrap(value)
    return value, depth


def _list(inner: object) -> object:
    return [inner]


def _dict(inner: object) -> object:
    return {"k": inner}


def _tuple(inner: object) -> object:
    return (inner,)


def _link(inner: object) -> object:
    return Link(next=cast("Link | None", inner or None))


class TestDeepTrees:
    @pytest.mark.parametrize(
        "wrap",
        [_list, _dict, _tuple],
        ids=["list", "dict", "tuple"],
    )
    def test_a_tree_deeper_than_the_recursion_limit_round_trips(
        self,
        wrap: Callable[[object], object],
    ) -> None:
        value, _ = _deep(wrap)

        with recursion_headroom(_HEADROOM):
            restored = _round_trip(value)

        assert restored == value

    def test_a_deep_dataclass_chain_round_trips_typed(self) -> None:
        value, depth = _deep(_link)

        with recursion_headroom(_HEADROOM):
            restored: Link | None = from_plain(to_plain(value), Link)

        for _ in range(depth):
            assert isinstance(restored, Link)
            restored = restored.next
        assert restored is None

    def test_a_deep_union_reads(self) -> None:
        value, _ = _deep(_list)

        with recursion_headroom(_HEADROOM):
            restored = from_plain(to_plain(value), _Chain)

        assert restored == value

    def test_a_deep_bad_leaf_is_a_read_error(self) -> None:
        tree, depth = _deep(_list)
        node = tree
        for _ in range(depth - 1):
            node = cast("list[object]", node)[0]
        cast("list[object]", node)[0] = "bad"

        with recursion_headroom(_HEADROOM), pytest.raises(ReadError) as raised:
            from_plain(tree, _Chain)

        assert len(raised.value.bad) == 1

    def test_a_deep_shared_reference_round_trips(self) -> None:
        value, _ = _deep(_list)

        with recursion_headroom(_HEADROOM):
            restored = cast("list[object]", _round_trip([value, value]))

        assert restored[0] is restored[1]


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
