"""Tests for ``treekle.serialize``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Protocol,
    SupportsIndex,
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
import weakref

import pytest

from treekle import serialize
from treekle.deserialize import from_plain
from treekle.lib.testing.resources import recursion_headroom
from treekle.serialize import to_plain
from treekle.types import ReadError


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


def _ranked(z: int, a: int, label: str = "") -> MutablePlainTree:
    return {"py/object": f"{__name__}.Ranked", "z": z, "a": a, "label": label}


# Equal reprs leave the encoded tree to break the tie. Sorting by ``a`` (the
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
                {"py/set": [_ranked(2, 2, "a"), _ranked(1, 1, "b")]},
            ),
        ],
    )
    def test_set_members_sort_by_repr_then_encoding(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

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
        cache[id(Point)] = (Point, "cached.Path")
        assert serialize._import_path(Point) == "cached.Path"

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
