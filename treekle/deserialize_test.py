"""Tests for ``treekle.deserialize``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, make_dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    NewType,
    NotRequired,
    Optional,  # pyright: ignore[reportDeprecated] -- Imported to test its pre-3.14 origin.
    Protocol,
    Required,
    SupportsIndex,
    TypedDict,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] -- Imported to test its pre-3.14 origin.
    cast,
    get_origin,
    get_type_hints,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import datetime as dt
import gc
import math
import numbers
import sys
import weakref

import pytest

from treekle import deserialize
from treekle.deserialize import from_plain
from treekle.serialize import to_plain
from treekle.types import (
    FieldPath,
    Hooks,
    Invalid,
    MutablePlainTree,
    Plain,
    PlainTree,
    ReadError,
)


if TYPE_CHECKING:
    from numbers import Rational


pytestmark = pytest.mark.usefixtures("implementation")


_T = TypeVar("_T")


def _round_trip(value: object) -> object:
    return from_plain(to_plain(value), object, allow_imports=True)


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


@dataclass(frozen=True, kw_only=True, slots=True)
class Point:
    x: int
    y: float = 0.0


@dataclass(kw_only=True, slots=True)
class Node:
    name: str
    children: list[Node] = field(default_factory=list["Node"])


@dataclass(kw_only=True, slots=True)
class HintedLater:
    point: "Rational"  # Bound only under ``TYPE_CHECKING``.
    count: int


class HintedLaterDict(TypedDict):
    point: Rational
    count: int


class Shape(Protocol):
    def area(self) -> float: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class Square:
    side: float

    def area(self) -> float:
        return self.side**2


@dataclass(kw_only=True, slots=True)
class Flagged:
    """A dataclass with a slot that is state but not an ``__init__`` field."""

    name: str
    _done: bool = field(default=False, init=False)


@dataclass(frozen=True, kw_only=True, slots=True)
class Boxed:
    color: Color


class Tensor:
    """A leaf only a hook can encode."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    __slots__ = ("values",)


def _tensor_values(tensor: Tensor) -> list[float]:
    return tensor.values


type Ints = list[int]
type IntsAgain = Ints
type _Nested = list[_Nested] | tuple[_Nested, ...] | int


_HOOKS: Hooks = MappingProxyType({Tensor: (_tensor_values, Tensor)})


class Stateful:
    """Restores reduce state only through ``__setstate__``."""

    def __init__(self) -> None:
        self.state: object = None

    def __setstate__(self, state: object) -> None:
        self.state = ("set", state)


class TwoStates:
    """Has both a ``__dict__`` and a slot, so reduce state comes as a pair."""

    __slots__ = ("__dict__", "a")

    def __init__(self) -> None:
        self.a = 0


class Abstract(ABC):
    @abstractmethod
    def area(self) -> float: ...


@dataclass(kw_only=True, slots=True)
class AbstractBox(Abstract, ABC):
    """A dataclass that is still abstract: it implements no ``area``."""

    x: int


class SlotBase:
    __slots__ = ("extra",)

    def __init__(self) -> None:
        self.extra = 0


@dataclass(kw_only=True, slots=True)
class SlotChild(SlotBase):
    """A dataclass with a slot its base declares but no field or hint names."""

    x: int


class AliasLoop:
    """Stands in for ``type A = B; type B = A``, whose ``__value__`` chain loops.

    The real spelling is legal at runtime but rejected by both type checkers.
    """

    def __init__(self) -> None:
        self.__value__: object = AliasLoopBack(self)


class AliasLoopBack:
    def __init__(self, back: AliasLoop) -> None:
        self.__value__: object = back


@dataclass(frozen=True, kw_only=True, slots=True)
class Pointer:
    """Names itself only inside a union."""

    to: Pointer | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class _RenamedOriginal:
    """Bound under another name, so its own qualname names nothing."""

    x: int


Renamed = _RenamedOriginal
del _RenamedOriginal


class Holder:
    """Holds a dataclass, as a configgle class holds its ``Config``."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Part:
        x: int


class Bundle(list[object]):
    """A list only a hook can encode."""


_BUNDLE_HOOKS: Hooks = MappingProxyType({Bundle: (list, Bundle)})


class Shelf(list[object]):
    """A list whose items a ``py/object`` record restores through one attribute."""

    @property
    def entries(self) -> list[object]:
        return list(self)

    @entries.setter
    def entries(self, items: list[object]) -> None:
        self[:] = items


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


class Unbuildable:
    """Refuses construction, so a reduce replay fails."""

    def __init__(self, x: int) -> None:
        if x < 0:
            raise ValueError("negative")
        self.x = x


@dataclass(frozen=True, kw_only=True, slots=True)
class Positive:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")


class Level(Enum):
    LOW = 1
    HIGH = 2


def _reduce(*parts: MutablePlainTree) -> MutablePlainTree:
    return {"py/reduce": list(parts)}


@dataclass(frozen=True, kw_only=True, slots=True)
class Tagged:
    pair: tuple[int, int] = (0, 0)


@dataclass(frozen=True, kw_only=True, slots=True)
class OtherTagged:
    name: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class Pair1:
    x: int = 0


@dataclass(frozen=True, kw_only=True, slots=True)
class Pair2:
    x: int = 0


class Meta(TypedDict):
    size: int
    name: NotRequired[str]


class TotalBase(TypedDict):
    a: int
    b: NotRequired[str]


class PartialMiddle(TotalBase, total=False):
    c: int
    d: Required[int]


class TotalLeaf(PartialMiddle):
    e: int
    f: NotRequired[int]


# This module's annotations are strings, which hide ``Required``/``NotRequired``
# from ``__required_keys__``; the TypedDict metaclass given objects sees them.
def _eager_typed_dict(
    name: str,
    bases: tuple[type, ...],
    hints: dict[str, object],
    *,
    total: bool,
) -> type:
    """Build a TypedDict over ``bases`` with object, not string, hints."""
    meta = cast("Callable[..., type]", type(TotalBase))
    return meta(name, bases, {"__annotations__": hints}, total=total)


EagerBase = _eager_typed_dict(
    "EagerBase",
    (),
    {"a": int, "b": NotRequired[str]},
    total=True,
)
EagerMiddle = _eager_typed_dict(
    "EagerMiddle",
    (EagerBase,),
    {"c": int, "d": Required[int]},
    total=False,
)
EagerLeaf = _eager_typed_dict(
    "EagerLeaf",
    (EagerMiddle,),
    {"e": int, "f": NotRequired[int]},
    total=True,
)


NamedInt = NewType("NamedInt", int)


class TypeAnyLookalike:
    """Shares ``typing.Any``'s name but not its module."""


TypeAnyLookalike.__qualname__ = "Any"


@dataclass(frozen=True, kw_only=True, slots=True)
class Doubled:
    """Stores twice what it is built with, so a rebuild shows as a second doubling."""

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value * 2)


class TestTypedDecode:
    def test_a_dataclass_reads_untagged_fields(self) -> None:
        assert from_plain({"x": 1, "y": 2}, Point) == Point(x=1, y=2.0)

    def test_a_dataclass_reads_its_non_init_slot(self) -> None:
        value = from_plain({"name": "n", "_done": True}, Flagged)

        assert value.name == "n"
        assert value._done is True

    def test_a_dataclass_rejects_an_attribute_it_lacks(self) -> None:
        with pytest.raises(ReadError, match=r"unknown field\(s\) \['_nope'\]"):
            from_plain({"name": "n", "_nope": True}, Flagged)

    def test_nested_generics_read(self) -> None:
        target = dict[str, list[tuple[int, str]]]

        assert from_plain({"a": [[1, "b"]]}, target) == {"a": [(1, "b")]}

    def test_typed_non_str_keys_read(self) -> None:
        assert from_plain({"1": "a"}, dict[str, str]) == {"1": "a"}
        assert from_plain({"json://1": "a"}, dict[int, str]) == {1: "a"}

    def test_an_enum_reads_its_value(self) -> None:
        assert from_plain("red", Color) is Color.RED

    def test_a_literal_keeps_bool_and_int_apart(self) -> None:
        assert from_plain(1, Literal[True, 1]) == 1
        assert from_plain(True, Literal[True, 1]) is True
        with pytest.raises(ReadError):
            from_plain(2, Literal[True, 1])

    def test_an_optional_reads_null_and_its_member(self) -> None:
        assert from_plain(None, int | None) is None
        assert from_plain(3, int | None) == 3

    def test_a_none_target_reads_only_null(self) -> None:
        # A hint spells ``NoneType`` as ``None``.
        assert from_plain(None, None) is None
        with pytest.raises(ReadError):
            from_plain(1, None)

    def test_a_typing_union_checks_its_members(self) -> None:
        # Before 3.14, ``Optional``/``Union`` have origin ``typing.Union``, not
        # ``types.UnionType``, and once read every value unchecked.
        assert from_plain(3, Optional[int]) == 3  # noqa: UP045 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.
        with pytest.raises(ReadError):
            from_plain("x", Optional[int])  # noqa: UP045 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.
        with pytest.raises(ReadError):
            from_plain("x", Union[int, bytes])  # noqa: UP007 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.

    def test_a_union_picks_the_matching_kind(self) -> None:
        assert from_plain("1", int | str) == "1"
        assert from_plain([1], str | list[int]) == [1]
        assert from_plain(1, float | int) == 1
        assert type(from_plain(1, float | int)) is int

    def test_a_union_tries_members_in_order(self) -> None:
        assert from_plain("/a", UUID | Path) == Path("/a")

    def test_a_nested_union_reads_each_node_once_per_member(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        depth = 12
        tree: MutablePlainTree = "bad"
        for _ in range(depth):
            tree = [tree]
        calls: list[object] = []

        def counted(
            data: PlainTree,
            target: object,
            at: FieldPath,
            state: deserialize._Decoding,
        ) -> object:
            calls.append(data)
            return deserialize._decode_list(data, target, at, state)

        monkeypatch.setattr(
            deserialize,
            "_DECODERS",
            MappingProxyType({**deserialize._DECODERS, list: counted}),
        )

        with pytest.raises(ReadError):
            deserialize._python_from_plain(tree, _Nested)

        assert len(calls) <= 2 * depth

    def test_a_union_retries_a_member_whose_reference_reads_differently(self) -> None:
        shared = Ints | int
        target = tuple[tuple[int, ...], shared] | tuple[list[int], shared]

        assert from_plain([[1], {"py/ref": [0]}], target) == ([1], [1])

    def test_a_float_accepts_an_int_and_a_tag(self) -> None:
        read = from_plain(1, float)

        assert read == 1.0
        assert type(read) is float
        assert from_plain({"py/float": "-inf"}, float) == -math.inf

    def test_a_pep695_alias_resolves(self) -> None:
        assert from_plain([1], Ints) == [1]

    @pytest.mark.parametrize(
        ("data", "target", "message"),
        [
            (True, int, "cannot read True as int"),
            (1.5, int, "cannot read 1.5 as int"),
            ("1", int, "cannot read '1' as int"),
            (1, str, "cannot read 1 as str"),
            (1, bool, "cannot read 1 as bool"),
            (None, int, "cannot read None as int"),
            (1, type(None), "cannot read 1 as None"),
            (1, None, "cannot read 1 as None"),
            ("x", float, "cannot read 'x' as float"),
            ({"py/float": "1.0"}, float, "cannot read '1.0' as float"),
            ([1], dict[str, int], "cannot read [1] as dict[str, int]"),
            ({"a": 1}, list[int], "cannot read {'a': 1} as list[int]"),
            ("x", list[int], "cannot read 'x' as list[int]"),
            ([1, 2], tuple[int], "expected 1 items for tuple[int], got 2"),
            ("!", bytes, "cannot read '!' as bytes: Only base64 data is allowed"),
            (1, bytes, "cannot read 1 as bytes"),
            ("not-a-uuid", UUID, "cannot read 'not-a-uuid' as UUID"),
            (1, UUID, "cannot read 1 as UUID"),
            (1, dt.datetime, "cannot read 1 as datetime"),
            ("2026-13-01", dt.datetime, "cannot read '2026-13-01' as datetime"),
            (
                "2026-01-01T00:00:00[Europe/Paris]",
                dt.datetime,
                (
                    "cannot read '2026-01-01T00:00:00[Europe/Paris]' as datetime: "
                    "timestamp is naive"
                ),
            ),
            *(
                (
                    text,
                    dt.datetime,
                    f"cannot read {text!r} as datetime: invalid named zone",
                )
                for text in (
                    "2026-01-01T00:00:00+00:00[Nowhere/Zone]",
                    "2026-01-01T00:00:00+00:00[Europe/Paris",
                    "2026-01-01T00:00:00+00:00[]",
                    "2026-01-01T00:00:00+00:00[Europe/Paris][x]",
                )
            ),
            ("green", Color, "cannot read 'green' as <enum 'Color'>"),
            (True, Level, "cannot read True as <enum 'Level'>"),
            (2, Literal[True, 1], "cannot read 2 as typing.Literal[True, 1]"),
            ([1], Point, "expected object for Point, got [1]"),
            (
                {"x": 1, "z": 2},
                Point,
                "Point: unknown field(s) ['z']; valid: ['x', 'y']",
            ),
            (
                {"py/object": f"{__name__}.Point"},
                Point,
                "cannot build Point: missing field(s) ['x']",
            ),
            ({"py/tuple": [1]}, list[int], "cannot read (1,) as list[int]"),
            (
                {"py/tuple": [1], "x": 2},
                tuple[int, ...],
                "invalid py/tuple envelope: {'py/tuple': [1], 'x': 2}",
            ),
            (1, Path, "cannot read 1 as Path"),
            (Decimal(1), int, "cannot read Decimal('1') as <class 'int'>"),
            ("x", int | float, "cannot read 'x' as int | float"),
            ("x", Annotated[int, "meta"], "cannot read 'x' as int"),
            *(
                (
                    ["a"],
                    alias,
                    "1 part(s) do not read; first at [0]: cannot read 'a' as int",
                )
                for alias in (Ints, IntsAgain, Annotated[Ints, "meta"])
            ),
        ],
        ids=repr,
    )
    def test_a_mismatch_names_why(
        self,
        data: PlainTree,
        target: object,
        message: str,
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain(data, target)

        assert str(raised.value) == message

    def test_a_bad_part_is_kept_as_invalid(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"a": 1, "b": "x"}, dict[str, int])

        assert raised.value.bad == {
            ("b",): Invalid(raw="x", reason="cannot read 'x' as int"),
        }
        assert raised.value.partial == {
            "a": 1,
            "b": Invalid(raw="x", reason="cannot read 'x' as int"),
        }

    def test_a_bad_dataclass_field_keeps_the_rest(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"x": "one", "y": 2}, Point)

        partial = raised.value.partial
        assert isinstance(partial, Point)
        assert partial.y == 2
        assert isinstance(partial.x, Invalid)
        assert list(raised.value.bad) == [("x",)]


class TestCapabilities:
    def test_imports_need_allow_imports(self) -> None:
        tree = to_plain(Bare(1))

        with pytest.raises(ReadError, match="allow_imports"):
            from_plain(tree, object)

    @pytest.mark.parametrize(
        "tree",
        [
            {"py/type": "pathlib.Path"},
            {"py/function": "builtins.max"},
            {"py/reduce": [{"py/type": "builtins.int"}, {"py/tuple": ["1"]}]},
        ],
        ids=repr,
    )
    def test_every_executable_tag_is_gated(self, tree: MutablePlainTree) -> None:
        with pytest.raises(ReadError, match="allow_imports"):
            from_plain(tree, object)

    def test_a_path_resolving_to_a_class_decodes(self) -> None:
        assert from_plain(
            {"py/type": "pathlib.PurePosixPath"},
            object,
            allow_imports=True,
        ) is (PurePosixPath)

    def test_an_unknown_path_is_rejected(self) -> None:
        with pytest.raises(ReadError, match="no_such_module_xyz"):
            from_plain(
                {"py/type": "no_such_module_xyz.thing"},
                object,
                allow_imports=True,
            )

    def test_import_free_tags_need_no_capability(self) -> None:
        tree = to_plain({"t": (1,), "s": {1}, "m": MappingProxyType({"a": 1})})

        assert from_plain(tree, object) == {
            "t": (1,),
            "s": {1},
            "m": MappingProxyType({"a": 1}),
        }


class TestReferences:
    def test_a_path_reference_resolves_to_the_same_object(self) -> None:
        tree = {"a": {"b": {"py/tuple": [{"k": 1}]}}, "c": {"py/ref": ["a", "b", 0]}}

        restored = from_plain(tree, object)

        assert isinstance(restored, dict)
        assert restored["c"] is restored["a"]["b"][0]

    def test_a_path_reference_closes_a_cycle(self) -> None:
        tree: MutablePlainTree = [{"py/ref": []}]

        restored = from_plain(tree, object)

        assert isinstance(restored, list)
        assert restored[0] is restored

    def test_a_legacy_index_reference_still_reads(self) -> None:
        restored = from_plain([[1], {"py/id": 1}], object)

        assert isinstance(restored, list)
        assert restored[1] is restored[0]

    def test_a_legacy_dataclass_cycle_still_reads(self) -> None:
        tree = {
            "py/object": f"{__name__}.Node",
            "name": "p",
            "children": [{"py/id": 0}],
        }

        restored = from_plain(tree, object, allow_imports=True)

        assert isinstance(restored, Node)
        assert restored.children[0] is restored

    def test_a_shared_object_inside_a_reduce_round_trips(self) -> None:
        shared = [1]
        value = OrderedDict([("a", shared), ("b", shared)])

        restored = _round_trip(value)

        assert isinstance(restored, OrderedDict)
        assert restored["a"] is restored["b"]

    def test_a_shared_hooked_leaf_round_trips(self) -> None:
        shared = Tensor([1.0])
        tree = to_plain([shared, shared], hooks=_HOOKS)

        restored = from_plain(tree, object, hooks=_HOOKS, allow_imports=True)

        assert isinstance(tree, list)
        assert tree[1] == {"py/ref": [0]}
        assert isinstance(restored, list)
        assert restored[0] is restored[1]

    def test_a_shared_set_member_round_trips(self) -> None:
        shared = Point(x=1)

        first, second = cast("list[object]", _round_trip([{shared}, shared]))

        assert isinstance(first, set)
        assert next(iter(cast("set[object]", first))) is second

    def test_a_stored_dataclass_cycle_reads_as_its_declared_type(self) -> None:
        node = Node(name="p")
        node.children.append(node)

        restored = from_plain(to_plain(node), Node)

        assert restored.name == "p"
        assert restored.children[0] is restored

    def test_a_reference_into_an_unbuilt_dataclass_is_invalid(self) -> None:
        tree: MutablePlainTree = {"name": "n", "children": [{"py/ref": []}]}

        with pytest.raises(ReadError) as raised:
            from_plain(tree, Node)

        assert list(raised.value.bad) == [("children", 0)]
        assert "still being built" in raised.value.bad["children", 0].reason

    def test_a_typed_dataclass_is_placed_for_a_later_reference(self) -> None:
        tree = {"a": {"x": 1}, "b": {"py/ref": ["a"]}}

        restored = from_plain(tree, dict[str, Point])

        assert restored["b"] is restored["a"]

    @pytest.mark.parametrize(
        "ref",
        [["missing"], [0, 0], "a", 1, [True], [1.5], None],
        ids=repr,
    )
    def test_a_bad_path_reference_is_rejected(
        self,
        ref: MutablePlainTree,
    ) -> None:
        with pytest.raises(ReadError, match="py/ref"):
            from_plain({"a": [1], "b": {"py/ref": ref}}, object)

    def test_a_reference_to_a_later_node_is_rejected(self) -> None:
        with pytest.raises(ReadError, match="py/ref"):
            from_plain({"a": {"py/ref": ["b"]}, "b": [1]}, object)

    def test_a_reference_with_extra_keys_is_rejected(self) -> None:
        with pytest.raises(ReadError):
            from_plain({"a": [1], "b": {"py/ref": ["a"], "x": 1}}, object)


class TestMalformedTrees:
    @pytest.mark.parametrize("ref", [-1, 99, True, "0"])
    def test_a_bad_reference_is_rejected(self, ref: Plain) -> None:
        with pytest.raises(ReadError, match="py/id"):
            from_plain([{"py/id": ref}], object)

    @pytest.mark.parametrize(
        "tree",
        [
            {"py/unknown": 1},
            {"py/tuple": [1], "extra": 2},
            {"py/tuple": 1},
            {"py/b64": 1},
            {"py/reduce": [1]},
            {"py/inline": [1]},
            {"py/hook": ["x.Y", 1]},
            {"json://{": 1},
            {"py/id": 0, "x": 1},
        ],
        ids=repr,
    )
    def test_malformed_tags_are_rejected(self, tree: MutablePlainTree) -> None:
        with pytest.raises(ReadError):
            from_plain(tree, object, allow_imports=True)

    def test_a_tag_must_fit_its_target(self) -> None:
        with pytest.raises(ReadError, match="as <class 'int'>"):
            from_plain({"py/tuple": [1]}, int)


class TestDecodeEdges:
    def test_bool_reads_a_bool(self) -> None:
        assert from_plain(True, bool) is True

    def test_complex_reads_a_real(self) -> None:
        assert from_plain(2, complex) == 2 + 0j

    @pytest.mark.parametrize(
        "value",
        [
            complex(math.inf, 0),
            complex(0, math.nan),
            complex(-math.inf, math.inf),
            1 + 2j,
        ],
        ids=repr,
    )
    @pytest.mark.parametrize("target", [object, complex], ids=repr)
    def test_a_complex_round_trips(self, value: complex, target: object) -> None:
        restored = from_plain(to_plain(value), target)

        assert type(restored) is complex
        assert repr(restored) == repr(value)

    def test_a_complex_part_sits_below_its_pair(self) -> None:
        part = _reduce({"py/type": "builtins.float"}, {"py/tuple": ["1.5"]}, {})
        tree: MutablePlainTree = [{"py/complex": [0.0, part]}, {"py/ref": [0, 1]}]

        assert from_plain(tree, object, allow_imports=True) == [1.5j, 1.5]

    @pytest.mark.parametrize(
        ("data", "reason"),
        [
            ({"py/complex": [True, 0.0]}, "cannot read True as float"),
            ({"py/complex": [0.0, "x"]}, "cannot read 'x' as float"),
            ({"py/complex": [1.0]}, r"cannot read [1.0] as complex"),
            ({"py/complex": "ab"}, "cannot read 'ab' as complex"),
            (True, "cannot read True as complex"),
        ],
        ids=repr,
    )
    def test_a_bad_complex_is_bad_where_the_pair_sits(
        self,
        data: MutablePlainTree,
        reason: str,
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([data], list[complex])

        assert raised.value.bad == {(0,): Invalid(raw=data, reason=reason)}

    def test_a_variadic_tuple_reads_any_length(self) -> None:
        assert from_plain([1, 2, 3], tuple[int, ...]) == (1, 2, 3)

    def test_a_variadic_tuple_reads_each_item_as_its_element_type(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([1, "x"], tuple[int, ...])

        assert list(raised.value.bad) == [(1,)]

    def test_a_sequence_target_reads_each_item_as_its_element_type(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"py/tuple": [1, "x"]}, Sequence[int])

        assert list(raised.value.bad) == [(1,)]

    def test_a_fixed_tuple_of_the_wrong_length_names_both_lengths(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([1, 2, 3], tuple[int, int])

        assert raised.value.bad[()].reason == (
            "expected 2 items for tuple[int, int], got 3"
        )

    def test_a_non_sequence_tuple_names_the_data(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain("ab", tuple[str, ...])

        assert raised.value.bad[()].reason == ("cannot read 'ab' as tuple[str, ...]")

    def test_a_whole_failure_reports_the_input_as_its_one_bad_part(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain("x", int)

        whole = Invalid(raw="x", reason="cannot read 'x' as int")
        assert str(raised.value) == "cannot read 'x' as int"
        assert raised.value.partial == whole
        assert raised.value.bad == {(): whole}

    def test_a_whole_failure_keeps_the_bad_parts_found_before_it(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"a": "x", "b": [1]}, dict[str, int] | list[int] | Point)

        assert list(raised.value.bad) == [()]

    def test_a_literal_enum_member_reads(self) -> None:
        assert from_plain(2, Literal[Level.HIGH]) is Level.HIGH

    def test_a_union_prefers_the_data_kind(self) -> None:
        assert from_plain({"a": 1}, int | dict[str, int]) == {"a": 1}

    def test_an_annotated_target_reads_its_type(self) -> None:
        assert from_plain(1, Annotated[int, "meta"]) == 1

    def test_a_typevar_target_reads_as_is(self) -> None:
        assert from_plain([1], _T) == [1]

    def test_a_tagged_value_fits_a_protocol(self) -> None:
        tree = to_plain(Square(side=1.0))

        assert from_plain(tree, Shape, allow_imports=True) == Square(side=1.0)

    def test_a_tagged_value_fits_a_literal(self) -> None:
        tree = to_plain(Color.RED)

        assert from_plain(tree, Literal[Color.RED], allow_imports=True) is Color.RED
        with pytest.raises(ReadError):
            from_plain(tree, Literal[Color.BLUE], allow_imports=True)

    def test_a_tagged_value_fits_a_typevar(self) -> None:
        assert from_plain({"py/tuple": [1]}, _T) == (1,)

    def test_reduce_items_replay(self) -> None:
        ordered = _reduce(
            {"py/type": "collections.OrderedDict"},
            {"py/tuple": []},
            None,
            None,
            [{"py/tuple": ["a", 1]}],
        )
        listed = _reduce({"py/type": "builtins.list"}, {"py/tuple": []}, None, [1, 2])

        assert from_plain(ordered, object, allow_imports=True) == OrderedDict(a=1)
        assert from_plain(listed, object, allow_imports=True) == [1, 2]

    def test_a_reference_to_an_unbuilt_dataclass_is_invalid(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"name": "n", "children": [{"py/id": 0}]}, Node)

        assert list(raised.value.bad) == [("children", 0)]
        assert "still being built" in raised.value.bad["children", 0].reason

    @pytest.mark.parametrize(
        ("data", "target"),
        [
            ("1j", complex),
            (1, UUID),
            (1, dt.datetime),
            ("2026-01-01T00:00:00+00:00[Europe/Paris", dt.datetime),
            (2, Literal[Level.LOW]),
            ("x", set[int]),
            ("x", frozenset[int]),
            ([[1]], set[object]),
            ([[1]], frozenset[object]),
            ({"json://[1]": 1}, dict[object, int]),
            ({"n": 0}, Positive),
            ({"x": 1}, Bare),
            ("x", int | float),
            ({"py/type": 1}, object),
            ({"py/type": "builtins.max"}, type),
            ({"py/type": "pathlib.NoSuchThing"}, object),
            ({"py/function": 1}, object),
            ({"py/function": "math.pi"}, object),
            ({"py/object": 1}, object),
            ({"py/object": "builtins.max"}, object),
            ({"py/reduce": 1}, object),
            (_reduce(1, {"py/tuple": []}), object),
            (
                _reduce({"py/type": f"{__name__}.Unbuildable"}, {"py/tuple": [-1]}),
                object,
            ),
            (_reduce({"py/type": "builtins.int"}, {"py/tuple": []}, None, [1]), object),
            (
                _reduce({"py/type": "builtins.int"}, {"py/tuple": []}, None, None, []),
                object,
            ),
            (
                _reduce(
                    {"py/type": "builtins.dict"},
                    {"py/tuple": []},
                    None,
                    None,
                    [1],
                ),
                object,
            ),
        ],
        ids=repr,
    )
    def test_malformed_input_raises(
        self,
        data: MutablePlainTree,
        target: object,
    ) -> None:
        with pytest.raises(ReadError):
            from_plain(data, target, allow_imports=True)

    @pytest.mark.parametrize(
        ("data", "target", "allow_imports", "match"),
        [
            ([1], Point, False, r"^expected object for Point, got \[1\]$"),
            ({"x": 1}, Bare, False, r"^cannot read \{'x': 1\} as Bare$"),
            ({"n": 0}, Positive, False, "^cannot build Positive: n must be positive$"),
            ({"py/object": 1}, object, True, "^invalid py/object path: 1$"),
            (
                {"py/object": "builtins.max"},
                object,
                True,
                "^'builtins.max' does not name a class$",
            ),
        ],
        ids=repr,
    )
    def test_an_object_that_does_not_read_names_why(
        self,
        data: MutablePlainTree,
        target: object,
        allow_imports: bool,
        match: str,
    ) -> None:
        with pytest.raises(ReadError, match=match):
            from_plain(data, target, allow_imports=allow_imports)

    def test_a_non_class_target_is_named_in_the_error(self) -> None:
        # Through ``from_plain`` the union decoder raises its own message first.
        with pytest.raises(ReadError, match=r"^expected object for .*Shape \| int"):
            deserialize._decode_object(
                [1],
                Shape | int,
                (),
                deserialize._Decoding(hooks={}, allow_imports=False),
            )

    def test_a_tagged_member_decodes_by_its_tag(self) -> None:
        tree = {"py/object": f"{__name__}.Bare", "x": {"py/tuple": [1]}}

        restored = from_plain(tree, object, allow_imports=True)

        assert isinstance(restored, Bare)
        assert restored.x == (1,)

    def test_a_dataclass_is_numbered_for_later_references(self) -> None:
        state = deserialize._Decoding(hooks={}, allow_imports=False)

        value = deserialize._decode({"x": 1}, Point, (), state)

        assert state.built == [value]

    def test_each_py_id_read_counts_one_reference(self) -> None:
        state = deserialize._Decoding(hooks={}, allow_imports=False, built=[1])

        counts = [
            (deserialize._decode_reference({"py/id": 0}, state), state.references)
            for _ in range(2)
        ]

        assert counts == [(1, 1), (1, 2)]

    def test_each_py_ref_read_counts_one_reference(self) -> None:
        state = deserialize._Decoding(
            hooks={},
            allow_imports=False,
            placed={("a",): 1},
        )

        counts = [
            (
                deserialize._decode_path_reference({"py/ref": ["a"]}, state),
                state.references,
            )
            for _ in range(2)
        ]

        assert counts == [(1, 1), (1, 2)]

    def test_a_remembered_union_failure_raises_the_same_error(self) -> None:
        state = deserialize._Decoding(hooks={}, allow_imports=False)
        data, target = "x", int | None
        with pytest.raises(ReadError):
            deserialize._decode(data, target, (), state)

        with pytest.raises(ReadError, match=r"^cannot read 'x' as int \| None$"):
            deserialize._decode(data, target, (), state)

        assert state.unread == {(id(data), id(target)): data}

    @pytest.mark.parametrize("target", [dict, Mapping], ids=["dict", "Mapping"])
    def test_a_bare_mapping_target_reads_entries_as_object(
        self,
        target: object,
    ) -> None:
        assert from_plain({"a": [1], "b": {"py/tuple": [2]}}, target) == {
            "a": [1],
            "b": (2,),
        }

    def test_a_union_failure_blocks_neither_other_data_nor_other_unions(self) -> None:
        state = deserialize._Decoding(hooks={}, allow_imports=False)
        data, target = "x", int | None
        with pytest.raises(ReadError):
            deserialize._decode(data, target, (), state)

        assert deserialize._decode(1, target, (), state) == 1
        assert deserialize._decode(data, str | None, (), state) == "x"

    def test_a_value_reads_back_as_its_type_despite_an_unresolvable_hint(self) -> None:
        value = HintedLater(point=Fraction(3, 2), count=2)

        restored = from_plain(to_plain(value), HintedLater, allow_imports=True)

        assert restored == value

    def test_an_unresolvable_hint_reads_its_part_by_tags(self) -> None:
        tree = {"point": {"py/tuple": [1]}, "count": 2}

        restored = from_plain(tree, HintedLater)

        assert (restored.point, restored.count) == ((1,), 2)


class TestTypedDict:
    def test_a_typed_dict_reads_its_fields(self) -> None:
        assert from_plain({"size": 3, "name": "a"}, Meta) == {"size": 3, "name": "a"}
        assert from_plain({"size": 3}, Meta) == {"size": 3}

    def test_a_field_reads_as_its_declared_type(self) -> None:
        with pytest.raises(ReadError):
            from_plain({"size": "3"}, Meta)
        assert from_plain({"size": "3"}, Meta, strict=False) == {"size": 3}

    def test_a_missing_required_field_raises(self) -> None:
        with pytest.raises(ReadError, match="size"):
            from_plain({"name": "a"}, Meta)

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (TotalBase, {"a"}),
            (PartialMiddle, {"a", "d"}),
            (TotalLeaf, {"a", "d", "e"}),
            (EagerBase, {"a"}),
            (EagerMiddle, {"a", "d"}),
            (EagerLeaf, {"a", "d", "e"}),
        ],
        ids=[
            "TotalBase",
            "PartialMiddle",
            "TotalLeaf",
            "EagerBase",
            "EagerMiddle",
            "EagerLeaf",
        ],
    )
    def test_each_class_in_a_chain_keeps_its_own_totality(
        self,
        kind: type,
        expected: set[str],
    ) -> None:
        assert deserialize._required_keys(kind) == expected

    @pytest.mark.parametrize(
        "kind",
        [TotalLeaf, EagerLeaf],
        ids=["TotalLeaf", "EagerLeaf"],
    )
    def test_a_chain_reads_only_with_every_required_key(self, kind: type) -> None:
        name = kind.__name__
        with pytest.raises(
            ReadError,
            match=rf"^{name}: missing required field\(s\) \['a', 'e'\]$",
        ):
            from_plain({"d": 1}, kind)
        assert from_plain({"a": 1, "d": 2, "e": 3}, kind) == {"a": 1, "d": 2, "e": 3}

    def test_an_unknown_field_is_dropped(self) -> None:
        assert from_plain({"size": 3, "extra": 1}, Meta) == {"size": 3}

    def test_a_non_mapping_raises(self) -> None:
        with pytest.raises(ReadError, match=r"^expected object for Meta, got \[1\]$"):
            from_plain([1], Meta)

    def test_a_shared_typed_dict_round_trips_shared(self) -> None:
        meta: Meta = {"size": 3}

        first, second = from_plain(to_plain([meta, meta]), list[Meta])

        assert first == meta
        assert second is first

    def test_a_legacy_index_reference_names_a_typed_dict(self) -> None:
        first, second = from_plain([{"size": 3}, {"py/id": 1}], list[Meta])

        assert second is first

    @pytest.mark.parametrize(
        "ref",
        [{"py/ref": []}, {"py/id": 0}],
        ids=repr,
    )
    def test_a_reference_to_a_non_typed_dict_is_refused(
        self,
        ref: MutablePlainTree,
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([{"size": 3}, ref], list[Meta])

        assert list(raised.value.bad) == [(1,)]

    def test_a_reference_missing_a_required_key_is_refused(self) -> None:
        with pytest.raises(ReadError):
            from_plain(
                {"a": {"name": "x"}, "b": {"py/ref": ["a"]}},
                tuple[dict[str, str], Meta],
            )

    def test_an_unresolvable_hint_reads_its_key_as_object(self) -> None:
        data = {"point": [1], "count": 2}

        assert from_plain(data, HintedLaterDict) == data

    def test_an_unresolvable_hint_keeps_its_key_required(self) -> None:
        with pytest.raises(
            ReadError,
            match=r"^HintedLaterDict: missing required field\(s\) \['point'\]$",
        ):
            from_plain({"count": 2}, HintedLaterDict)

    def test_an_unresolvable_eager_hint_keeps_every_other_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("annotationlib")
        module = ModuleType("_treekle_eager_hint_test")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        source = """
from typing import TypedDict

class Eager(TypedDict):
    count: int
    point: Missing
"""
        exec(  # noqa: S102 -- Native deferred annotations need a module compiled without the test file's future import.
            compile(source, "<treekle-eager-hint-test>", "exec", dont_inherit=True),
            vars(module),
        )
        eager = cast("type", vars(module)["Eager"])

        assert from_plain({"count": 2, "point": [1]}, eager) == {
            "count": 2,
            "point": [1],
        }
        with pytest.raises(ReadError, match=r"missing required field\(s\) \['point'\]"):
            from_plain({"count": 2}, eager)


class TestTypeBound:
    @pytest.mark.parametrize(
        ("data", "target"),
        [
            ({"py/type": "builtins.str"}, type[int]),
            ({"py/type": "builtins.int"}, type[Shape]),
            (str, type[int]),
            (int, type[Shape]),
        ],
        ids=repr,
    )
    def test_a_class_outside_the_bound_is_refused(
        self,
        data: object,
        target: object,
    ) -> None:
        with pytest.raises(ReadError):
            from_plain(data, target, allow_imports=True)

    @pytest.mark.parametrize(
        ("data", "target", "expected"),
        [
            ({"py/type": "builtins.bool"}, type[int], bool),
            ({"py/type": "builtins.str"}, type[int | str], str),
            (bool, type[int], bool),
            (Square, type[Shape], Square),
            (str, type[object], str),
            (str, type, str),
        ],
        ids=repr,
    )
    def test_a_class_inside_the_bound_reads(
        self,
        data: object,
        target: object,
        expected: type,
    ) -> None:
        assert from_plain(data, target, allow_imports=True) is expected


class TestReferenceFit:
    def test_a_reference_must_fit_a_protocol(self) -> None:
        tree: MutablePlainTree = [1, {"py/ref": []}]

        with pytest.raises(ReadError):
            from_plain(tree, list[Shape])

    def test_a_reference_to_a_protocol_member_reads(self) -> None:
        tree = to_plain([Square(side=1.0), Square(side=2.0)])
        assert isinstance(tree, list)
        tree.append({"py/ref": [1]})

        restored = from_plain(tree, list[Shape], allow_imports=True)

        assert restored[2] is restored[1]

    def test_a_reference_must_fit_its_element_types(self) -> None:
        with pytest.raises(ReadError):
            from_plain([["a"], {"py/ref": [0]}], tuple[list[str], list[int]])

    def test_a_reference_fitting_its_element_types_reads(self) -> None:
        first, second = from_plain(
            [[1], {"py/ref": [0]}],
            tuple[list[int], Sequence[int]],
        )

        assert second is first

    def test_a_referenced_mapping_checks_keys_and_values(self) -> None:
        good: MutablePlainTree = [{"a": 1}, {"py/ref": [0]}]
        bad_value: MutablePlainTree = [{"a": "x"}, {"py/ref": [0]}]
        bad_key: MutablePlainTree = [{"json://1": 1}, {"py/ref": [0]}]

        first, second = from_plain(good, tuple[dict[str, int], dict[str, int]])
        assert second is first
        with pytest.raises(ReadError):
            from_plain(bad_value, tuple[dict[str, str], dict[str, int]])
        with pytest.raises(ReadError):
            from_plain(bad_key, tuple[dict[int, int], dict[str, int]])

    def test_a_value_inside_a_mappingproxy_is_named_through_its_tag(self) -> None:
        tree: MutablePlainTree = [
            {"py/mappingproxy": {"k": [1]}},
            {"py/ref": [0, "py/mappingproxy", "k"]},
            {"py/ref": [0]},
        ]

        proxy, inner, again = from_plain(
            tree,
            tuple[MappingProxyType[str, list[int]], list[int], Mapping[str, list[int]]],
        )

        assert inner is proxy["k"]
        assert again is proxy
        assert from_plain(
            {"py/mappingproxy": {"k": [1, 2]}},
            MappingProxyType[str, tuple[int, int]],
        ) == MappingProxyType({"k": (1, 2)})
        with pytest.raises(ReadError):
            from_plain(
                tree,
                tuple[
                    MappingProxyType[str, list[int]],
                    list[int],
                    Mapping[str, list[str]],
                ],
            )

    def test_a_referenced_mapping_checks_nested_items(self) -> None:
        tree: MutablePlainTree = [{"a": [1]}, {"py/ref": [0]}]

        with pytest.raises(ReadError):
            from_plain(tree, tuple[dict[str, list[int]], dict[str, list[str]]])

    def test_a_referenced_list_checks_nested_items(self) -> None:
        tree: MutablePlainTree = [[[1]], {"py/ref": [0]}]

        with pytest.raises(ReadError):
            from_plain(tree, tuple[list[list[int]], list[list[str]]])

    def test_a_referenced_set_checks_its_members(self) -> None:
        tree: MutablePlainTree = [{"py/set": [1]}, {"py/ref": [0]}]

        with pytest.raises(ReadError):
            from_plain(tree, tuple[set[int], set[str]])

    @pytest.mark.parametrize(
        "tree",
        [
            _reduce({"py/type": "builtins.list"}, {"py/tuple": [["a"]]}),
            {"py/hook": [f"{__name__}.Bundle", ["a"]]},
            {"py/object": f"{__name__}.Shelf", "entries": ["a"]},
        ],
        ids=repr,
    )
    def test_a_tag_decoded_value_is_checked_item_by_item(
        self,
        tree: MutablePlainTree,
    ) -> None:
        restored = from_plain(tree, object, hooks=_BUNDLE_HOOKS, allow_imports=True)
        with pytest.raises(ReadError) as raised:
            from_plain(tree, list[int], hooks=_BUNDLE_HOOKS, allow_imports=True)

        assert restored == ["a"]
        assert str(raised.value) == f"cannot read {restored!r} as list[int]"

    @pytest.mark.parametrize(
        ("value", "target", "expected"),
        [
            ([1], int | list[int], True),
            ([1], int | list[str], False),
            (b"a", int | str, False),
            ((1, "a"), tuple[int, int], True),
            (frozenset({"a"}), frozenset[int], True),
            ({"a": 1}, Mapping[str, int], True),
            ({"a": 1}, Mapping[str, str], False),
            ({"a": [1]}, Mapping[str, list[str]], False),
            ({1: [1]}, Mapping[str, list[int]], False),
            ({(1,): 1}, Mapping[tuple[str], int], True),
            ([1], Sequence[str], False),
            ([1, "a"], list[int], False),
            ((1,), Sequence[str], True),
            ([[1]], Sequence[list[str]], False),
        ],
        ids=repr,
    )
    def test_a_value_is_checked_item_by_item(
        self,
        value: object,
        target: object,
        expected: bool,
    ) -> None:
        assert deserialize._fits(value, target) is expected

    def test_a_value_of_a_bare_generic_alias_is_checked(self) -> None:
        assert deserialize._fits([1], list) is True
        assert deserialize._fits({"a": 1}, dict) is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(Square(side=1.0), True), (Point(x=1), False), (1, False)],
        ids=repr,
    )
    def test_a_protocol_fits_by_its_members(
        self,
        value: object,
        expected: bool,
    ) -> None:
        assert deserialize._fits(value, Shape) is expected

    @pytest.mark.parametrize(
        ("value", "target", "expected"),
        [
            (1, Any, True),
            (1, object, True),
            (1, _T, True),
            (1, NamedInt, True),
        ],
        ids=repr,
    )
    def test_a_target_isinstance_cannot_judge_fits_anything(
        self,
        value: object,
        target: object,
        expected: bool,
    ) -> None:
        assert deserialize._fits(value, target) is expected

    @pytest.mark.parametrize(
        ("kind", "bound", "expected"),
        [
            (bool, int, True),
            (str, int, False),
            (str, int | str, True),
            (bytes, int | str, False),
            (str, Any, True),
            (str, _T, True),
            (Square, Shape, True),
            (int, Shape, False),
            (list, list[int], True),
        ],
        ids=repr,
    )
    def test_a_class_lies_inside_its_bound(
        self,
        kind: type,
        bound: object,
        expected: bool,
    ) -> None:
        assert deserialize._is_subclass(kind, bound) is expected

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [(Any, True), (TypeAnyLookalike, False), (object, False), (int, False)],
        ids=repr,
    )
    def test_only_typing_any_is_any(self, kind: type, expected: bool) -> None:
        assert deserialize._is_any(kind) is expected

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (int | str, True),
            (Union[int, str], True),  # noqa: UP007 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.
            (Optional[int], True),  # noqa: UP045 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.
            (list[int], False),
            (int, False),
        ],
        ids=repr,
    )
    def test_both_union_spellings_are_unions(
        self,
        target: object,
        expected: bool,
    ) -> None:
        assert deserialize._is_union(get_origin(target) or target) is expected

    @pytest.mark.parametrize("origin", [UnionType, Union], ids=repr)  # pyright: ignore[reportDeprecated] -- Spelling under test.
    def test_each_union_origin_class_alone_is_a_union(self, origin: object) -> None:
        # Before 3.14 these are two classes; a check needing both would read no
        # union at all there.
        assert deserialize._is_union(origin) is True


class TestStoredState:
    def test_a_typed_read_restores_stored_state_without_init(self) -> None:
        stored = Doubled(value=1)

        assert from_plain(to_plain(stored), Doubled).value == 2
        assert from_plain(to_plain(stored), Doubled, allow_imports=True).value == 2

    def test_an_untagged_record_is_built_through_init(self) -> None:
        assert from_plain({"value": 1}, Doubled).value == 2

    def test_stored_state_fills_a_missing_defaulted_field(self) -> None:
        tree = {"py/object": f"{__name__}.Point", "x": 1}

        assert from_plain(tree, Point) == Point(x=1)

    def test_stored_state_fills_a_missing_factory_field(self) -> None:
        tree = {"py/object": f"{__name__}.Node", "name": "n"}

        assert from_plain(tree, Node).children == []

    def test_stored_state_missing_a_required_field_is_refused(self) -> None:
        with pytest.raises(ReadError, match=r"missing field\(s\) \['x'\]"):
            from_plain({"py/object": f"{__name__}.Point"}, Point)


class TestDefault:
    def test_none_reads_as_the_default(self) -> None:
        assert from_plain(None, int, default=7) == 7
        assert from_plain(None, dict[str, int], default={}) == {}

    def test_a_present_value_ignores_the_default(self) -> None:
        assert from_plain(3, int, default=7) == 3

    def test_a_bad_value_still_raises(self) -> None:
        with pytest.raises(ReadError):
            from_plain("x", int, default=0)

    def test_without_a_default_none_reads_as_none(self) -> None:
        with pytest.raises(ReadError):
            from_plain(None, int)
        assert from_plain(None, int | None, default=None) is None


def _bad_of(tree: MutablePlainTree, hooks: Hooks = _HOOKS) -> dict[FieldPath, str]:
    """Return each bad part's reason when ``tree`` reads as ``object``."""
    with pytest.raises(ReadError) as raised:
        from_plain(tree, object, hooks=hooks, allow_imports=True)
    return {at: bad.reason for at, bad in raised.value.bad.items()}


_NO_ARGS: MutablePlainTree = {"py/tuple": []}
_LIST: MutablePlainTree = {"py/type": "builtins.list"}
_DICT: MutablePlainTree = {"py/type": "builtins.dict"}
_NEWOBJ: MutablePlainTree = {"py/function": "copyreg.__newobj__"}


class TestReduce:
    @pytest.mark.parametrize(
        ("payload", "reason"),
        list[tuple[MutablePlainTree, str]](
            [
                ("x", "invalid py/reduce payload: 'x'"),
                (1, "invalid py/reduce payload: 1"),
                ([_LIST], "py/reduce requires two to five elements"),
                (
                    [_LIST, _NO_ARGS, None, None, None, None],
                    "py/reduce requires two to five elements",
                ),
                ([1, _NO_ARGS], "py/reduce needs a callable and an argument tuple"),
                ([_LIST, [1]], "py/reduce needs a callable and an argument tuple"),
                (
                    [{"py/type": f"{__name__}.Unbuildable"}, {"py/tuple": [-1]}],
                    "py/reduce call failed: negative",
                ),
                (
                    [{"py/type": "builtins.int"}, _NO_ARGS, None, [1]],
                    "reduce target cannot take list items: 0",
                ),
                (
                    [_LIST, _NO_ARGS, None, {"py/tuple": [1]}],
                    "reduce target cannot take list items: []",
                ),
                (
                    [{"py/type": "builtins.int"}, _NO_ARGS, None, None, [None]],
                    "reduce target cannot take dict items: 0",
                ),
                (
                    [_DICT, _NO_ARGS, None, None, _NO_ARGS],
                    "reduce target cannot take dict items: {}",
                ),
                ([_DICT, _NO_ARGS, None, None, [1]], "invalid reduce dict item: 1"),
                (
                    [_DICT, _NO_ARGS, None, None, [{"py/tuple": [1]}]],
                    "invalid reduce dict item: (1,)",
                ),
            ],
        ),
        ids=repr,
    )
    def test_a_bad_recipe_names_why(
        self,
        payload: MutablePlainTree,
        reason: str,
    ) -> None:
        assert _bad_of({"r": {"py/reduce": payload}}) == {("r",): reason}

    def test_each_recipe_part_sits_at_its_index(self) -> None:
        # The callable is a reduce of its own: ``getattr(list, "copy")`` yields a
        # bound ``list.copy`` whose recipe holds a list a later ``py/ref`` names.
        func = _reduce(
            {"py/function": "builtins.getattr"},
            {"py/tuple": [[7], "copy"]},
        )
        tree: MutablePlainTree = {
            "r": _reduce(func, _NO_ARGS),
            "f": {"py/ref": ["r", 0, 1, 0]},
        }

        restored = cast(
            "dict[str, object]",
            from_plain(tree, object, allow_imports=True),
        )

        assert restored == {"r": [7], "f": [7]}
        assert restored["r"] is not restored["f"]

    def test_the_largest_recipe_replays_every_part(self) -> None:
        tree = _reduce(
            {"py/type": "collections.OrderedDict"},
            {"py/tuple": []},
            None,
            None,
            [{"py/tuple": ["a", 1]}],
        )

        assert from_plain({"r": tree}, dict[str, object], allow_imports=True) == {
            "r": OrderedDict(a=1),
        }

    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            ((None, [1, 2]), [1, 2]),
            ((None, None), []),
            ((None, None, None), []),
        ],
        ids=repr,
    )
    def test_trailing_parts_apply_only_when_present(
        self,
        parts: tuple[MutablePlainTree, ...],
        expected: list[int],
    ) -> None:
        assert from_plain(
            {"r": _reduce(_LIST, {"py/tuple": []}, *parts)},
            dict[str, object],
            allow_imports=True,
        ) == {"r": expected}

    def test_state_and_items_are_read_where_they_sit(self) -> None:
        tree = {
            "r": _reduce(
                _NEWOBJ,
                {"py/tuple": [{"py/type": f"{__name__}.Stateful"}]},
                [[1], {"py/ref": ["r", 2, 0]}],
            ),
            "s": {"py/ref": ["r", 2]},
            "l": _reduce(_LIST, {"py/tuple": []}, None, [[1]]),
            "li": {"py/ref": ["l", 3]},
            "d": _reduce(
                _DICT,
                {"py/tuple": []},
                None,
                None,
                [{"py/tuple": ["k", [2]]}],
            ),
            "di": {"py/ref": ["d", 4]},
        }

        restored = cast(
            "dict[str, object]",
            from_plain(tree, object, allow_imports=True),
        )

        stateful = restored["r"]
        assert isinstance(stateful, Stateful)
        assert stateful.state == ("set", [[1], [1]])
        _, state = cast("tuple[str, list[object]]", stateful.state)
        assert state[1] is state[0]
        assert restored["s"] is state
        assert restored["l"] == [[1]]
        assert restored["li"] == [[1]]
        assert restored["d"] == {"k": [2]}
        assert restored["di"] == [("k", [2])]

    def test_a_stateful_value_is_placed_before_its_arguments_read(self) -> None:
        tree = {
            "r": _reduce(
                _NEWOBJ,
                {"py/tuple": [{"py/type": f"{__name__}.Bare"}, {"py/ref": ["r"]}]},
                {},
            ),
        }

        assert _bad_of(tree) == {
            ("r", 1, 1): "py/ref ['r'] names a value still being built",
        }

    def test_a_stateful_value_is_built_before_its_state_reads(self) -> None:
        tree = {
            "r": _reduce(
                _NEWOBJ,
                {"py/tuple": [{"py/type": f"{__name__}.Bare"}]},
                {"me": {"py/ref": ["r"]}},
            ),
        }

        restored = from_plain({"r": tree["r"]}, dict[str, object], allow_imports=True)[
            "r"
        ]

        assert isinstance(restored, Bare)
        assert vars(restored) == {"me": restored}

    @pytest.mark.parametrize("state", [{}, [1], {"py/tuple": [None, None]}], ids=repr)
    def test_a_stateful_value_is_found_by_id_and_path(
        self,
        state: MutablePlainTree,
    ) -> None:
        tree: MutablePlainTree = [
            _reduce(_NEWOBJ, {"py/tuple": [{"py/type": f"{__name__}.Bare"}]}, state),
            {"py/id": 1},
            {"py/ref": [0]},
        ]

        first, by_id, by_path = cast(
            "list[object]",
            from_plain(tree, object, allow_imports=True),
        )

        assert isinstance(first, Bare)
        assert by_id is first
        assert by_path is first

    def test_a_stateless_mutable_value_is_found_by_path_only(self) -> None:
        shared = bytearray(b"a")
        tree = to_plain([shared, shared])

        first, second = cast(
            "list[object]",
            from_plain(tree, object, allow_imports=True),
        )

        assert first == bytearray(b"a")
        assert second is first
        assert _bad_of([to_plain(bytearray(b"a")), {"py/id": 1}]) == {
            (1,): "invalid py/id reference: {'py/id': 1}",
        }

    def test_a_stateless_hashable_value_is_not_placed(self) -> None:
        tree: MutablePlainTree = [to_plain(Level.LOW), {"py/ref": [0]}]

        assert _bad_of(tree) == {(1,): "py/ref [0] names no earlier value"}

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            ({"x": 1}, {"x": 1}),
            ({"py/tuple": [{"x": 1}, {"a": 2}]}, {"x": 1, "a": 2}),
            ({"py/tuple": [None, {"a": 2}]}, {"a": 2}),
            ({"py/tuple": [{"x": 1}, None]}, {"x": 1}),
            ({"py/tuple": [{"x": 1}]}, {}),
            ({"py/tuple": [{"x": 1}, {"a": 2}, {"b": 3}]}, {}),
            ([1], {}),
        ],
        ids=repr,
    )
    def test_state_without_setstate_sets_attributes(
        self,
        state: MutablePlainTree,
        expected: dict[str, int],
    ) -> None:
        tree = _reduce(
            _NEWOBJ,
            {"py/tuple": [{"py/type": f"{__name__}.TwoStates"}]},
            state,
        )

        restored = from_plain({"r": tree}, dict[str, object], allow_imports=True)["r"]

        assert isinstance(restored, TwoStates)
        found = {
            name: getattr(restored, name)
            for name in ("x", "a", "b")
            if hasattr(restored, name)
        }
        assert found == expected

    def test_state_with_setstate_is_handed_over_whole(self) -> None:
        tree = _reduce(
            _NEWOBJ,
            {"py/tuple": [{"py/type": f"{__name__}.Stateful"}]},
            {"x": 1},
        )

        restored = from_plain({"r": tree}, dict[str, object], allow_imports=True)["r"]

        assert isinstance(restored, Stateful)
        assert restored.state == ("set", {"x": 1})
        assert not hasattr(restored, "x")

    def test_a_recipe_naming_the_target_needs_no_import(self) -> None:
        tree = {"py/reduce": [{"py/type": f"{__name__}.Level"}, {"py/tuple": [2]}]}

        assert from_plain(tree, Level) is Level.HIGH
        assert from_plain(tree, Level | None) is Level.HIGH


class TestHooked:
    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ("x", "invalid py/hook payload: 'x'"),
            ([f"{__name__}.Tensor"], f"invalid py/hook payload: ['{__name__}.Tensor']"),
            (
                [f"{__name__}.Tensor", [1.0], 1],
                f"invalid py/hook payload: ['{__name__}.Tensor', [1.0], 1]",
            ),
            (["x.Y", 1], "hook 'x.Y' is not registered"),
            ({"a": 1, "b": 2}, "invalid py/hook payload: {'a': 1, 'b': 2}"),
        ],
        ids=repr,
    )
    def test_a_bad_hook_names_why(self, payload: MutablePlainTree, reason: str) -> None:
        assert _bad_of({"r": {"py/hook": payload}}, hooks=_HOOKS) == {("r",): reason}

    def test_a_hooked_value_is_placed_before_its_payload_reads(self) -> None:
        tree: MutablePlainTree = {
            "r": {"py/hook": [f"{__name__}.Tensor", [{"py/ref": ["r"]}]]},
        }

        assert _bad_of(tree, hooks=_HOOKS) == {
            ("r", 1, 0): "py/ref ['r'] names a value still being built",
        }

    def test_a_hook_payload_decodes_by_its_tags_where_it_sits(self) -> None:
        tree: MutablePlainTree = [
            {"py/hook": [f"{__name__}.Tensor", {"py/tuple": [1.0]}]},
            {"py/hook": [f"{__name__}.Tensor", [2.0]]},
            {"py/ref": [1, 1]},
            {"py/id": 1},
        ]

        first, second, payload, again = cast(
            "list[object]",
            from_plain(tree, object, hooks=_HOOKS, allow_imports=True),
        )

        assert isinstance(first, Tensor)
        assert first.values == (1.0,)
        assert isinstance(second, Tensor)
        assert payload is second.values
        assert again is first


class TestTypeTags:
    @pytest.mark.parametrize(
        ("tree", "target", "reason"),
        [
            ({"py/type": 1}, dict[str, object], "cannot read 1 as a class path"),
            (
                {"py/type": "builtins.max"},
                dict[str, type],
                "'builtins.max' does not name a class",
            ),
            (
                {"py/type": "builtins.max"},
                dict[str, type[int]],
                "'builtins.max' does not name a class",
            ),
            (
                {"py/function": 1},
                dict[str, object],
                "cannot read 1 as a function path",
            ),
            (
                {"py/function": "math.pi"},
                dict[str, object],
                "'math.pi' does not name a callable",
            ),
            (
                {"py/type": "no_such_module_xyz.thing"},
                dict[str, object],
                (
                    "cannot import 'no_such_module_xyz.thing': "
                    "Cannot resolve path: 'no_such_module_xyz.thing'"
                ),
            ),
            ({"py/type": "math.pi"}, dict[str, object], None),
        ],
        ids=repr,
    )
    def test_a_class_path_reads_or_names_why(
        self,
        tree: MutablePlainTree,
        target: object,
        reason: str | None,
    ) -> None:
        if reason is None:
            assert from_plain({"r": tree}, target, allow_imports=True) == {"r": math.pi}
            return
        with pytest.raises(ReadError) as raised:
            from_plain({"r": tree}, target, allow_imports=True)

        assert {at: bad.reason for at, bad in raised.value.bad.items()} == {
            ("r",): reason,
        }

    def test_a_function_path_resolves(self) -> None:
        assert from_plain(
            {"r": {"py/function": "builtins.max"}},
            dict[str, object],
            allow_imports=True,
        ) == {"r": max}

    def test_a_resolved_path_is_cached_by_its_path(self) -> None:
        path = f"{__name__}.Bare"
        deserialize._RESOLVED.pop(path, None)

        assert from_plain(
            {"r": {"py/type": path}},
            dict[str, object],
            allow_imports=True,
        ) == {"r": Bare}
        assert deserialize._RESOLVED[path] is Bare
        deserialize._RESOLVED[path] = Square
        try:
            assert from_plain(
                {"r": {"py/type": path}},
                dict[str, object],
                allow_imports=True,
            ) == {"r": Square}
        finally:
            del deserialize._RESOLVED[path]


class TestBadParts:
    @pytest.mark.parametrize(
        ("tree", "target", "bad"),
        [
            (
                {"json://{": 1, "b": 2},
                dict[str, int],
                {("json://{",): "invalid escaped key 'json://{'"},
            ),
            (
                {"json://[1]": 1, "b": 2},
                dict[object, int],
                {("json://[1]",): "key [1] is unhashable"},
            ),
            (
                {'json://{"py/tuple": [[1]]}': 1, "b": 2},
                dict[object, int],
                {('json://{"py/tuple": [[1]]}',): "key ([1],) is unhashable"},
            ),
            (
                {"x": 1, "y": 2},
                dict[int, int],
                {("x",): "cannot read 'x' as int", ("y",): "cannot read 'y' as int"},
            ),
            ({"py/set": [[1]]}, object, {(): "set member [1] is unhashable"}),
            (
                {"py/frozenset": [[1]]},
                object,
                {(): "frozenset members [[1]] are unhashable"},
            ),
            ({"py/set": "ab"}, object, {(): "cannot read 'ab' as <class 'object'>"}),
            ({"py/set": 1}, object, {(): "cannot read 1 as <class 'object'>"}),
            (
                {"py/frozenset": "ab"},
                object,
                {(): "cannot read 'ab' as <class 'object'>"},
            ),
            ({"py/frozenset": 1}, object, {(): "cannot read 1 as <class 'object'>"}),
            ("ab", set[str], {(): "cannot read 'ab' as set[str]"}),
            ("ab", frozenset[str], {(): "cannot read 'ab' as frozenset[str]"}),
            (
                {"py/set": [1, "a"]},
                set[int],
                {(1,): "cannot read 'a' as int"},
            ),
            (
                {"py/frozenset": [1, "a"]},
                frozenset[int],
                {(1,): "cannot read 'a' as int"},
            ),
            (
                {"py/ref": [], "py/tuple": []},
                object,
                {(): "invalid py/ref reference: {'py/ref': [], 'py/tuple': []}"},
            ),
            (
                {"py/id": 0, "py/tuple": []},
                object,
                {(): "invalid py/id reference: {'py/id': 0, 'py/tuple': []}"},
            ),
            (
                {"py/tuple": [], "x": 1},
                object,
                {(): "invalid py/tuple envelope: {'py/tuple': [], 'x': 1}"},
            ),
            (
                {"py/unknown": 1},
                object,
                {(): "unknown tag among ['py/unknown']"},
            ),
            (
                {"py/unknown": 1, "b": 2},
                Point,
                {(): "unknown tag among ['b', 'py/unknown']"},
            ),
        ],
        ids=repr,
    )
    def test_each_bad_part_names_why(
        self,
        tree: MutablePlainTree,
        target: object,
        bad: dict[FieldPath, str],
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain(tree, target)

        reasons = {at: part.reason for at, part in raised.value.bad.items()}
        assert reasons == bad

    def test_a_bad_key_keeps_its_raw_key_and_the_rest(self) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain({"x": 1, "json://2": 2, "y": 3}, dict[int, int])

        assert raised.value.bad == {
            ("x",): Invalid(raw="x", reason="cannot read 'x' as int"),
            ("y",): Invalid(raw="y", reason="cannot read 'y' as int"),
        }
        assert raised.value.partial == {2: 2}

    def test_a_non_str_key_that_does_not_fit_is_bad(self) -> None:
        data = cast("PlainTree", {True: 1, 2: 2, "a": 3})

        with pytest.raises(ReadError) as raised:
            from_plain(data, dict[str, int])

        assert raised.value.bad == {
            (True,): Invalid(raw=True, reason="cannot read key True as <class 'str'>"),
            (2,): Invalid(raw=2, reason="cannot read key 2 as <class 'str'>"),
        }
        assert raised.value.partial == {"a": 3}

    def test_a_non_str_key_that_fits_reads_as_is(self) -> None:
        data = cast("PlainTree", {True: 1, 2: 2, None: 3})

        assert from_plain(data, dict[int | None, int]) == {True: 1, 2: 2, None: 3}
        assert from_plain(cast("PlainTree", {2: 2}), dict[int, int]) == {2: 2}
        with pytest.raises(ReadError) as raised:
            from_plain(cast("PlainTree", {None: 1}), dict[int, int])
        assert dict(raised.value.bad) == {
            (None,): Invalid(raw=None, reason="cannot read key None as <class 'int'>"),
        }

    def test_a_set_and_its_members_sit_where_they_are(self) -> None:
        tree: MutablePlainTree = [
            {"py/set": [{"py/tuple": [1]}]},
            {"py/ref": [0]},
            {"py/id": 1},
        ]

        first, by_path, by_id = cast("list[object]", from_plain(tree, object))

        assert first == {(1,)}
        assert by_path is first
        assert by_id is first

    @pytest.mark.parametrize(
        ("tree", "target", "expected"),
        [
            ({"py/frozenset": [[1], [2]]}, frozenset[tuple[int, ...]], {(1,), (2,)}),
            ({"py/set": [[1], [2]]}, set[tuple[int, ...]], {(1,), (2,)}),
            ({"py/frozenset": [1]}, frozenset[int], frozenset({1})),
        ],
        ids=repr,
    )
    def test_set_members_read_as_their_element_type(
        self,
        tree: MutablePlainTree,
        target: object,
        expected: object,
    ) -> None:
        assert from_plain(tree, target) == expected


_MALFORMED_NODES: Final[tuple[tuple[MutablePlainTree, str], ...]] = (
    (
        {"py/ref": [0], "py/object": "x"},
        "invalid py/ref reference: {'py/ref': [0], 'py/object': 'x'}",
    ),
    (
        {"py/id": 0, "py/object": "x"},
        "invalid py/id reference: {'py/id': 0, 'py/object': 'x'}",
    ),
    (
        {"py/ref": [0], "py/tuple": []},
        "invalid py/ref reference: {'py/ref': [0], 'py/tuple': []}",
    ),
    (
        {"py/id": 0, "py/tuple": []},
        "invalid py/id reference: {'py/id': 0, 'py/tuple': []}",
    ),
    ({"py/ref": [0], "x": 1}, "invalid py/ref reference: {'py/ref': [0], 'x': 1}"),
    ({"py/id": 0, "x": 1}, "invalid py/id reference: {'py/id': 0, 'x': 1}"),
    ({"py/ref": "a"}, "invalid py/ref reference: {'py/ref': 'a'}"),
    ({"py/id": True}, "invalid py/id reference: {'py/id': True}"),
    (
        {"py/tuple": [1], "x": 2},
        "invalid py/tuple envelope: {'py/tuple': [1], 'x': 2}",
    ),
    ({"py/unknown": 1}, "unknown tag among ['py/unknown']"),
    ({"py/unknown": 1, "b": 2}, "unknown tag among ['b', 'py/unknown']"),
)
"""Malformed nodes beside a ``[1]`` at index 0, and why each does not read."""


class TestFastPath:
    def test_an_untagged_object_read_returns_the_input_itself(self) -> None:
        tree = cast("PlainTree", {"a": [1, {"b": "c"}], "d": None})

        assert from_plain(tree, object) is tree

    def test_a_tagged_tree_keeps_its_untagged_leaves(self) -> None:
        text, number, flag = "a" * 300, 10**30, True
        tree: MutablePlainTree = {"t": {"py/tuple": [text]}, "n": number, "f": flag}

        restored = cast("dict[str, object]", from_plain(tree, object))

        assert restored == {"t": (text,), "n": number, "f": flag}
        assert cast("tuple[object, ...]", restored["t"])[0] is text
        assert restored["n"] is number
        assert restored["f"] is flag

    @pytest.mark.parametrize("leaf", ["a" * 300, 10**30, 1.5, True, None], ids=repr)
    def test_a_plain_leaf_is_returned_as_is(self, leaf: Plain) -> None:
        assert from_plain(leaf, object) is leaf

    @pytest.mark.parametrize(
        "data",
        ["a", 1, None, {"py/tuple": [1]}, {"a": {"py/set": [2]}}],
        ids=repr,
    )
    def test_an_object_read_resolves_no_type_aliases(
        self,
        data: MutablePlainTree,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Every node of a config reads as ``object``; resolving and dispatching
        # its target per node made such a read 3x slower (2.2 ms -> 7.1 ms).
        expected = from_plain(data, object)
        resolved: list[object] = []
        monkeypatch.setattr(deserialize, "_resolve_alias", resolved.append)

        assert deserialize._python_from_plain(data, object) == expected
        assert resolved == []

    @pytest.mark.parametrize(("node", "reason"), _MALFORMED_NODES, ids=repr)
    @pytest.mark.parametrize("allow_imports", [False, True], ids=repr)
    @pytest.mark.parametrize(
        "target",
        [object, Any, list[object], list[Shape], Sequence[object]],
        ids=repr,
    )
    def test_a_malformed_node_reads_alike_on_every_path(
        self,
        node: MutablePlainTree,
        reason: str,
        allow_imports: bool,
        target: object,
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([[1], node], target, allow_imports=allow_imports)

        assert dict(raised.value.bad) == {(1,): Invalid(raw=node, reason=reason)}


class TestDispatch:
    @pytest.mark.parametrize(
        ("data", "target", "expected"),
        [
            (["a", {"b": [1]}], object, ["a", {"b": [1]}]),
            ({"py/tuple": [{"b": [1]}]}, object, ({"b": [1]},)),
            ([{"py/tuple": [1]}, {"b": [2]}], list[object], [(1,), {"b": [2]}]),
            ({"py/mappingproxy": {"a": 1}}, Mapping[str, int], {"a": 1}),
        ],
        ids=repr,
    )
    def test_an_object_part_reads_by_its_tags(
        self,
        data: MutablePlainTree,
        target: object,
        expected: object,
    ) -> None:
        assert from_plain(data, target) == expected

    def test_an_object_mapping_is_numbered_where_it_sits(self) -> None:
        tree: MutablePlainTree = [
            {"py/tuple": []},
            [{"a": 1}, {"py/ref": [1, 0]}, {"py/id": 2}],
        ]

        _, inner = from_plain(tree, list[object])

        first, by_path, by_id = cast("list[object]", inner)
        assert first == {"a": 1}
        assert by_path is first
        assert by_id is first

    def test_an_object_mapping_reads_its_keys_and_values(self) -> None:
        tree: MutablePlainTree = [
            {"py/tuple": []},
            {"json://1": [2], "b": {"py/ref": [1, "json://1"]}},
        ]

        _, inner = from_plain(tree, list[object])

        assert inner == {1: [2], "b": [2]}
        mapping = cast("dict[object, object]", inner)
        assert mapping["b"] is mapping[1]

    def test_a_proxy_tag_does_not_read_as_a_dict(self) -> None:
        with pytest.raises(
            ReadError,
            match=r"^cannot read mappingproxy\(\{'a': 1\}\) as dict\[str, object\]$",
        ):
            from_plain({"py/mappingproxy": {"a": 1}}, dict[str, object])

    def test_an_untagged_object_member_with_tags_elsewhere_is_copied(self) -> None:
        tree = cast("PlainTree", [{"py/tuple": []}, {"a": 1}, [2]])

        _, mapping, items = from_plain(tree, list[object])

        assert mapping == {"a": 1}
        assert items == [2]
        assert mapping is not cast("list[object]", tree)[1]
        assert items is not cast("list[object]", tree)[2]

    @pytest.mark.parametrize(
        ("tree", "target", "expected"),
        [
            ({"py/mappingproxy": {"a": 1}}, MappingProxyType[str, int], {"a": 1}),
            (
                {"py/object": f"{__name__}.Point", "x": 1},
                Point | dict[str, object],
                Point(x=1),
            ),
        ],
        ids=repr,
    )
    def test_a_tag_a_mapping_target_reads(
        self,
        tree: MutablePlainTree,
        target: object,
        expected: object,
    ) -> None:
        assert from_plain(tree, target, allow_imports=True) == expected

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (Shape, {"py/object": f"{__name__}.Square", "side": 1.0}),
            (Any, {"py/object": f"{__name__}.Square", "side": 1.0}),
        ],
        ids=repr,
    )
    def test_without_imports_an_untyped_object_record_stays_a_record(
        self,
        target: object,
        expected: object,
    ) -> None:
        tree = to_plain(Square(side=1.0))

        assert from_plain(tree, target) == expected
        assert from_plain(tree, target, allow_imports=True) == Square(side=1.0)

    def test_without_imports_a_typed_object_record_reads_its_fields(self) -> None:
        tree = to_plain(Square(side=1.0))

        assert from_plain(tree, Square) == Square(side=1.0)
        with pytest.raises(
            ReadError,
            match=rf"^'{__name__}\.Square' needs allow_imports=True$",
        ):
            from_plain(tree, int)

    @pytest.mark.parametrize(
        ("target", "decoder"),
        [
            (AbstractBox, "_decode_object"),
            (Abstract, "_decode_any"),
            (Shape, "_decode_any"),
            (Any, "_decode_any"),
            (Point, "_decode_object"),
            (Bare, "_decode_object"),
            (Meta, "_decode_typeddict"),
            (Color, "_decode_enum"),
            (list[int], "_decode_list"),
            (_T, "_decode_any"),
        ],
        ids=repr,
    )
    def test_each_target_picks_its_decoder(self, target: object, decoder: str) -> None:
        assert deserialize._decoder_for(target) is getattr(deserialize, decoder)

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (Annotated[Ints, "meta"], list[int]),
            (Annotated[Annotated[int, "a"], "b"], int),
            (IntsAgain, list[int]),
        ],
        ids=repr,
    )
    def test_an_alias_resolves_to_its_type(
        self,
        target: object,
        expected: object,
    ) -> None:
        assert deserialize._resolve_alias(target) == expected

    def test_an_alias_loop_stops_where_it_returns(self) -> None:
        loop = AliasLoop()

        assert deserialize._resolve_alias(loop) is loop

    @pytest.mark.parametrize(
        ("data", "target", "expected"),
        [
            ([1], Ints | str, [1]),
            ("a", Ints | str, "a"),
            (None, int | None, None),
            (None, Optional[int], None),  # noqa: UP045 -- Spelling under test.  # pyright: ignore[reportDeprecated] -- Spelling under test.
            (1.0, int | float, 1.0),
            (True, int | bool, True),
            ({"py/tuple": [1]}, list[int] | tuple[int, ...], (1,)),
        ],
        ids=repr,
    )
    def test_a_union_reads_the_data_kind_first(
        self,
        data: MutablePlainTree,
        target: object,
        expected: object,
    ) -> None:
        restored = from_plain(data, target, strict=False)

        assert restored == expected
        assert type(restored) is type(expected)

    @pytest.mark.parametrize(
        ("data", "members", "expected"),
        [
            (None, (int, type(None)), (type(None), int)),
            (1, (str, Ints), (str, Ints)),
            ([1], (str, Ints), (Ints, str)),
            (
                [1],
                (str, Annotated[list[int], "meta"]),
                (Annotated[list[int], "meta"], str),
            ),
            ((1,), (str, Sequence[int]), (Sequence[int], str)),
            (
                {"a": 1},
                (str, MappingProxyType[str, int], Mapping[str, int]),
                (Mapping[str, int], str, MappingProxyType[str, int]),
            ),
        ],
        ids=repr,
    )
    def test_union_candidates_put_the_data_kind_first(
        self,
        data: PlainTree,
        members: tuple[object, ...],
        expected: tuple[object, ...],
    ) -> None:
        assert deserialize._union_candidates(data, members) == expected


class TestScalars:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("TRUE", True), ("fAlSe", False), ("1", True), ("0", False)],
    )
    def test_lax_bool_text_ignores_case(self, raw: str, expected: bool) -> None:
        assert from_plain(raw, bool, strict=False) is expected

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ("maybe", "cannot read 'maybe' as bool"),
            ("2", "cannot read '2' as bool"),
            (2, "cannot read 2 as bool"),
            (0.5, "cannot read 0.5 as bool"),
        ],
        ids=repr,
    )
    def test_lax_bool_names_the_input_it_cannot_read(
        self,
        raw: PlainTree,
        message: str,
    ) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain(raw, bool, strict=False)

        assert str(raised.value) == message

    @pytest.mark.parametrize(
        "raw",
        ["nan", "inf", "-inf", math.inf, math.nan],
        ids=repr,
    )
    def test_lax_int_refuses_a_non_finite_number(self, raw: PlainTree) -> None:
        with pytest.raises(ReadError, match=r"^cannot read .* as int$"):
            from_plain(raw, int, strict=False)

    @pytest.mark.parametrize(
        "raw",
        [
            "9007199254740993",
            "-9007199254740993",
            " 123456789012345678901234567890 ",
            "1_000_000_000_000_000_001",
        ],
        ids=repr,
    )
    def test_lax_int_reads_integer_text_exactly(self, raw: str) -> None:
        assert from_plain(raw, int, strict=False) == int(raw)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1e3", 1000),
            ("2.0", 2),
            ("9007199254740993.0", 9_007_199_254_740_993),
            ("-9007199254740993e0", -9_007_199_254_740_993),
            ("1.5e30", 15 * 10**29),
            # Integral only after rounding to a float: keeps the rounded value.
            ("1.0000000000000000001", 1),
        ],
        ids=repr,
    )
    def test_lax_int_reads_integral_float_text(self, raw: str, expected: int) -> None:
        assert from_plain(raw, int, strict=False) == expected

    @pytest.mark.parametrize("raw", [None, [1], {"a": 1}], ids=repr)
    def test_lax_int_refuses_a_non_number(self, raw: PlainTree) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain([raw], list[int], strict=False)

        assert dict(raised.value.bad) == {
            (0,): Invalid(raw=raw, reason=f"cannot read {raw!r} as int"),
        }

    @pytest.mark.parametrize(
        ("text", "zone"),
        [
            ("2026-01-01T00:00:00+00:00[Europe/Paris]", "Europe/Paris"),
            ("2026-06-01T12:00:00-04:00[America/New_York]", "America/New_York"),
        ],
    )
    def test_a_named_zone_is_restored(self, text: str, zone: str) -> None:
        moment = from_plain({"py/datetime": text}, dt.datetime)

        assert str(moment.tzinfo) == zone
        assert moment == dt.datetime.fromisoformat(text.partition("[")[0])
        assert moment.utcoffset() == ZoneInfo(zone).utcoffset(moment)

    def test_a_literal_skips_an_enum_option_that_does_not_read(self) -> None:
        assert from_plain("red", Literal[Level.LOW, Color.RED]) is Color.RED

    @pytest.mark.parametrize(
        ("data", "expected"),
        [(1, 1), (True, True), ("a", "a")],
        ids=repr,
    )
    def test_a_literal_matches_by_kind_and_value(
        self,
        data: PlainTree,
        expected: object,
    ) -> None:
        restored = from_plain(data, Literal[True, 1, "a"])

        assert restored == expected
        assert type(restored) is type(expected)

    @pytest.mark.parametrize("data", [1.0, False, "b", None], ids=repr)
    def test_a_literal_refuses_another_kind_or_value(self, data: PlainTree) -> None:
        with pytest.raises(ReadError) as raised:
            from_plain(data, Literal[True, 1, "a"])

        assert str(raised.value) == (
            f"cannot read {data!r} as typing.Literal[True, 1, 'a']"
        )


class TestForeignInput:
    def test_a_non_str_key_from_yaml_reads(self) -> None:
        # ``yaml.safe_load`` reads ``on:`` as the bool key ``True``.
        data = cast("PlainTree", {True: 1, "py": 2})
        assert from_plain(data, dict[object, int]) == {True: 1, "py": 2}

    def test_a_typed_leaf_passes_through(self) -> None:
        moment = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        assert from_plain(moment, dt.datetime) is moment
        assert from_plain({"at": moment}, dict[str, dt.datetime]) == {"at": moment}

    def test_an_untagged_object_member_is_the_input_itself(self) -> None:
        # Callers edit a read record's ``object`` members in place, as
        # ``custom_json.convert`` let them.
        inner: dict[str, object] = {"b": [1]}
        record = cast("PlainTree", {"a": inner})
        read = from_plain(record, dict[str, object])
        assert read == record
        assert read is not record
        assert read["a"] is inner
        assert from_plain(record, object) is record

    def test_a_tagged_object_member_still_decodes(self) -> None:
        record = cast("PlainTree", {"a": {"b": {"py/tuple": [1]}}})
        assert from_plain(record, dict[str, object]) == {"a": {"b": (1,)}}

    def test_a_typed_leaf_of_the_wrong_type_raises(self) -> None:
        with pytest.raises(ReadError):
            from_plain(dt.datetime(2026, 1, 1, tzinfo=dt.UTC), int)

    def test_a_tag_under_parts_built_on_each_read_still_decodes(self) -> None:
        # Each read builds a new list, and dropping it frees its id for the next
        # one. The tag scan must not mistake that list for one it already scanned.
        data = cast(
            "PlainTree",
            [_FreshParts([{"py/tuple": [1, 2]}]), _FreshParts([1])],
        )
        assert from_plain(data, object) == [[[(1, 2)]], [[1]]]


class _FreshParts(Sequence[object]):
    """A sequence that returns a new one-item list each time an item is read."""

    def __init__(self, items: list[object]) -> None:
        self.items = items

    @override
    def __len__(self) -> int:
        return len(self.items)

    @override
    def __getitem__(self, index: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride] -- Accepts any index to raise IndexError as a misbehaving sequence would.  # ty: ignore[invalid-method-override] -- Same.
        if not isinstance(index, int) or index >= len(self.items):
            raise IndexError(index)
        return [self.items[index]]


class TestLax:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("False", False), ("0", False), ("1", True), (1, True)],
    )
    def test_bool_reads_tokens_and_zero_or_one(
        self,
        raw: object,
        expected: bool,
    ) -> None:
        assert from_plain(raw, bool, strict=False) is expected

    @pytest.mark.parametrize("raw", ["maybe", " Yes ", "", 0.5, 2, math.nan])
    def test_bool_rejects_what_it_cannot_read(self, raw: object) -> None:
        with pytest.raises(ReadError):
            from_plain(raw, bool, strict=False)

    def test_numbers_coerce_only_without_loss(self) -> None:
        assert from_plain("5", int, strict=False) == 5
        assert from_plain(3.0, int, strict=False) == 3
        assert from_plain("2.5", float, strict=False) == 2.5
        with pytest.raises(ReadError):
            from_plain(1.9, int, strict=False)

    @pytest.mark.parametrize(
        ("target", "raw"),
        [(str, 5), (int | None, True), (Literal[1], True), (int, "x")],
    )
    def test_other_json_types_are_refused(self, target: object, raw: object) -> None:
        with pytest.raises(ReadError):
            from_plain(raw, target, strict=False)

    def test_strict_is_the_default(self) -> None:
        with pytest.raises(ReadError):
            from_plain("3", int)

    def test_lax_reaches_nested_fields(self) -> None:
        assert from_plain({"a": "3"}, dict[str, int], strict=False) == {"a": 3}


class TestObjectTagForTarget:
    def test_a_tag_naming_the_target_needs_no_imports(self) -> None:
        tree = to_plain(Tagged(pair=(1, 2)))
        assert from_plain(tree, Tagged) == Tagged(pair=(1, 2))

    def test_the_tag_reads_with_declared_field_types(self) -> None:
        tree = {"py/object": f"{__name__}.Tagged", "pair": [1, 2]}
        assert from_plain(tree, Tagged).pair == (1, 2)

    def test_the_tag_beats_member_order_when_fields_overlap(self) -> None:
        tree = {"py/object": f"{__name__}.Pair2"}
        assert type(from_plain(tree, Pair1 | Pair2)) is Pair2

    def test_the_tag_selects_a_union_member(self) -> None:
        tree = {"py/object": f"{__name__}.OtherTagged", "name": "a"}
        assert from_plain(tree, Tagged | OtherTagged) == OtherTagged(name="a")

    def test_a_mapping_target_keeps_the_tag_as_a_key(self) -> None:
        tree = {"py/object": "no.such.Class", "x": 1}
        assert from_plain(tree, dict[str, object]) == tree
        assert from_plain([tree], list[dict[str, object]]) == [tree]

    def test_a_mapping_target_keeps_a_value_tag_as_a_key(self) -> None:
        tree = {"py/tuple": [1, 6]}
        assert from_plain(tree, dict[str, object]) == tree
        assert from_plain(tree, tuple[int, ...]) == (1, 6)

    def test_an_object_member_without_imports_stays_a_record(self) -> None:
        # No class can be built without imports, so the record is data, as
        # ``custom_json.convert`` kept it; a typed read later builds it.
        inner = {"py/object": f"{__name__}.Tagged", "pair": [1, 2]}
        outer = from_plain({"spine": inner}, dict[str, object])
        assert outer == {"spine": inner}
        assert from_plain(outer["spine"], Tagged) == Tagged(pair=(1, 2))

    def test_an_object_member_with_imports_builds(self) -> None:
        inner = {"py/object": f"{__name__}.Tagged", "pair": [1, 2]}
        read = from_plain({"spine": inner}, dict[str, object], allow_imports=True)
        assert isinstance(read["spine"], Tagged)

    def test_an_enum_field_named_by_its_target_needs_no_imports(self) -> None:
        assert from_plain(to_plain(Boxed(color=Color.RED)), Boxed) == Boxed(
            color=Color.RED,
        )
        assert from_plain(to_plain(Color.BLUE), Color) is Color.BLUE
        assert from_plain(to_plain(Color.BLUE), Color | None) is Color.BLUE

    def test_a_reduce_naming_another_class_still_needs_imports(self) -> None:
        tree = {"py/reduce": [{"py/type": f"{__name__}.Level"}, {"py/tuple": [1]}]}

        with pytest.raises(ReadError, match="allow_imports"):
            from_plain(tree, Color)

    def test_without_imports_a_dataclass_target_reads_any_tag(self) -> None:
        # Data written before a class moved names its old path; the declared
        # target reads it, as ``custom_json.convert`` did.
        tree = {"py/object": "old.home.Tagged", "pair": [1, 2]}
        assert from_plain(tree, Tagged) == Tagged(pair=(1, 2))

    def test_with_imports_a_tag_naming_another_class_builds_it(self) -> None:
        tree = {"py/object": f"{__name__}.OtherTagged", "name": "a"}
        with pytest.raises(ReadError, match="cannot read"):
            from_plain(tree, Tagged, allow_imports=True)

    def test_a_decimal_reads_as_a_float(self) -> None:
        assert from_plain(Decimal("1.50"), float) == 1.5
        assert type(from_plain(Decimal("1.50"), float)) is float


class TestFieldTypes:
    @pytest.fixture(autouse=True)
    def cold_cache(self) -> None:
        """Clear cached types, so each test computes the ones it checks."""
        deserialize._FIELD_TYPES.clear()
        deserialize._PARTIAL_HINTS.clear()

    def test_a_dataclass_lists_fields_then_non_init_slots(self) -> None:
        assert dict(deserialize._field_types(Flagged)) == {"name": str, "_done": bool}

    def test_a_typed_dict_lists_its_hints(self) -> None:
        assert dict(deserialize._field_types(Meta)) == {"size": int, "name": str}

    def test_a_missing_hint_reads_as_object(self) -> None:
        assert deserialize._field_types(Slotted) == {}
        assert deserialize._field_types(LocalReduce) == {}
        assert dict(deserialize._field_types(SlotChild)) == {
            "x": int,
            "extra": object,
        }

    def test_a_module_level_class_is_cached(self) -> None:
        first = deserialize._field_types(Point)

        assert dict(first) == {"x": int, "y": float}
        assert deserialize._field_types(Point) is first
        assert deserialize._FIELD_TYPES[Point] is first

    def test_a_nested_class_its_module_reaches_is_cached(self) -> None:
        first = deserialize._field_types(Holder.Part)

        assert dict(first) == {"x": int}
        assert deserialize._field_types(Holder.Part) is first
        assert deserialize._FIELD_TYPES[Holder.Part] is first

    def test_a_nested_class_its_owner_does_not_name_is_not_cached(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        part = Holder.Part
        monkeypatch.delattr(Holder, "Part")

        assert dict(deserialize._field_types(part)) == {"x": int}
        assert part not in deserialize._FIELD_TYPES

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (Node, {"name": str, "children": list[Node]}),
            (Pointer, {"to": Pointer | None}),
        ],
        ids=repr,
    )
    def test_a_self_naming_class_is_not_cached(
        self,
        kind: type,
        expected: dict[str, object],
    ) -> None:
        assert dict(deserialize._field_types(kind)) == expected
        assert kind not in deserialize._FIELD_TYPES

    def test_a_class_named_elsewhere_in_its_module_is_not_cached(self) -> None:
        assert dict(deserialize._field_types(Renamed)) == {"x": int}
        assert Renamed not in deserialize._FIELD_TYPES

    def test_a_class_of_an_unloaded_module_is_not_cached(self) -> None:
        unloaded = make_dataclass("Unloaded", [("x", int)])
        unloaded.__module__ = "no_such_module_xyz"

        assert dict(deserialize._field_types(unloaded)) == {"x": int}
        assert unloaded not in deserialize._FIELD_TYPES

    def test_a_class_its_module_does_not_name_is_not_cached(self) -> None:
        @dataclass(kw_only=True, slots=True)
        class Local:
            x: int

        assert deserialize._field_types(Local) == {"x": int}
        assert Local not in deserialize._FIELD_TYPES

    def test_a_class_with_an_unresolved_hint_is_not_cached(self) -> None:
        assert dict(deserialize._field_types(HintedLater)) == {
            "point": object,
            "count": int,
        }
        assert HintedLater not in deserialize._FIELD_TYPES

    def test_a_class_with_an_unresolved_hint_retries_only_that_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = dict(deserialize._field_types(HintedLater))
        calls: list[object] = []

        def counted(*args: object, **kwargs: object) -> object:
            calls.append(args[0])
            return get_type_hints(*args, **kwargs)  # pyright: ignore[reportArgumentType] -- Forwards the spied call's arguments unchanged.  # ty: ignore[invalid-argument-type] -- Same.

        monkeypatch.setattr(deserialize, "get_type_hints", counted)

        assert dict(deserialize._field_types(HintedLater)) == first
        assert len(calls) == 1

    def test_a_retry_survives_its_entry_being_dropped_meanwhile(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = dict(deserialize._field_types(HintedLater))
        assert HintedLater in deserialize._PARTIAL_HINTS

        # Stands in for another thread reading the same class between the
        # entry's lookup and its removal.
        def dropped_meanwhile(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            deserialize._PARTIAL_HINTS.pop(HintedLater, None)
            return True

        monkeypatch.setattr(deserialize, "_resolves", dropped_meanwhile)

        assert dict(deserialize._field_types(HintedLater)) == first

    def test_a_hint_that_resolves_later_reads_as_its_type(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            deserialize,
            "_FIELD_TYPES",
            weakref.WeakKeyDictionary[type, Mapping[str, object]](),
        )
        deserialize._field_types(HintedLater)
        monkeypatch.setitem(globals(), "Rational", numbers.Rational)

        assert deserialize._field_types(HintedLater)["point"] is numbers.Rational


class TestUnionRetries:
    def test_a_failed_key_does_not_fail_a_later_key(self) -> None:
        data: dict[str, int] = {}
        for index in range(200):
            data[f'json://[1, "x{index}"]'] = index
            data[f"json://[{index}, {index}]"] = index

        with pytest.raises(ReadError) as raised:
            from_plain(data, dict[tuple[int, int] | tuple[str, str], int])

        assert sorted(str(at[0]) for at in raised.value.bad) == sorted(
            f'json://[1, "x{index}"]' for index in range(200)
        )


class TestCyclicInput:
    def test_untagged_cyclic_data_reads_as_itself(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)

        assert from_plain(cyclic, object) is cyclic

    def test_cyclic_data_read_as_a_recursive_type_is_a_read_error(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)

        with pytest.raises(ReadError):
            from_plain(cyclic, _Nested)

    def test_tagged_cyclic_data_is_a_read_error(self) -> None:
        cyclic: list[object] = [{"py/tuple": [1]}]
        cyclic.append(cyclic)

        with pytest.raises(ReadError):
            from_plain(cyclic, object)

    def test_a_reference_to_a_finished_cycle_fits_a_recursive_type(self) -> None:
        loop: list[object] = []
        loop.append(loop)

        first, second = from_plain(to_plain([loop, loop]), list[_Nested])

        assert isinstance(first, list)
        assert second is first
        assert first[0] is first


class TestCaches:
    def test_a_local_dataclass_stays_collectible(self) -> None:
        @dataclass(kw_only=True, slots=True)
        class Local:
            x: int

        assert from_plain({"x": 1}, Local) == Local(x=1)
        ref = weakref.ref(Local)
        del Local
        gc.collect()

        assert ref() is None


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
