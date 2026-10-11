"""Tests for ``treekle.deserialize``."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, make_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import (
    Annotated,
    Literal,
    NotRequired,
    Optional,  # pyright: ignore[reportDeprecated] -- Imported to test its pre-3.14 origin.
    Protocol,
    SupportsIndex,
    TypedDict,
    TypeVar,
    Union,  # pyright: ignore[reportDeprecated] -- Imported to test its pre-3.14 origin.
    cast,
    override,
)
from uuid import UUID

import datetime as dt
import gc
import math
import weakref

import pytest

from treekle import deserialize
from treekle.deserialize import from_plain
from treekle.serialize import to_plain
from treekle.types import (
    Hooks,
    Invalid,
    MutablePlainTree,
    Plain,
    PlainTree,
    ReadError,
)


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


_HOOKS: Hooks = MappingProxyType({Tensor: (_tensor_values, Tensor)})


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

    def test_a_float_accepts_an_int_and_a_tag(self) -> None:
        assert from_plain(1, float) == 1.0
        assert from_plain({"py/float": "-inf"}, float) == -math.inf

    def test_a_pep695_alias_resolves(self) -> None:
        assert from_plain([1], Ints) == [1]

    @pytest.mark.parametrize(
        ("data", "target"),
        [
            (True, int),
            (1.5, int),
            ("1", int),
            (1, str),
            (1, bool),
            (None, int),
            (1, type(None)),
            ("x", float),
            ({"py/float": "1.0"}, float),
            ([1], dict[str, int]),
            ({"a": 1}, list[int]),
            ([1, 2], tuple[int]),
            ("!", bytes),
            ("not-a-uuid", UUID),
            ("2026-13-01", dt.datetime),
            ("2026-01-01T00:00:00[Europe/Paris]", dt.datetime),
            ("2026-01-01T00:00:00+00:00[Nowhere/Zone]", dt.datetime),
            ("green", Color),
            ([1], Point),
            ({"x": 1, "z": 2}, Point),
            ({"py/tuple": [1]}, list[int]),
            (1, Path),
        ],
        ids=repr,
    )
    def test_mismatches_raise(
        self,
        data: PlainTree,
        target: object,
    ) -> None:
        with pytest.raises(ReadError):
            from_plain(data, target)

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

        value = deserialize._decode_object({"x": 1}, Point, (), state)

        assert state.built == [value]

    def test_an_unresolvable_annotation_raises(self) -> None:
        broken = make_dataclass("Broken", [("x", "NoSuchName")])

        with pytest.raises(ReadError, match="field types"):
            from_plain({"x": 1}, broken)


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

    def test_an_unknown_field_is_dropped(self) -> None:
        assert from_plain({"size": 3, "extra": 1}, Meta) == {"size": 3}

    def test_a_non_mapping_raises(self) -> None:
        with pytest.raises(ReadError):
            from_plain([1], Meta)


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
    def test_a_dataclass_lists_fields_then_non_init_slots(self) -> None:
        assert dict(deserialize._field_types(Flagged)) == {"name": str, "_done": bool}

    def test_a_typed_dict_lists_its_hints(self) -> None:
        assert dict(deserialize._field_types(Meta)) == {"size": int, "name": str}

    def test_a_missing_hint_reads_as_object(self) -> None:
        assert deserialize._field_types(Slotted) == {}
        assert deserialize._field_types(LocalReduce) == {}

    def test_a_module_level_class_is_cached(self) -> None:
        first = deserialize._field_types(Point)

        assert deserialize._field_types(Point) is first
        assert deserialize._FIELD_TYPES[Point] is first

    def test_a_self_naming_class_is_not_cached(self) -> None:
        deserialize._field_types(Node)

        assert Node not in deserialize._FIELD_TYPES

    def test_a_class_its_module_does_not_name_is_not_cached(self) -> None:
        @dataclass(kw_only=True, slots=True)
        class Local:
            x: int

        assert deserialize._field_types(Local) == {"x": int}
        assert Local not in deserialize._FIELD_TYPES


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
    from treekle.lib.testing.main import test_main

    test_main(__file__)
