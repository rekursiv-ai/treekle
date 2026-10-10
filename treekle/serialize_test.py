"""Tests for ``treekle.serialize``."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
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
import math
import threading

import pytest

from treekle.deserialize import from_plain
from treekle.serialize import to_plain


if TYPE_CHECKING:
    from treekle.types import Hooks, MutablePlainTree


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

    def test_json_dialect_tags_non_finite_floats(self) -> None:
        assert to_plain([math.inf, 1.5]) == [{"py/float": "inf"}, 1.5]

    def test_python_dialect_keeps_non_finite_floats(self) -> None:
        tree = to_plain([math.inf], dialect="python")

        assert tree == [math.inf]

    def test_immutable_output_is_read_only(self) -> None:
        tree = to_plain({"a": [(1,)]}, mutable=False)

        assert isinstance(tree, MappingProxyType)
        assert tree["a"] == (MappingProxyType({"py/tuple": (1,)}),)

    def test_shared_objects_become_references(self) -> None:
        shared = [1]

        assert to_plain([shared, shared]) == [[1], {"py/ref": [0]}]

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

    def test_enum_encodes_by_reduce(self) -> None:
        assert to_plain(Color.RED) == {
            "py/reduce": [{"py/type": f"{__name__}.Color"}, {"py/tuple": ["red"]}],
        }

    def test_a_partial_encodes_the_same_once_its_dict_was_read(self) -> None:
        # CPython creates a partial's ``__dict__``, empty, the first time it is read.
        bound = partial(max, 0)
        before = to_plain(bound)
        _ = bound.__dict__

        assert to_plain(bound) == before

    def test_set_order_is_deterministic(self) -> None:
        assert to_plain({"b", "a", "c"}) == {"py/set": ["a", "b", "c"]}

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

    def test_an_unknown_leaf_names_the_hook_to_add(self) -> None:
        with pytest.raises(TypeError, match=r"hooks=\{lock:"):
            to_plain(threading.Lock())


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
        ],
        ids=repr,
    )
    def test_an_unreducible_container_encodes_by_kind(
        self,
        value: object,
        expected: MutablePlainTree,
    ) -> None:
        assert to_plain(value) == expected

    def test_a_named_reduce_has_no_plain_form(self) -> None:
        with pytest.raises(TypeError, match="NamedReduce"):
            to_plain(NamedReduce())

    def test_an_import_path_naming_another_object_is_rejected(self) -> None:
        def shadow() -> None:
            pass

        shadow.__qualname__ = "max"
        shadow.__module__ = "builtins"

        with pytest.raises(TypeError, match="does not resolve"):
            to_plain(shadow)

    def test_an_unimportable_module_is_rejected(self) -> None:
        def orphan() -> None:
            pass

        orphan.__qualname__ = "orphan"
        orphan.__module__ = "no_such_module_xyz"

        with pytest.raises(TypeError, match="does not resolve"):
            to_plain(orphan)


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
