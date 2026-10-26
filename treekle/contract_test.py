"""Tests that treekle keeps the contract its package docstring states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from types import MappingProxyType, ModuleType
from typing import TypedDict, cast

import doctest
import math
import sys

import pytest

from treekle.deserialize import from_plain
from treekle.serialize import to_plain

import treekle


pytestmark = pytest.mark.usefixtures("implementation")


class Bound(Enum):
    TOP = math.inf


class Named(TypedDict):
    name: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Inner:
    n: int = 0


@dataclass(frozen=True, kw_only=True, slots=True)
class Outer:
    inner: Inner = field(default_factory=Inner)


@dataclass(frozen=True, kw_only=True, slots=True)
class Fields:
    """Fields each holding a value that is not of its declared type."""

    ratio: float = 0.0
    pair: tuple[int, str] = (0, "")
    named: Named | None = None
    inner: Inner | None = None
    bound: Bound | None = None


def _kinds(value: object) -> object:
    """Return ``value`` with each node shown as its type and its parts' kinds."""
    if isinstance(value, Fields):
        return type(value), {
            "ratio": _kinds(value.ratio),
            "pair": _kinds(value.pair),
            "named": _kinds(value.named),
            "inner": _kinds(value.inner),
            "bound": _kinds(value.bound),
        }
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return type(mapping), {key: _kinds(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return type(items), [_kinds(item) for item in items]
    return type(value), value


def _mistyped(**changes: object) -> Fields:
    """Return ``Fields`` holding ``changes``, which its field types do not declare."""
    return replace(Fields(), **changes)  # pyrefly: ignore[bad-argument-type] -- the cases are values their declared types reject, by design.


_MISTYPED: list[Fields] = [
    _mistyped(ratio=Decimal("1.5")),
    _mistyped(pair=(1, "b", (2,))),
    _mistyped(pair=(1,)),
    _mistyped(named={"name": "a", "extra": 1}),
    _mistyped(inner=Outer()),
]


class TestAValueNotOfItsDeclaredTypeIsTagged:
    @pytest.mark.parametrize("value", _MISTYPED, ids=repr)
    @pytest.mark.parametrize("target", [object, Fields], ids=repr)
    def test_it_round_trips_with_every_type_kept(
        self,
        value: Fields,
        target: object,
    ) -> None:
        tree = to_plain(value, target=target)
        back = from_plain(tree, target, allow_imports=True)

        assert _kinds(back) == _kinds(value)

    def test_a_non_finite_enum_value_keeps_its_tag(self) -> None:
        fields = Fields(bound=Bound.TOP)

        tree = to_plain(fields, target=Fields)

        assert isinstance(tree, dict)
        assert isinstance(tree["bound"], dict)
        assert from_plain(tree, Fields) == fields


def test_a_union_member_whose_items_do_not_fit_is_not_implied() -> None:
    target = tuple[list[int], ...] | object

    tree = to_plain((0,), target=target)

    assert tree == {"py/tuple": [0]}
    assert from_plain(tree, target) == (0,)


def test_a_tagged_union_member_reads_its_parts_as_written() -> None:
    target = object | Mapping[str, Mapping[str, int]]
    frozen: Mapping[type, type] = {Mapping: MappingProxyType}

    tree = to_plain({"a": {}}, target=target, default_concrete_types=frozen)
    back = from_plain(tree, target, default_concrete_types=frozen)

    assert _kinds(back) == _kinds({"a": {}})


class TestClasses:
    def test_an_implied_record_need_not_be_listed(self) -> None:
        tree = to_plain(Outer(inner=Inner(n=1)), classes={"outer": Outer})

        assert tree == {"py/object": "outer", "inner": {"n": 1}}
        assert from_plain(tree, Outer, classes={"outer": Outer}) == Outer(
            inner=Inner(n=1),
        )


def test_the_package_docstring_example_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    # The example's classes are written as ``__main__``'s, as a script's are, so
    # their import paths resolve through the module the example runs in.
    main = ModuleType("__main__")
    monkeypatch.setitem(sys.modules, "__main__", main)
    (example,) = doctest.DocTestFinder().find(treekle)
    # ``find`` runs on a copy of the globals; the classes must land in ``main``.
    example.globs = vars(main)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)

    runner.run(example)

    assert runner.summarize(verbose=False).failed == 0


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
