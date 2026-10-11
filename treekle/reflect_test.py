"""Tests for ``treekle.reflect``."""

from __future__ import annotations

from dataclasses import dataclass
from email import mime
from email.mime import text as mime_text
from typing import Protocol

import importlib
import os
import sys
import types
import weakref

import pytest

from treekle import reflect
from treekle.reflect import is_mutable, resolve_import, slot_names


class Shape(Protocol):
    def area(self) -> float: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class Holder:
    shape: Shape


class Slotted:
    __slots__ = ("__dict__", "a")

    def __init__(self) -> None:
        self.a = 1


class SlottedChild(Slotted):
    __slots__ = ("b",)

    def __init__(self) -> None:
        super().__init__()
        self.b = 2


# Python takes a bare string as one slot name, not as its letters; built through
# ``type`` because a class body spelling it is linted away (PLC0205).
StringSlot = type("StringSlot", (), {"__slots__": "only"})


# Built through ``type``: these classes exist only for their slot declarations, so
# nothing assigns the slots a class body would make the checker expect set.
Repeated = type("Repeated", (Slotted,), {"__slots__": ("__dict__x",)})
Twice = type("Twice", (Repeated,), {"__slots__": ("a2",)})
WeakSlot = type("WeakSlot", (), {"__slots__": ("__weakref__", "w")})


class Unslotted:
    pass


class TestResolveImport:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("os.path.join", os.path.join),
            ("email.mime.text.MIMEText.__init__", mime_text.MIMEText.__init__),
            ("builtins.max", max),
        ],
        ids=["os.path.join", "MIMEText.__init__", "builtins.max"],
    )
    def test_a_loaded_module_resolves_without_an_import(
        self,
        path: str,
        expected: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attempts: list[str] = []

        def refuse(name: str) -> object:
            attempts.append(name)
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", refuse)

        assert resolve_import(path) is expected
        assert attempts == []

    @pytest.mark.parametrize("depth", [1, 2, 3], ids=repr)
    def test_the_loaded_walk_tries_each_shorter_prefix(
        self,
        depth: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Only a prefix ``depth`` segments short of the full path is loaded, so the
        # walk must step down one segment at a time to reach it.
        names = ["treekle_test_root", "b", "c", "d"]
        prefix = ".".join(names[: len(names) - depth])
        module = types.ModuleType(prefix)
        node: object = module
        for name in names[len(names) - depth : -1]:
            child = types.SimpleNamespace()
            monkeypatch.setattr(node, name, child, raising=False)
            node = child
        monkeypatch.setattr(node, names[-1], Holder, raising=False)
        monkeypatch.setitem(sys.modules, prefix, module)

        def refuse(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", refuse)

        assert resolve_import(".".join(names)) is Holder

    def test_the_full_path_is_never_looked_up_as_a_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A module registered under the whole dotted path is not what the path names.
        real = types.ModuleType("treekle_test_full")
        monkeypatch.setattr(real, "leaf", Holder, raising=False)
        monkeypatch.setitem(sys.modules, "treekle_test_full", real)
        monkeypatch.setitem(
            sys.modules,
            "treekle_test_full.leaf",
            types.ModuleType("treekle_test_full.leaf"),
        )

        assert resolve_import("treekle_test_full.leaf") is Holder

    def test_a_loaded_ancestor_lacking_the_rest_stops_the_loaded_walk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``os`` is loaded but holds no ``no_such``; the walk must not retry the
        # shorter loaded prefix ``os`` and must go on to import, which fails.
        attempts: list[str] = []
        real = importlib.import_module

        def spy(name: str) -> object:
            attempts.append(name)
            return real(name)

        monkeypatch.setattr(importlib, "import_module", spy)

        with pytest.raises(AttributeError, match=r"names no attribute of 'os'$"):
            resolve_import("os.no_such.thing")

        assert attempts == ["os.no_such", "os"]

    def test_an_import_tries_the_longest_prefix_first(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attempts: list[str] = []

        def refuse(name: str) -> object:
            attempts.append(name)
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", refuse)

        with pytest.raises(ImportError, match=r"^Cannot resolve path: 'a\.b\.c\.d'$"):
            resolve_import("a.b.c.d")

        assert attempts == ["a.b.c", "a.b", "a"]

    def test_an_import_falls_back_past_a_failing_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``pkg`` imports and holds ``sub.thing``, but ``pkg.sub`` is no module.
        package = types.SimpleNamespace(
            __name__="treekle_test_pkg",
            sub=types.SimpleNamespace(thing=Holder),
        )
        attempts: list[str] = []

        def only_package(name: str) -> object:
            attempts.append(name)
            if name != "treekle_test_pkg":
                raise ImportError(name)
            return package

        monkeypatch.setattr(importlib, "import_module", only_package)

        assert resolve_import("treekle_test_pkg.sub.thing") is Holder
        assert attempts == ["treekle_test_pkg.sub", "treekle_test_pkg"]

    def test_a_nested_class_resolves_without_an_import_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attempts: list[str] = []
        real = importlib.import_module

        def spy(name: str) -> object:
            attempts.append(name)
            return real(name)

        monkeypatch.setattr(importlib, "import_module", spy)

        assert resolve_import(f"{__name__}.Holder.__init__") is (Holder.__init__)
        assert attempts == []

    def test_an_unloaded_submodule_is_imported(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The loaded parent lacks the attribute until its submodule imports.
        monkeypatch.delattr(mime, "text")
        monkeypatch.delitem(sys.modules, "email.mime.text")

        resolved = resolve_import("email.mime.text.MIMEText")

        assert isinstance(resolved, type)
        assert resolved.__name__ == mime_text.MIMEText.__name__

    def test_a_missing_attribute_of_a_loaded_module_raises(self) -> None:
        with pytest.raises(AttributeError, match=f"no attribute of '{__name__}'"):
            resolve_import(f"{__name__}.no_such_name")

    def test_an_unimportable_path_raises(self) -> None:
        with pytest.raises(ImportError, match="no_such_module_xyz"):
            resolve_import("no_such_module_xyz.thing")


class TestSlotNames:
    @pytest.fixture(autouse=True)
    def empty_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start each test uncached, so a slot list is computed, not recalled."""
        monkeypatch.setattr(
            reflect,
            "_SLOT_NAMES",
            weakref.WeakKeyDictionary[type, tuple[str, ...]](),
        )

    def test_slots_list_in_mro_order_without_bookkeeping(self) -> None:
        assert slot_names(SlottedChild) == ("b", "a")

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (StringSlot, ("only",)),
            (Repeated, ("__dict__x", "a")),
            (Twice, ("a2", "__dict__x", "a")),
            (WeakSlot, ("w",)),
            (Unslotted, ()),
            (int, ()),
        ],
        ids=repr,
    )
    def test_each_declared_slot_is_listed_once(
        self,
        kind: type,
        expected: tuple[str, ...],
    ) -> None:
        assert slot_names(kind) == expected

    def test_the_result_is_cached(self) -> None:
        first = slot_names(Slotted)

        assert slot_names(Slotted) is first
        assert reflect._SLOT_NAMES[Slotted] is first

    def test_a_cached_result_is_returned_unrecomputed(self) -> None:
        reflect._SLOT_NAMES[Slotted] = ("planted",)

        assert slot_names(Slotted) == ("planted",)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (bytearray(b"a"), True),
        ([1], True),
        ({"a": 1}, True),
        (b"a", False),
        ((1,), False),
        (1.5, False),
        # repr(object()) holds an address, which differs per xdist worker.
        pytest.param(object(), False, id="object()-False"),
    ],
    ids=repr,
)
def test_a_class_that_opts_out_of_hashing_is_mutable(
    value: object,
    expected: bool,
) -> None:
    assert is_mutable(value) is expected


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
