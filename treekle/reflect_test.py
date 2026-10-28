"""Tests for ``treekle.reflect``."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableSequence,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass
from email import mime
from email.mime import text as mime_text
from typing import Protocol, cast

import importlib
import os
import sys
import types
import weakref

import pytest

from treekle import reflect
from treekle.reflect import (
    attribute_names,
    is_mutable,
    resolve_import,
    resolves_to,
    slot_names,
)


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
Underscored = type("__Underscored", (), {"__slots__": ("__p",)})
Underscores = type("___", (), {"__slots__": ("__p",)})
Renamed = type("Created", (), {"__slots__": ("__p", "q")})
Renamed.__name__ = "Renamed"
Clashing = type("Clashing", (), {"__slots__": ("__x", "_Clashing__y", "x")})
PrivateWeak = type("PrivateWeak", (), {"__slots__": ("__weakref__", "__p")})
Bereft = type("Bereft", (), {"__slots__": ("__p", "q")})
delattr(Bereft, "_Bereft__p")
Shadowed = type("Shadowed", (), {"__slots__": ("__x", "_Q__x")})


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


class TestResolvesTo:
    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (f"{__name__}.Holder", Holder),
            (f"{__name__}.Holder.__init__", Holder.__init__),
            ("email.mime.text.MIMEText", mime_text.MIMEText),
            ("builtins.max", max),
        ],
        ids=["class", "method", "submodule", "builtin"],
    )
    def test_a_path_naming_the_value_holds(self, path: str, value: object) -> None:
        assert resolves_to(path, value) is True

    @pytest.mark.parametrize(
        "path",
        [
            f"{__name__}.Slotted",
            f"{__name__}.no_such_name",
            "no_such_module_xyz.Holder",
        ],
        ids=repr,
    )
    def test_a_path_naming_anything_else_does_not(self, path: str) -> None:
        assert resolves_to(path, Holder) is False

    def test_a_rebound_name_no_longer_holds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path, original = f"{__name__}.Holder", Holder
        monkeypatch.setitem(globals(), "Holder", Slotted)

        assert resolves_to(path, original) is False

    def test_an_unloaded_module_is_not_imported(self) -> None:
        assert resolves_to("treekle_never_imported_xyz.thing", None) is False
        assert "treekle_never_imported_xyz" not in sys.modules

    def test_a_name_rebound_on_an_outer_class_no_longer_holds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path, original = f"{__name__}.Holder.__init__", Holder.__init__
        assert resolves_to(path, original) is True

        monkeypatch.setattr(Holder, "__init__", Slotted.__init__)

        assert resolves_to(path, original) is False

    def test_it_agrees_with_resolve_import_on_the_longest_loaded_prefix(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A module loaded under the whole path is never the walk's start; one
        # loaded under a longer prefix than the class's module is.
        path = f"{__name__}.Holder.__init__"
        monkeypatch.setitem(sys.modules, path, types.ModuleType(path))
        assert resolves_to(path, Holder.__init__) is (
            resolve_import(path) is Holder.__init__
        )

        shadow = types.ModuleType(f"{__name__}.Holder")
        monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
        assert resolve_import(path) is not Holder.__init__
        assert resolves_to(path, Holder.__init__) is False


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
            # A private slot is named as CPython stores it, mangled by its class.
            (Repeated, ("_Repeated__dict__x", "a")),
            (Twice, ("a2", "_Repeated__dict__x", "a")),
            (Underscored, ("_Underscored__p",)),
            (Underscores, ("__p",)),
            # Mangled by the name the class was created under, not its current one.
            (Renamed, ("_Created__p", "q")),
            # A public slot spelled like another's mangled name stays itself.
            (Clashing, ("_Clashing__x", "_Clashing__y", "x")),
            (PrivateWeak, ("_PrivateWeak__p",)),
            # With every private descriptor deleted, the current name decides.
            (Bereft, ("_Bereft__p", "q")),
            # A public slot ending in a private one's name is not its mangling.
            (Shadowed, ("_Shadowed__x", "_Q__x")),
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


class Mixed:
    """Slots across two classes plus a ``__dict__``, declared out of name order."""

    __slots__ = ("__dict__", "beta", "zeta")

    def __init__(self) -> None:
        self.zeta = 1
        self.beta = 2
        self.mid = 3
        self.alpha = 4


class MixedChild(Mixed):
    __slots__ = ("gamma",)

    def __init__(self) -> None:
        super().__init__()
        self.gamma = 5


class TestAttributeNames:
    def test_names_come_sorted_across_slots_bases_and_dict(self) -> None:
        assert list(attribute_names(MixedChild())) == [
            "alpha",
            "beta",
            "gamma",
            "mid",
            "zeta",
        ]

    def test_the_declaring_class_does_not_change_the_order(self) -> None:
        moved = type("Moved", (), {"__slots__": ("beta", "gamma", "zeta", "__dict__")})
        value = moved()
        for name, item in (("beta", 2), ("gamma", 5), ("zeta", 1)):
            setattr(value, name, item)
        vars(value).update(mid=3, alpha=4)

        assert list(attribute_names(value)) == list(attribute_names(MixedChild()))

    def test_unsorted_keeps_slots_in_mro_order_then_dict_keys(self) -> None:
        assert list(attribute_names(MixedChild(), sort=False)) == [
            "gamma",
            "beta",
            "zeta",
            "mid",
            "alpha",
        ]

    def test_a_name_in_both_a_slot_and_the_dict_comes_once(self) -> None:
        value = Slotted()
        vars(value)["a"] = 9

        assert list(attribute_names(value)) == ["a"]

    def test_an_unset_slot_is_still_named(self) -> None:
        assert list(attribute_names(WeakSlot())) == ["w"]

    @pytest.mark.parametrize(
        "value",
        [1, "s", [1], object()],
        ids=["int", "str", "list", "object"],
    )
    def test_a_value_without_attributes_names_none(self, value: object) -> None:
        assert list(attribute_names(value)) == []

    def test_an_empty_instance_dict_names_only_the_slots(self) -> None:
        assert list(attribute_names(Slotted())) == ["a"]
        assert list(attribute_names(types.SimpleNamespace())) == []

    def test_a_class_without_a_dict_never_asks_getattr_for_one(self) -> None:
        asked: list[str] = []

        class Hooked:
            __slots__ = ("a",)

            def __init__(self) -> None:
                self.a = 1

            def __getattr__(self, name: str) -> object:
                asked.append(name)
                return {"extra": 2}

        assert list(attribute_names(Hooked())) == ["a"]
        assert asked == []

    def test_a_class_with_no_slots_sorts_them_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sorted_slots = reflect._sorted_slots
        sorted_for: list[type] = []

        def counted(kind: type) -> tuple[str, ...]:
            sorted_for.append(kind)
            return sorted_slots(kind)

        monkeypatch.setattr(reflect, "_sorted_slots", counted)
        kind = type("NoSlots", (), {})

        assert list(attribute_names(kind())) == []
        assert list(attribute_names(kind())) == []
        assert sorted_for == [kind]


class TestClassTags:
    def test_each_class_maps_to_its_name(self) -> None:
        assert reflect.class_tags({"h": Holder, "s": Slotted}) == {
            Holder: "h",
            Slotted: "s",
        }

    @pytest.mark.parametrize(
        ("classes", "error", "match"),
        [
            ({1: Holder}, TypeError, "keys must be str, not int"),
            ({"h": Slotted()}, TypeError, r"classes\['h'\] is not a class"),
            ({"a": Holder, "b": Holder}, ValueError, "one class twice: 'a' and 'b'"),
        ],
        ids=["non-str key", "non-class value", "one class twice"],
    )
    def test_a_bad_mapping_is_refused(
        self,
        classes: Mapping[object, object],
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            reflect.class_tags(classes)


class TestConcreteOverrides:
    def test_only_entries_off_the_baseline_are_kept(self) -> None:
        overrides = reflect.concrete_overrides(
            {Sequence: list, MutableSequence: list, AbstractSet: set},
        )

        assert overrides == {AbstractSet: set}

    def test_a_generic_alias_names_its_abc(self) -> None:
        alias = cast("type", Sequence[int])

        assert reflect.concrete_overrides({alias: tuple}) == {Sequence: tuple}

    @pytest.mark.parametrize(
        ("concrete", "match"),
        [
            ({int: int}, "keys must be one of .* not <class 'int'>"),
            ({Sequence: str}, r"default_concrete_types\[Sequence\] must be one of"),
        ],
        ids=["not a baseline key", "not a buildable subclass"],
    )
    def test_a_bad_entry_is_refused(
        self,
        concrete: dict[type, type],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            reflect.concrete_overrides(concrete)


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
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
