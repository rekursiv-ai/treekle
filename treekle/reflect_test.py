"""Tests for ``treekle.reflect``."""

from __future__ import annotations

from dataclasses import dataclass
from email import mime
from email.mime import text as mime_text
from typing import Protocol

import importlib
import sys

import pytest

from treekle.reflect import resolve_import, slot_names


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


class TestResolveImport:
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
    def test_slots_list_in_mro_order_without_bookkeeping(self) -> None:
        assert slot_names(SlottedChild) == ("b", "a")

    def test_the_result_is_cached(self) -> None:
        assert slot_names(Slotted) is slot_names(Slotted)


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
