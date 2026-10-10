"""Tests for ``treekle.plain``."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import json
import math

import pytest

from treekle.plain import immutable, loads, mutable


if TYPE_CHECKING:
    from treekle.types import Plain


class TestImmutableMutable:
    def test_immutable_freezes_every_level(self) -> None:
        frozen = immutable({"a": [1, {"b": [2]}]})

        assert isinstance(frozen, MappingProxyType)
        inner = frozen["a"]
        assert isinstance(inner, tuple)
        assert isinstance(inner[1], MappingProxyType)
        assert inner[1]["b"] == (2,)

    def test_mutable_thaws_every_level(self) -> None:
        thawed = mutable(MappingProxyType({"a": (1, MappingProxyType({"b": (2,)}))}))

        assert thawed == {"a": [1, {"b": [2]}]}

    def test_both_always_copy(self) -> None:
        source = {"a": [1]}

        thawed = mutable(source)

        assert thawed == source
        assert thawed is not source
        assert thawed["a"] is not source["a"]

    @pytest.mark.parametrize("leaf", [None, True, 1, 1.5, "s"])
    def test_leaves_pass_through(self, leaf: Plain) -> None:
        assert immutable(leaf) is leaf
        assert mutable(leaf) is leaf


class TestLoads:
    def test_text_reads_as_a_mutable_tree(self) -> None:
        assert loads('{"a": [1, 2.5, null]}') == {"a": [1, 2.5, None]}
        assert loads(b"[true]") == [True]

    def test_non_finite_floats_read(self) -> None:
        loaded = loads("[NaN, Infinity]")
        assert isinstance(loaded, list)
        assert math.isnan(cast("float", loaded[0]))
        assert loaded[1] == math.inf

    def test_bad_text_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            loads("{")


class TestFreezeChecks:
    def test_immutable_and_mutable_accept_any_plain_object(self) -> None:
        source: dict[str, object] = {"a": [1, {"b": None}]}
        assert immutable(source) == MappingProxyType(
            {"a": (1, MappingProxyType({"b": None}))},
        )
        assert mutable(cast("object", (1, "x"))) == [1, "x"]

    @pytest.mark.parametrize(
        "bad",
        [{1: "a"}, [object()], {"a": b"x"}],
        ids=["int-key", "object", "bytes"],
    )
    def test_a_non_plain_part_raises(self, bad: object) -> None:
        with pytest.raises(TypeError):
            immutable(bad)
        with pytest.raises(TypeError):
            mutable(bad)


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
