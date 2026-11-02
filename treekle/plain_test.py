"""Tests for ``treekle.plain``."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from treekle.lib.testing.resources import recursion_headroom
from treekle.plain import immutable, mutable


if TYPE_CHECKING:
    from treekle.types import MutablePlainTree, Plain


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

    def test_a_tree_deeper_than_the_recursion_limit_copies(self) -> None:
        depth = 200
        tree: object = "leaf"
        for index in range(depth):
            tree = list[object]([tree]) if index % 2 else dict[str, object](k=tree)

        with recursion_headroom(100):
            node: MutablePlainTree = mutable(immutable(tree))
        for _ in range(depth):
            assert isinstance(node, (list, dict))
            node = node[0] if isinstance(node, list) else node["k"]
        assert node == "leaf"


class TestFreezeChecks:
    def test_immutable_and_mutable_accept_any_plain_object(self) -> None:
        source: dict[str, object] = {"a": [1, {"b": None}]}
        assert immutable(source) == MappingProxyType(
            {"a": (1, MappingProxyType({"b": None}))},
        )
        assert mutable(cast("object", (1, "x"))) == [1, "x"]

    @pytest.mark.parametrize(
        ("bad", "says"),
        [
            ({1: "a"}, "plain mapping key must be str, got 1"),
            ([object()], "cannot represent object as plain data"),
            ({"a": b"x"}, "cannot represent bytes as plain data"),
        ],
        ids=["int-key", "object", "bytes"],
    )
    def test_a_non_plain_part_raises(self, bad: object, says: str) -> None:
        with pytest.raises(TypeError) as frozen:
            immutable(bad)
        with pytest.raises(TypeError) as thawed:
            mutable(bad)
        assert str(frozen.value) == str(thawed.value) == says

    def test_a_cyclic_tree_raises(self) -> None:
        cyclic: list[object] = []
        cyclic.append({"k": cyclic})

        with pytest.raises(TypeError) as frozen:
            immutable(cyclic)
        with pytest.raises(TypeError) as thawed:
            mutable(cyclic)
        assert (
            str(frozen.value) == str(thawed.value) == "plain data cannot hold a cycle"
        )

    def test_a_cycle_below_the_root_raises(self) -> None:
        inner: list[object] = []
        inner.append(inner)

        with pytest.raises(TypeError, match="cycle"):
            mutable({"a": inner})

    def test_a_shared_subtree_copies_at_each_use(self) -> None:
        shared = [1]

        assert mutable({"a": shared, "b": [shared]}) == {"a": [1], "b": [[1]]}


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
