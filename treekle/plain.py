"""Plain trees: the data ``to_plain`` makes and ``from_plain`` reads.

Plain data is ``str``, ``int``, ``float``, ``bool``, ``None``, and mappings and
sequences of them. :data:`PlainTree` is any such tree, read-only;
:data:`MutablePlainTree` is the dict-and-list form, which is also a
``PlainTree``.

``immutable`` and ``mutable`` copy a plain tree to ``MappingProxyType`` and
tuples, or to dicts and lists, whichever form it starts in. Both always copy.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, overload


if TYPE_CHECKING:
    from treekle.types import MutablePlainTree, Plain, PlainTree


__all__ = [
    "immutable",
    "mutable",
]


@overload
def immutable(tree: Plain) -> Plain: ...


@overload
def immutable(tree: Mapping[str, object]) -> Mapping[str, PlainTree]: ...


@overload
def immutable(tree: Sequence[object]) -> Sequence[PlainTree]: ...


@overload
def immutable(tree: object) -> PlainTree: ...


def immutable(tree: object) -> PlainTree:
    """Copy a plain tree: mappings to ``MappingProxyType``, sequences to tuples.

    Args:
      tree: A plain tree, mutable or not.

    Returns:
      frozen: The read-only copy.

    Raises:
      TypeError: ``tree`` holds a non-str key or a value that is not plain.

    """
    return cast("PlainTree", _rebuild(tree, sequence=tuple, mapping=MappingProxyType))


@overload
def mutable(tree: Plain) -> Plain: ...


@overload
def mutable(tree: Mapping[str, object]) -> dict[str, MutablePlainTree]: ...


@overload
def mutable(tree: Sequence[object]) -> list[MutablePlainTree]: ...


@overload
def mutable(tree: object) -> MutablePlainTree: ...


def mutable(tree: object) -> MutablePlainTree:
    """Copy a plain tree: each mapping to a dict, each sequence to a list.

    Args:
      tree: A plain tree, mutable or not.

    Returns:
      thawed: The mutable copy.

    Raises:
      TypeError: ``tree`` holds a non-str key or a value that is not plain.

    """
    return cast("MutablePlainTree", _rebuild(tree, sequence=list, mapping=dict))


type _Frame = tuple[int, list[str] | None, list[object], Iterator[object]]
"""A node being copied: its id, keys (``None`` for a sequence), copied items, the rest."""


# A frame per open node rather than recursion, so a tree of any depth copies.
def _rebuild(
    tree: object,
    *,
    sequence: Callable[[list[object]], object],
    mapping: Callable[[dict[str, object]], object],
) -> object:
    """Copy ``tree`` bottom-up, building each node through ``sequence``/``mapping``."""
    if tree is None or isinstance(tree, (str, int, float)):
        return tree
    stack = [_open(tree)]
    open_ids = {stack[0][0]}
    while True:
        identity, keys, items, rest = stack[-1]
        for item in rest:
            if item is None or isinstance(item, (str, int, float)):
                items.append(item)
            elif id(item) in open_ids:
                raise TypeError("plain data cannot hold a cycle")
            else:
                stack.append(_open(item))
                open_ids.add(id(item))
                break
        else:
            stack.pop()
            open_ids.discard(identity)
            node = (
                sequence(items)
                if keys is None
                else mapping(dict(zip(keys, items, strict=True)))
            )
            if not stack:
                return node
            stack[-1][2].append(node)


def _open(node: object) -> _Frame:
    """Start copying ``node``, a plain mapping or sequence."""
    identity = id(node)
    if isinstance(node, Mapping):
        entries = cast("Mapping[object, object]", node)
        return (
            identity,
            [_plain_key(key) for key in entries],
            [],
            iter(entries.values()),
        )
    return identity, None, [], iter(_plain_sequence(node))


def _plain_key(key: object) -> str:
    """Return ``key`` when it is a str, the only key plain data allows."""
    if isinstance(key, str):
        return key
    raise TypeError(f"plain mapping key must be str, got {key!r}")


def _plain_sequence(tree: object) -> Sequence[object]:
    """Return ``tree`` when it is a plain sequence."""
    if isinstance(tree, Sequence) and not isinstance(tree, (bytes, bytearray)):
        return tree
    raise TypeError(f"cannot represent {type(tree).__name__} as plain data")
