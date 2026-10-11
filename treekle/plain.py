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
    return _freeze(tree)


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
    return _thaw(tree)


def _freeze(tree: object) -> PlainTree:
    """Copy ``tree`` to ``MappingProxyType`` and tuples at every level."""
    if tree is None or isinstance(tree, (str, int, float)):
        return tree
    if isinstance(tree, Mapping):
        node = cast("Mapping[object, object]", tree)
        return MappingProxyType(
            {_plain_key(key): _freeze(item) for key, item in node.items()},
        )
    return tuple(_freeze(item) for item in _plain_sequence(tree))


def _thaw(tree: object) -> MutablePlainTree:
    """Copy ``tree`` to dicts and lists at every level."""
    if tree is None or isinstance(tree, (str, int, float)):
        return tree
    if isinstance(tree, Mapping):
        node = cast("Mapping[object, object]", tree)
        return {_plain_key(key): _thaw(item) for key, item in node.items()}
    return [_thaw(item) for item in _plain_sequence(tree)]


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
