"""Wire-format readers: text straight to a typed value.

The rest of treekle is wire-invariant; each reader here parses one format and
hands the tree to ``from_plain``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast, overload

import json

from treekle.absent import ABSENT
from treekle.deserialize import from_plain


if TYPE_CHECKING:
    from treekle.types import Hooks, MutablePlainTree


__all__ = [
    "json_loads",
]


@overload
def json_loads(text: str | bytes) -> MutablePlainTree: ...


@overload
def json_loads[T](
    text: str | bytes,
    target: type[T],
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def json_loads(
    text: str | bytes,
    target: object,
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...


def json_loads(
    text: str | bytes,
    target: object = ABSENT,
    *,
    hooks: Hooks = MappingProxyType({}),
    allow_imports: bool = False,
    strict: bool = True,
) -> object:
    """Parse JSON text, as ``target`` when given; ``NaN``/``Infinity`` read too.

    Args:
      text: JSON document.
      target: The type to read the document as; omit for the plain tree.
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
      strict: When false, also read numeric and boolean text.

    Returns:
      value: The document as a ``target``, or its plain tree.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.
      ReadError: The document does not fit ``target``.

    """
    tree = cast("MutablePlainTree", json.loads(text))
    if target is ABSENT:
        return tree
    return from_plain(
        tree,
        target,
        hooks=hooks,
        allow_imports=allow_imports,
        strict=strict,
    )
