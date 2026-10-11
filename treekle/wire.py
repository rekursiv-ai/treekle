"""Wire-format readers: text to a plain tree, typed for ``from_plain``.

The rest of treekle is wire-invariant; each reader here only names its
parser's untyped result as a :data:`MutablePlainTree`.
"""

from __future__ import annotations

from typing import cast

import json

from treekle.types import MutablePlainTree


__all__ = [
    "json_loads",
]


def json_loads(text: str | bytes) -> MutablePlainTree:
    """Parse JSON text into a mutable plain tree; ``NaN``/``Infinity`` read too.

    Args:
      text: JSON document.

    Returns:
      tree: Plain dicts, lists, and scalars.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.

    """
    return cast(MutablePlainTree, json.loads(text))
