"""Read what a person typed, such as a command-line override.

``parse(text, T)`` is lenient about syntax and strict about type: it accepts
JSON, falls back to the bare word, and lets ``T`` decide, so
``parse("123", str)`` is ``"123"``. The result then passes through
``from_plain``, so text that does not fit ``T`` raises :class:`ReadError`, never
a default.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePath
from typing import TYPE_CHECKING, cast, overload
from uuid import UUID

import json
import math

from treekle.deserialize import from_plain
from treekle.types import ReadError


if TYPE_CHECKING:
    from treekle.types import MutablePlainTree


__all__ = [
    "parse",
]


@overload
def parse[T](text: str, target: type[T]) -> T: ...


@overload
def parse(text: str, target: object) -> object: ...


def parse(text: str, target: object) -> object:
    """Read typed ``text`` as ``target``, accepting JSON or the bare word.

    Args:
      text: What a person typed, e.g. ``3e-4``, ``run_a``, or ``[1, 2]``.
      target: The type to produce.

    Returns:
      value: ``text`` as a ``target``.

    Raises:
      ReadError: ``text`` does not read as ``target``.

    """
    plain = _text_to_plain(text, target)
    try:
        return from_plain(plain, target)
    except ReadError as error:
        if plain == text:
            raise
        # The bare word is the fallback reading; when it fails too, the JSON
        # reading's error says more about what the person meant.
        try:
            return from_plain(text, target)
        except ReadError:
            raise error from None


def _text_to_plain(text: str, target: object) -> MutablePlainTree:
    """Turn ``text`` into a plain tree the way ``target`` reads it."""
    if _is_verbatim(target):
        loaded = _loads_or_word(text)
        return loaded if isinstance(loaded, str) else text
    number = _non_finite(text)
    if number is not None:
        return number
    loaded = _loads_or_word(text)
    # Quoting is syntax: ``count="7"`` reads the quoted text as typed.
    if isinstance(loaded, str) and loaded != text:
        return _text_to_plain(loaded, target)
    return loaded


def _is_verbatim(target: object) -> bool:
    """Return whether ``target`` reads the raw text, e.g. str, Path, or UUID."""
    return isinstance(target, type) and issubclass(
        target,
        (str, PurePath, UUID, datetime),
    )


def _loads_or_word(text: str) -> MutablePlainTree:
    """Return ``text`` as JSON, or the bare word when it is not JSON."""
    try:
        return cast("MutablePlainTree", json.loads(text))
    except json.JSONDecodeError:
        return text


def _non_finite(text: str) -> float | None:
    """Return the float ``inf``, ``-inf``, or ``nan`` spells, else ``None``."""
    try:
        number = float(text.strip())
    except ValueError:
        return None
    return None if math.isfinite(number) else number
