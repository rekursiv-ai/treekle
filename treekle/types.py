"""Treekle's public types: plain trees, codec hooks, and read failures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast, override


__all__ = [
    "Dialect",
    "FieldPath",
    "Hooks",
    "Invalid",
    "MutablePlainTree",
    "Plain",
    "PlainTree",
    "ReadError",
]


# A leaf of a plain tree. ``float`` includes IEEE-754 NaN and signed infinities.
# This is not Python PODs but rather types supported by JSON, TOML, YAML, msgpack etc.
type Plain = bool | int | float | str | None


type PlainTree = Plain | Sequence[PlainTree] | Mapping[str, PlainTree]
"""Any plain tree, read-only: dicts and lists, or ``MappingProxyType`` and tuples."""


type MutablePlainTree = Plain | list[MutablePlainTree] | dict[str, MutablePlainTree]
"""A plain tree of dicts and lists, which may be edited."""


type Hooks = Mapping[type, tuple[Callable[..., object], Callable[..., object]]]
"""Runtime types paired with their ``(encode, decode)`` callbacks."""


type FieldPath = tuple[str | int, ...]
"""Where a value sits: field names and indices leading to it; ``()`` is the value."""


type Dialect = Literal["json", "python"]
"""A target format; it decides which values ``to_plain`` must tag.

``"json"`` tags non-finite floats; ``"python"`` keeps them native. Both tag
everything else alike, so either result fits the same tree types.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class Invalid:
    """A value that did not match its target type, kept as it was stated."""

    raw: PlainTree
    reason: str = ""


class ReadError(TypeError, ValueError):
    """Plain data does not match the requested type.

    Attributes:
      partial: The value as far as it reads, an :class:`Invalid` in place of
        each bad part; the whole value is one when nothing reads.
      bad: Each bad part, by where it sits.

    """

    partial: object
    bad: Mapping[FieldPath, Invalid]

    def __init__(
        self,
        message: str,
        *,
        partial: object = None,
        bad: Mapping[FieldPath, Invalid] = MappingProxyType({}),
    ) -> None:
        super().__init__(message)
        self.partial = partial
        self.bad = bad

    # ``BaseException`` pickles its ``args`` alone and rebuilds by calling the
    # class on them, so a process pool would hand back an error with neither
    # ``partial`` nor ``bad``, or fail on a subclass whose ``__init__`` takes
    # more; and ``bad`` is often a ``MappingProxyType``, which cannot pickle.
    @override
    def __reduce__(self) -> tuple[object, ...]:
        state = dict(cast("dict[str, object]", vars(self)))
        state["bad"] = dict(self.bad)
        return (_rebuilt_read_error, (type(self), self.args), state)

    @override
    def __setstate__(self, state: object) -> None:
        """Restore what ``__reduce__`` kept, ``bad`` read-only again."""
        assert isinstance(state, dict)
        kept = cast("dict[str, object]", state)
        bad = cast("dict[FieldPath, Invalid]", kept.pop("bad", {}))
        vars(self).update(kept)
        self.bad = MappingProxyType(bad)


def _rebuilt_read_error(kind: type[ReadError], args: tuple[object, ...]) -> ReadError:
    """Allocate a pickled :class:`ReadError` without its ``__init__``."""
    error = kind.__new__(kind)
    error.args = args
    return error
