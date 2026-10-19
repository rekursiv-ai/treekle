"""Treekle: any Python object graph as a plain tree, and back.

``to_plain`` turns any Python object graph -- dataclasses, functions, shared
references, cycles -- into a plain tree of dicts, lists, and scalars that any
JSON, TOML, or YAML writer can store. ``from_plain`` turns it back, building the
classes the tree names, and reads bad input best-effort: the error carries the
partial value and the location of every bad part.

Modules:

- ``absent``: ``ABSENT``, ``Absent``.
- ``types``: every public type -- the plain trees, ``Hooks``, ``FieldPath``,
  ``Dialect``, ``Invalid``, ``ReadError``.
- ``plain``: ``mutable``, ``immutable``.
- ``wire``: ``json_loads``, the one wire-format reader.
- ``serialize``: ``to_plain``.
- ``deserialize``: ``from_plain``.
- ``parse``: ``parse``, for what a person typed.

What a dialect cannot carry natively is written as a ``py/*`` tag:

- ``py/object``, ``py/type``, ``py/function``: a class instance, a class, or a
  function, by dotted import path.
- ``py/tuple``, ``py/set``, ``py/frozenset``, ``py/mappingproxy``: containers a
  dialect flattens to lists or dicts.
- ``py/float``, ``py/complex``, ``py/b64``, ``py/path``, ``py/uuid``,
  ``py/datetime``: values a dialect lacks.
- ``py/reduce``, ``py/hook``: a ``__reduce__`` recipe and a caller-supplied codec.
- ``py/ref``: a back-reference by the path of the object's first occurrence, so a
  shared object or a cycle survives. ``py/id``, the older back-reference by
  encounter number, still reads.
- ``json://``: a dict key that is not a string, or that reads like a tag.

A class named in the data means an import and code run on read; treekle
gates that behind ``allow_imports``. One cycle shape fails to read: one
through a dataclass decoded as its declared type.
"""

from __future__ import annotations

from treekle.absent import ABSENT, Absent
from treekle.deserialize import from_plain
from treekle.parse import parse
from treekle.plain import immutable, mutable
from treekle.serialize import to_plain
from treekle.types import (
    Dialect,
    FieldPath,
    Hooks,
    Invalid,
    MutablePlainTree,
    Plain,
    PlainTree,
    ReadError,
)
from treekle.wire import json_loads


__all__ = [
    "ABSENT",
    "Absent",
    "Dialect",
    "FieldPath",
    "Hooks",
    "Invalid",
    "MutablePlainTree",
    "Plain",
    "PlainTree",
    "ReadError",
    "from_plain",
    "immutable",
    "json_loads",
    "mutable",
    "parse",
    "to_plain",
]
