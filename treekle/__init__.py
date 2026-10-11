"""Treekle: pickle semantics as plain python trees.

``to_plain`` turns any Python object graph -- dataclasses, functions, shared
references, cycles -- into a plain tree of dicts, lists, and scalars that any
JSON, TOML, or YAML writer can store. ``from_plain`` turns it back, building the
classes the tree names, and reads bad input best-effort: the error carries the
partial value and the location of every bad part.

Modules:

- ``types``: every public type -- the plain trees, ``Hooks``, ``FieldPath``,
  ``Dialect``, ``Invalid``, ``ReadError``.
- ``plain``: ``mutable``, ``immutable``.
- ``wire``: ``json_loads``, the one wire-format reader; not re-exported.
- ``serialize``: ``to_plain``.
- ``deserialize``: ``from_plain``.
- ``parse``: ``parse``, for what a person typed.

What a dialect cannot carry natively is written as a ``py/*`` tag, using
jsonpickle's spelling where one exists:

- ``py/object``, ``py/type``, ``py/function``: a class instance, a class, or a
  function, by dotted import path.
- ``py/tuple``, ``py/set``, ``py/frozenset``, ``py/mappingproxy``: containers a
  dialect flattens to lists or dicts.
- ``py/float``, ``py/complex``, ``py/b64``, ``py/path``, ``py/uuid``,
  ``py/datetime``: values a dialect lacks.
- ``py/reduce``, ``py/hook``: a pickle reduce recipe and a caller-supplied codec.
- ``py/ref``: a back-reference by the path of the object's first occurrence, so a
  shared object or a cycle survives. ``py/id``, the older back-reference by
  encounter number, still reads.
- ``json://``: a dict key that is not a string, or that reads like a tag.

Why not another serializer:

tl;dr: No library is perfect, including this one.

library     C/Rust  text  class from data  callables  shared refs  best-effort
----------  ------  ----  ---------------  ---------  -----------  -----------
msgspec     ✓       ✓     ✗                ✗          ✗            ✗
pydantic    ✓       ✓     ✗                ✗          ✗            ✗
cattrs      ✗       ✓     ✗                ✗          ✗            ✗
mashumaro   ✗       ✓     ✗                ✗          ✗            ✗
pickle      ✓       ✗     ✓                ✓          ✓            ✗
jsonpickle  ✗       ✓     ✓                ✓          ✓            ✗
treekle     ✗       ✓     ✓                ✓          ✓            ✓

- C/Rust: the core is a compiled extension, not pure Python.
- text: encodes to human-readable text (JSON), not opaque bytes.
- class from data: decode builds a class the payload names by import path,
  beyond the declared type and its subclasses.
- callables: a function round-trips by import path, with no special field type.
- shared refs: an object referenced twice decodes as one object; cycles too.
- best-effort: bad input yields the partial value and every bad part's location.

In our case being pure python is a bad thing; native implementations are
faster and our preference would be to use msgspec. However we need pickle
semantics but as raw text.

Each mark is what a probe of msgspec 0.22, pydantic 2.13, cattrs 26.2,
mashumaro 3.23, and jsonpickle 4.1 returned:

library     text    class from data  callables      shared refs  best-effort
----------  ------  ---------------  -------------  -----------  -----------------
msgspec     JSON    ``dict``         raises         split        first error
pydantic    JSON    ``dict``         raises         split        all, no value
cattrs      JSON    raises           raises         split        all, no value
mashumaro   JSON    raises           raises         split        first error
pickle      binary  the class        same function  kept         first error
jsonpickle  JSON    the class        same function  kept         no error, ``dict``
treekle     JSON    the class        same function  kept         partial value, all

- text: what one dataclass encodes to.
- class from data: an ``object`` field given a payload naming another class.
- callables: a field holding a module-level function.
- shared refs: one object in two fields; "split" decodes two separate copies.
- best-effort: two bad fields.

Fast libraries decode only classes known ahead of time: a class named in the
data means an import and code run on read. Treekle gates that behind
``allow_imports``. One cycle shape fails to read: one through a dataclass
decoded as its declared type.
"""

from __future__ import annotations

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


__all__ = [
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
    "mutable",
    "parse",
    "to_plain",
]
