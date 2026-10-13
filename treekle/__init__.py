"""Treekle: pickle semantics as plain python trees.

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

                 treekle  pickle  jsonpickle  pydantic  msgspec  cattrs  mashumaro
---------------  -------  ------  ----------  --------  -------  ------  ---------
C/Rust           ✗        ✓       ✗           ✓         ✓        ✗       ✗
stdlib           ✓        ✓       ✓           ✗         ✗        ✗       ✗
plain            ✓        ✗       ✓           ✓         ✓        ✓       ✓
typed result     ✓        ✗       ✗           ✓         ✓        ✓       ✓
typed contents   ✓        ✗       ✗           ✓         ✓        ✓       ✓
untagged input   ✓        ✗       ✗           ✓         ✓        ✓       ✓
class from data  ✓        ✗       ✗           ✗         ✗        ✗       ✗
callables        ✓        ✓       ✓           ✗         ✗        ✗       ✗
shared refs      ✓        ✓       ✓           ✗         ✗        ✗       ✗
best-effort      ✓        ✗       ✗           ✗         ✗        ✗       ✗
GitHub stars     --       stdlib  1.3k        29.0k     4.2k     1.1k    940

✓ = yes, ✗ = no. Star counts checked 2026-10-11.

- C/Rust: the core is a compiled extension, not pure Python.
- stdlib: part of, or pure Python importing only, the standard library.
- plain: encodes to plain dicts, lists, and scalars, so any writer stores it:
  JSON, YAML, TOML.
- typed result: a type checker infers ``T`` from ``decode(data, T)``, not
  ``Any``.
- typed contents: every nested field and element is checked against its
  annotation, e.g. ``list[Layer]``. An ``isinstance`` check on jsonpickle's
  result would fix typed result, not this.
- untagged input: data with no ``py/*`` tags reads as the declared type, so
  hand-written config works.
- class from data: decode builds the class the payload names by import path
  (a subclass, a Protocol implementation, or any class) and rejects one that
  does not fit the field's annotation.
- callables: a function round-trips by import path, with no special field type.
- shared refs: an object referenced twice decodes as one object; cycles too.
- best-effort: bad input yields the partial value and every bad part's location.

In our case being pure python is a bad thing; native implementations are
faster and our preference would be to use msgspec. However we need pickle
semantics but as raw text.

Each cell is what a probe of jsonpickle 4.1, pydantic 2.13, msgspec 0.22,
cattrs 26.2, and mashumaro 3.23 returned:

library     object       callables      shared refs  best-effort
----------  -----------  -------------  -----------  --------------------------
treekle     named class  same function  one object   partial value, all errors
pickle      named class  same function  one object   first error
jsonpickle  named class  same function  one object   no error, returns ``dict``
pydantic    ``dict``     raises         two copies   all errors, no value
msgspec     ``dict``     raises         two copies   first error
cattrs      raises       raises         two copies   all errors, no value
mashumaro   raises       raises         two copies   first error

- object: class from data, an ``object`` field given a payload naming a class.
- callables: a field holding a module-level function.
- shared refs: one object in two fields.
- best-effort: two bad fields.

Fast libraries decode only classes known ahead of time: a class named in the
data means an import and code run on read. Treekle gates that behind
``allow_imports``. One cycle shape fails to read: one through a dataclass
decoded as its declared type.
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
