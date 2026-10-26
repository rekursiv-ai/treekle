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
- ``wire``: ``json_dumps`` and ``json_loads``, the wire-format entry points.
- ``serialize``: ``to_plain``.
- ``deserialize``: ``from_plain``.
- ``parse``: ``parse``, for what a person typed.

What a dialect cannot carry natively is written as a ``py/*`` tag:

- ``py/object``, ``py/type``, ``py/function``: a class instance, a class, or a
  function, by dotted import path.
- ``py/tuple``, ``py/set``, ``py/frozenset``, ``py/mappingproxy``: containers a
  dialect flattens to lists or dicts.
- ``py/dict``, ``py/list``: a dict or list where ``default_concrete_types``
  makes an untagged mapping or sequence read as another type.
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

The contract, which ``serialize``, ``deserialize`` and the Rust core follow::

    # 1. A value is written untagged iff its declared type implies it:
    implied = (
        type(value) is reads_as(declared, untagged(value))
        and fits(value, declared)
    )
    #    reads_as: the class from_plain builds from untagged data under declared.
    #    fits: from_plain's own check of class, arity and keys; an int fits
    #    float.
    # 2. What declares a part:
    #    declared(root) == target                             (object: nothing)
    #    declared(record.field) == get_type_hints(type(record))[field]
    #    declared(part) == the declared container's part type, else object
    # 3. A tag decides its node's class and shape: a py/object tag names the
    #    record whose hints declare its fields, and below the root a tagged
    #    value is not refused for not fitting its declared type.
    # 4. An untagged node reads back as its tag would have: a record restores
    #    its state, and __init__ does not run.
    # 5. Hence, with every node's type kept (an int stays an int):
    assert from_plain(to_plain(v, target=T, default_concrete_types=C), T,
                      default_concrete_types=C) == v
    #    for every v of type T whose plain leaves (str, int, bool, None) are
    #    of their declared types: a plain leaf carries no tag.

``reads_as`` under an abstract declared origin is its default, the baseline
unless ``default_concrete_types`` overrides it; any other class is its own::

    | declared origin | baseline  | default_concrete_types may choose |
    |-----------------|-----------|-----------------------------------|
    | Mapping         | dict      | dict, MappingProxyType            |
    | MutableMapping  | dict      | dict                              |
    | Sequence        | list      | list, tuple                       |
    | MutableSequence | list      | list                              |
    | Set             | frozenset | frozenset, set                    |
    | MutableSet      | set       | set                               |
    | object, Any     | dict for a mapping, list for a sequence; fixed    |

``default_concrete_types`` merges as ``{**baseline, **default_concrete_types}``.
A key is one of the six abstract origins (a ``typing`` alias names its
``collections.abc`` class), and a value one of its row's choices, else
``ValueError``. Lookup is by exact origin: ``{Mapping: MappingProxyType}``
leaves ``MutableMapping`` reading as ``dict``. A union applies rule 1 to the
member its untagged data selects. A dict or list where the default is another
class is not implied, so it is written as ``py/dict`` or ``py/list``. Writer
and reader take the same ``default_concrete_types``.

Every rule at once:

>>> from collections.abc import Mapping
>>> from dataclasses import dataclass
>>> from types import MappingProxyType
>>> import treekle
>>> @dataclass
... class Leaf:
...     n: float
...
...     def __post_init__(self):
...         self.n *= 10
>>> @dataclass
... class Sub(Leaf):
...     pass
>>> @dataclass
... class Foo:
...     bar: Mapping[str, float]
...     baz: Mapping[str, float]
...     fiz: MappingProxyType[str, float]
...     leaf: Leaf
...     obj: object
...     sub: Leaf
>>> foo = Foo(
...     {"a": 1},
...     MappingProxyType({"b": 2.0}),
...     MappingProxyType({"c": 3}),
...     Leaf(4),
...     (6,),
...     Sub(5),
... )
>>> treekle.to_plain(foo) == {
...     "py/object": "__main__.Foo",
...     "bar": {"a": 1},
...     "baz": {"py/mappingproxy": {"b": 2.0}},
...     "fiz": {"c": 3},
...     "leaf": {"n": 40},
...     "obj": {"py/tuple": [6]},
...     "sub": {"py/object": "__main__.Sub", "n": 50},
... }
True
>>> frozen = {Mapping: MappingProxyType}
>>> tree = treekle.to_plain(foo, target=Foo, default_concrete_types=frozen)
>>> tree == {
...     "bar": {"py/dict": {"a": 1}},
...     "baz": {"b": 2.0},
...     "fiz": {"c": 3},
...     "leaf": {"n": 40},
...     "obj": {"py/tuple": [6]},
...     "sub": {"py/object": "__main__.Sub", "n": 50},
... }
True
>>> back = treekle.from_plain(
...     tree,
...     Foo,
...     default_concrete_types=frozen,
...     allow_imports=True,
... )
>>> back == foo
True
>>> type(back.bar).__name__, type(back.baz).__name__, type(back.bar["a"]).__name__
('dict', 'mappingproxy', 'int')
>>> back.leaf.n, type(back.sub).__name__
(40, 'Sub')
"""

from __future__ import annotations

from treekle.absent import ABSENT, Absent
from treekle.deserialize import from_plain
from treekle.parse import parse
from treekle.plain import immutable, mutable
from treekle.serialize import to_plain
from treekle.types import (
    Classes,
    Dialect,
    FieldPath,
    Hooks,
    Invalid,
    MutablePlainTree,
    Plain,
    PlainTree,
    ReadError,
)
from treekle.wire import json_dumps, json_loads


__all__ = [
    "ABSENT",
    "Absent",
    "Classes",
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
    "json_dumps",
    "json_loads",
    "mutable",
    "parse",
    "to_plain",
]
