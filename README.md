# treekle🌳

[![PyPI version](https://img.shields.io/pypi/v/treekle.svg)](https://pypi.org/project/treekle/)
[![CI](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

**tree** + pic**kle**: pickle's semantics, written as a plain tree of dicts,
lists, and scalars instead of opaque bytes.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv add treekle

# Alternatively: python -m pip install treekle
```

`to_plain` turns a value into plain data any JSON, TOML, or YAML writer can
store. `from_plain` turns it back into the type you ask for. Neither touches
text, so the wire format is yours.

## Example

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import json
import math

from treekle import ABSENT, ReadError, from_plain, to_plain


def by_name(user: User) -> str:
    return user.name


@dataclass
class User:
    name: str


@dataclass
class Job:
    owner: User
    reviewer: User
    fallback: object
    sort_key: Callable[[User], str]
    output: type
    root: Path
    timeout: float
    env: MappingProxyType[str, str]


ada = User("ada")
job = Job(
    owner=ada,
    reviewer=ada,  # the same object
    fallback=ABSENT,  # a singleton
    sort_key=by_name,  # a function
    output=dict,  # a class
    root=Path("/srv"),
    timeout=math.inf,  # JSON has no inf
    env=MappingProxyType({"LANG": "C"}),
)

back = from_plain(json.loads(json.dumps(to_plain(job))), Job, allow_imports=True)
assert back.owner is back.reviewer
assert back.fallback is ABSENT
assert back.sort_key is by_name
assert back.output is dict
assert back.root == Path("/srv")
assert back.timeout == math.inf
assert isinstance(back.env, MappingProxyType)
```

Every other library fails on the same `job`:

- jsonpickle

  ```
  AttributeError: 'Absent' object attribute '_instance' is read-only
  ```

- pickle

  ```
  TypeError: cannot pickle 'mappingproxy' object
  ```

- msgspec

  ```
  TypeError: Encoding objects of type Absent is unsupported
  ```

- pydantic

  ```
  PydanticSchemaGenerationError: Unable to generate pydantic-core schema for mappingproxy[str, str]. Set `arbitrary_types_allowed=True` in the model_config to ignore this error or implement `__get_pydantic_core_schema__` on your type to fully support it.
  ```

- cattrs

  ```
  TypeError: Object of type Absent is not JSON serializable
  ```

- mashumaro

  ```
  UnserializableField: Field "fallback" of type object in Job is not serializable
  ```

Additionally, `treekle` reads bad inputs all the way through:

```python
@dataclass
class Team:
    a: User
    b: User
    c: User


bad = '{"a": {"name": "ada"}, "b": {"name": 7}, "c": {"name": 8}}'
try:
    from_plain(json.loads(bad), Team)
except ReadError as error:
    error.partial
    # Team(a=User(name='ada'),
    #      b=User(name=Invalid(raw=7, reason='cannot read 7 as str')),
    #      c=User(name=Invalid(raw=8, reason='cannot read 8 as str')))
```

Every other library on the same `bad`:

- jsonpickle

  ```
  {'a': {'name': 'ada'}, 'b': {'name': 7}, 'c': {'name': 8}}
  ```

- msgspec

  ```
  ValidationError: Expected `str`, got `int` - at `$.b.name`
  ```

- pydantic

  ```
  ValidationError: 2 validation errors for Team
  b.name
    Input should be a valid string [type=string_type, input_value=7, input_type=int]
  c.name
    Input should be a valid string [type=string_type, input_value=8, input_type=int]
  ```

- cattrs

  ```
  Team(a=User(name='ada'), b=User(name='7'), c=User(name='8'))
  ```

- mashumaro

  ```
  Team(a=User(name='ada'), b=User(name='7'), c=User(name='8'))
  ```

Captured with jsonpickle 4.1.3, msgspec 0.22.0, pydantic 2.13.5, cattrs
26.2.1, and mashumaro 3.23.

A lambda has no import path, so it is refused when written, not when read:

```python
to_plain(lambda x: x)
# TypeError: Cannot serialize <function <lambda> ...>: import path
# '__main__.<lambda>' does not resolve to the same object.
```

## Features

### Tags for what JSON lacks

Anything a wire format cannot carry natively becomes a `py/*` tag, spelled as
jsonpickle spells it where one exists:

| Value | Tag |
|---|---|
| class instance, class, function | `py/object`, `py/type`, `py/function` |
| tuple, set, frozenset, mappingproxy | `py/tuple`, `py/set`, `py/frozenset`, `py/mappingproxy` |
| non-finite float, complex, bytes | `py/float`, `py/complex`, `py/b64` |
| `Path`, `UUID`, `datetime` | `py/path`, `py/uuid`, `py/datetime` |
| anything else picklable | `py/reduce` |
| a type you register with `hooks=` | `py/hook` |
| a repeat of an earlier object | `py/ref` |
| a non-string or tag-like dict key | `json://...` |

### Shared objects and cycles

A repeat is written as `py/ref`, the path to its first occurrence, so
`model.encoder is model.decoder` survives the round trip. Cycles terminate the
same way. Older `py/id` back-references still read.

### The type decides, unless a tag does

`from_plain(data, T)` reads `data` as `T`: a JSON array as a `tuple[int, int]`,
`"red"` as `Color.RED`, a dict as a dataclass. Where the data carries a tag, the
tag decides, so a field typed `object` or a Protocol reads back as whatever
class the payload names, if that class fits the annotation. Importing a class
by name is gated behind `allow_imports=True`.

### Best-effort decoding

A bad part does not stop the read. `ReadError.partial` is the value with an
`Invalid` in place of each bad part; `ReadError.bad` maps each bad part's
path to its `Invalid`.

### Wire-invariant

The core never parses or prints text. `treekle.json_loads` is the one
reader: `json_loads(text, T)` parses and reads as `T` in one call. This is its
whole body, minus overloads and docstring:

```python
import json


def json_loads(
    text: str | bytes,
    target: object = ABSENT,
    *,
    hooks: Hooks = MappingProxyType({}),
    allow_imports: bool = False,
    strict: bool = True,
) -> object:
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
```

## Why not another serializer

**tl;dr:** The pickle-style libraries have a nearly orthogonal feature set to
the JSON-style libraries. The group either is forgiving about types (like
callables) and forgoes type-strictness or cannot tolerate types to achieve
safety. `jsonpickle` is arguably the most similar to `treekle` (and was our
default choice until making treekle).


| | [treekle](https://github.com/rekursiv-ai/treekle) | [jsonpickle](https://github.com/jsonpickle/jsonpickle) | [pickle](https://docs.python.org/3/library/pickle.html) | [mashumaro](https://github.com/Fatal1ty/mashumaro) | [cattrs](https://github.com/python-attrs/cattrs) | [pydantic](https://github.com/pydantic/pydantic) | [msgspec](https://github.com/jcrist/msgspec) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C/Rust | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| plain | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| typed result | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| typed contents | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| untagged input | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| class from data | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| callables | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| shared refs | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| best-effort | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GitHub stars | -- | 1.3k | stdlib | 940 | 1.1k | 29.0k | 4.2k |

✅ = yes, ❌ = no. Star counts checked 2026-10-11. Corrections welcome --
[open a PR](https://github.com/rekursiv-ai/treekle/pulls).

- **C/Rust:** the core is a compiled extension, not pure Python.
- **plain:** encodes to plain dicts, lists, and scalars, so any writer stores
  it: JSON, YAML, TOML.
- **typed result:** a type checker infers `T` from `decode(data, T)`, not
  `Any`.
- **typed contents:** every nested field and element is checked against its
  annotation, e.g. `list[Layer]`. An `isinstance` check on jsonpickle's
  result would fix typed result, not this.
- **untagged input:** data with no `py/*` tags reads as the declared type, so
  hand-written config works.
- **class from data:** decode builds the class the payload names by import
  path (a subclass, a Protocol implementation, or any class) and rejects one
  that does not fit the field's annotation.
- **callables:** a function round-trips by import path, with no special field
  type.
- **shared refs:** an object referenced twice decodes as one object; cycles
  too.
- **best-effort:** bad input yields the partial value and every bad part's
  location.

**Details:** each cell is what a probe of msgspec 0.22, pydantic 2.13, cattrs
26.2, mashumaro 3.23, and jsonpickle 4.1 returned:

| | treekle | jsonpickle | pickle | mashumaro | cattrs | pydantic | msgspec |
|---|---|---|---|---|---|---|---|
| plain | `to_plain` | `Pickler().flatten` | bytes only | `encode` | `unstructure` | `dump_python` | `to_builtins` |
| typed result | `Layer` | `dict \| list \| ...` | `Any` | `Layer` | `Layer` | `Layer` | `Layer` |
| typed contents | raises at `layers[0].width` | `dict` holding `'x'` | no target type | raises at `layers` | raises at `layers[0].width` | raises at `layers[0].width` | raises at `layers[0].width` |
| untagged input | `Net` | `dict` | raises: bytes only | `Net` | `Net` | `Net` | `Net` |
| class from data: subclass | subclass | subclass | subclass | base class, subclass fields dropped | base class, subclass fields dropped | base class, subclass fields dropped | base class, subclass fields dropped |
| class from data: Protocol | implementing class | implementing class | implementing class | raises: no Protocol schema | raises: no Protocol schema | raises: no Protocol schema | raises: no Protocol schema |
| class from data: `object` | named class | named class | named class | raises | raises | `dict` | `dict` |
| class from data: misfit | raises | misfit accepted | misfit accepted | raises | raises | raises | raises |
| callables | same function | same function | same function | raises | raises | raises | raises |
| shared refs | one object | one object | one object | two copies | two copies | two copies | two copies |
| best-effort | partial value, all errors | no error, returns `dict` | first error | first error | all errors, no value | all errors, no value | first error |

- **plain:** the call that encodes to plain data.
- **typed result:** what basedpyright infers for decoding a dataclass `Layer`.
- **typed contents:** `Net` with `layers: list[Layer]`, given
  `layers[0].width = "x"`. jsonpickle with a top-level `isinstance(obj, Net)`
  check returned `Net(layers=[{'width': 'x'}])`.
- **untagged input:** a valid `Net` payload with no `py/*` tags.
- **class from data:** a payload naming a class, read into a base-class field,
  a Protocol field, and an `object` field. cattrs keeps the subclass with
  `include_subclasses`; msgspec and pydantic need a closed tagged union.
  "misfit": a payload naming a class that is neither a subclass of the
  base-class field nor an implementation of the Protocol field.
- **callables:** a field holding a module-level function.
- **shared refs:** one object in two fields.
- **best-effort:** two bad fields.

One cycle shape fails to read: one through a dataclass decoded as its declared
type.
