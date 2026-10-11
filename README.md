# treekle🌳

[![PyPI version](https://img.shields.io/pypi/v/treekle.svg)](https://pypi.org/project/treekle/)
[![CI](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

Pickle semantics as plain text: any Python object graph to a tree of dicts,
lists, and scalars, and back.

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
from dataclasses import dataclass, field

import json
import math

from treekle import ReadError, from_plain, to_plain


def mse(a: float, b: float) -> float:
    return (a - b) ** 2


@dataclass
class Layer:
    width: int


@dataclass
class Model:
    encoder: Layer
    decoder: Layer
    shape: tuple[int, int] = (2, 3)
    clip: float = math.inf
    loss: object = mse
    tags: set[str] = field(default_factory=lambda: {"base"})


layer = Layer(width=64)
text = json.dumps(to_plain(Model(encoder=layer, decoder=layer)))
# {"py/object": "__main__.Model",
#  "clip": {"py/float": "inf"},
#  "decoder": {"py/object": "__main__.Layer", "width": 64},
#  "encoder": {"py/ref": ["decoder"]},
#  "loss": {"py/function": "__main__.mse"},
#  "shape": {"py/tuple": [2, 3]},
#  "tags": {"py/set": ["base"]}}

model = from_plain(json.loads(text), Model, allow_imports=True)
assert model.encoder is model.decoder  # one object, not two copies
assert model.shape == (2, 3)  # a tuple, not a list
assert model.clip == math.inf  # JSON has no inf
assert model.loss is mse  # the function itself
```

A lambda has no import path, so it is refused when written, not when read:

```python
to_plain(lambda x: x)
# TypeError: Cannot serialize <function <lambda> ...>: import path
# '__main__.<lambda>' does not resolve to the same object.
```

Bad input reads as far as it can:

```python
try:
    from_plain({"width": "wide"}, Layer)
except ReadError as error:
    error.partial  # Layer(width=Invalid(raw='wide', reason="cannot read 'wide' as int"))
    error.bad  # {('width',): Invalid(raw='wide', reason="cannot read 'wide' as int")}
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
class the payload names. Importing a class by name is gated behind
`allow_imports=True`.

### Best-effort decoding

A bad part does not stop the read. `ReadError.partial` is the value with an
`Invalid` in place of each bad part; `ReadError.bad` maps each bad part's
path to its `Invalid`.

### Wire-invariant

The core never parses or prints text. `treekle.wire.json_loads` is the one
reader, typed so `from_plain(json_loads(text), T)` needs no cast.

## Why not another serializer

**tl;dr:** No library is perfect, including this one. Fast libraries decode
only classes known ahead of time; treekle trades speed for pickle semantics in
plain text.

| | C/Rust | text | class from data | callables | shared refs | best-effort |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| msgspec | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| pydantic | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| cattrs | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| mashumaro | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| pickle | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| jsonpickle | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| treekle | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |

- **C/Rust:** the core is a compiled extension, not pure Python.
- **text:** encodes to human-readable text (JSON), not opaque bytes.
- **class from data:** decode builds a class the payload names by import path,
  beyond the declared type and its subclasses.
- **callables:** a function round-trips by import path, with no special field
  type.
- **shared refs:** an object referenced twice decodes as one object; cycles
  too.
- **best-effort:** bad input yields the partial value and every bad part's
  location.

**Details:** each mark is what a probe of msgspec 0.22, pydantic 2.13, cattrs
26.2, mashumaro 3.23, and jsonpickle 4.1 returned:

| | text | class from data | callables | shared refs | best-effort |
|---|---|---|---|---|---|
| msgspec | JSON | `dict` | raises | split | first error |
| pydantic | JSON | `dict` | raises | split | all, no value |
| cattrs | JSON | raises | raises | split | all, no value |
| mashumaro | JSON | raises | raises | split | first error |
| pickle | binary | the class | same function | kept | first error |
| jsonpickle | JSON | the class | same function | kept | no error, `dict` |
| treekle | JSON | the class | same function | kept | partial value, all |

- **text:** what one dataclass encodes to.
- **class from data:** an `object` field given a payload naming another class.
- **callables:** a field holding a module-level function.
- **shared refs:** one object in two fields; "split" decodes two separate
  copies.
- **best-effort:** two bad fields.

One cycle shape fails to read: one through a dataclass decoded as its declared
type.
