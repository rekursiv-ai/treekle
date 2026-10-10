# treekle

[![PyPI version](https://img.shields.io/pypi/v/treekle.svg)](https://pypi.org/project/treekle/)
[![CI](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/treekle/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

Pickle semantics as plain text. `to_plain` turns any Python object graph --
dataclasses, functions, shared references, cycles -- into a plain tree of
dicts, lists, and scalars. `from_plain` turns it back.

## Quick Start

```bash
uv add treekle
# Alternatively: python -m pip install treekle
```

```python
import json
from dataclasses import dataclass

from treekle import from_plain, to_plain


@dataclass
class Leaf:
    k: int


@dataclass
class Pair:
    a: Leaf
    b: Leaf


leaf = Leaf(k=1)
text = json.dumps(to_plain(Pair(a=leaf, b=leaf)))
# {"py/object": "__main__.Pair", "a": {"py/object": "__main__.Leaf", "k": 1},
#  "b": {"py/ref": ["a"]}}

pair = from_plain(json.loads(text), object, allow_imports=True)
assert pair.a is pair.b
```

## What it keeps

- **Classes named by the data.** A field typed `object` or a Protocol reads
  back as whatever class the payload names. Reading code by name is gated
  behind `allow_imports=True`.
- **Functions and classes**, by import path.
- **Shared objects and cycles.** A repeat is written as a `py/ref` to the path
  of its first occurrence.
- **Best-effort decoding.** Bad input raises `ReadError` carrying the partial
  value, with an `Invalid` at each bad part, and every bad part's location.

## Why not another serializer

See the comparison in the `treekle` package docstring.
