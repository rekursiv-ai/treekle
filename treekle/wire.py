"""Wire-format readers and writers: text straight to or from a typed value.

The rest of treekle is wire-invariant; each reader here parses one format and
hands the tree to ``from_plain``, and each writer hands ``to_plain``'s tree to
one format -- or, with the Rust core, skips the tree entirely.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, overload

import importlib
import json
import warnings

from treekle.absent import ABSENT
from treekle.deserialize import _python_from_plain, from_plain
from treekle.serialize import _python_to_plain, to_plain


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle.types import Dialect, Hooks, MutablePlainTree


__all__ = [
    "json_dumps",
    "json_loads",
]


@overload
def json_loads(text: str | bytes) -> MutablePlainTree: ...


@overload
def json_loads[T](
    text: str | bytes,
    target: type[T],
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def json_loads(
    text: str | bytes,
    target: object,
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...


def json_loads(
    text: str | bytes,
    target: object = ABSENT,
    *,
    hooks: Hooks = MappingProxyType({}),
    allow_imports: bool = False,
    strict: bool = True,
) -> object:
    """Parse JSON text, as ``target`` when given; ``NaN``/``Infinity`` read too.

    ``json.loads`` as it is at the call parses the text. While it is still
    the one ``json`` had when treekle was imported, the Rust core parses the
    text itself, to what ``json.loads`` returns or raises, and calls it only
    for text it leaves to it.

    Args:
      text: JSON document.
      target: The type to read the document as; omit for the plain tree.
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
      strict: When false, also read numeric and boolean text.

    Returns:
      value: The document as a ``target``, or its plain tree.

    Raises:
      json.JSONDecodeError: ``text`` is not valid JSON.
      ReadError: The document does not fit ``target``.

    """
    # Perf: rows, best of 3: reading typed records straight from the text is 2.0-2.4x
    # over parsing then from_plain.
    typed_loads = _CORE_JSON.get("typed_loads")
    if target is not ABSENT and typed_loads is not None:
        return typed_loads(
            text,
            target,
            hooks=hooks,
            allow_imports=allow_imports,
            strict=strict,
        )
    loads = _CORE_JSON.get("loads", json.loads)
    tree = cast("MutablePlainTree", loads(text))
    if target is ABSENT:
        return tree
    return from_plain(
        tree,
        target,
        hooks=hooks,
        allow_imports=allow_imports,
        strict=strict,
    )


def json_dumps(
    obj: object,
    *,
    dialect: Dialect = "json",
    hooks: Hooks = MappingProxyType({}),
    sort: bool = True,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = None,
    indent: int | str | None = None,
    sort_keys: bool = False,
) -> str:
    """``json.dumps`` of ``to_plain(obj)``: the object straight to JSON text.

    ``json.dumps`` as it is at the call writes the text, called at most once.
    While it is still the one ``json`` had when treekle was imported, the
    Rust core writes the text itself, byte for byte what ``json.dumps``
    gives, and leaves to ``json`` only the parts it cannot write so.

    Args:
      obj: The value to write; anything ``to_plain`` takes.
      dialect: Forwarded to ``to_plain``.
      hooks: The codecs ``to_plain`` uses.
      sort: Order each object's attributes by name, as ``to_plain`` does.
      ensure_ascii: Escape non-ASCII text, as ``json.dumps`` does.
      separators: ``(item, key)`` separators, as ``json.dumps`` takes them.
      indent: Pretty-print by this many spaces or this text.
      sort_keys: Order dict keys, as ``json.dumps`` does.

    Returns:
      text: The JSON document, byte for byte ``json.dumps(to_plain(obj))``.

    """
    # Perf: exp000, min per call: the core's fused writer is 2.5x over
    # json.dumps(to_plain(obj)).
    dumps = _CORE_JSON.get("dumps")
    if dumps is not None:
        return cast(
            "str",
            dumps(
                obj,
                dialect=dialect,
                hooks=hooks,
                sort=sort,
                ensure_ascii=ensure_ascii,
                separators=separators,
                indent=indent,
                sort_keys=sort_keys,
            ),
        )
    return json.dumps(
        to_plain(obj, dialect=dialect, hooks=hooks, sort=sort),
        ensure_ascii=ensure_ascii,
        separators=separators,
        indent=indent,
        sort_keys=sort_keys,
    )


# The core's JSON functions serve only where its walkers do: the writer needs
# the encoder set up and the typed reader the decoder, so under
# ``TREEKLE_PURE_PYTHON``, or with a core that failed to set up, Python serves.
# Either way ``json.loads`` and ``json.dumps`` are looked up at each call, so
# one replaced after import is the one used.
def _core_json() -> dict[str, Callable[..., object]]:
    """Return the core's JSON functions that serve, by name; empty where ``json`` does."""
    writes = to_plain is not _python_to_plain
    reads = from_plain is not _python_from_plain
    if not (writes or reads):
        return {}
    package = cast("str", __package__)
    try:
        core = importlib.import_module(f"{package}._rust")
        found = {
            "loads": cast("Callable[..., object]", core.json_loads_plain),
            "dumps": cast("Callable[..., object]", core.json_dumps),
            "typed_loads": cast("Callable[..., object]", core.json_loads_typed),
        }
    except Exception as error:  # noqa: BLE001 -- Any failure of a stale core falls back to json.
        warnings.warn(
            f"{package}'s Rust core does not match this {package} ({error}); "
            "using the json module until the core is rebuilt",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}
    if not writes:
        del found["dumps"]
    if not reads:
        del found["typed_loads"]
    return found


_CORE_JSON: Final = _core_json()
"""The core's ``loads``, ``dumps`` and ``typed_loads`` that serve, by name."""
