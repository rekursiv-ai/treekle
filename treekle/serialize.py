"""Convert Python values to plain trees.

``to_plain`` takes a ``dialect`` naming the target format, which decides what
must be tagged; ``"json"`` tags non-finite floats, which ``"python"`` keeps
native. Both tag everything else alike.
``mutable`` picks dicts and lists, the default, or ``MappingProxyType`` and
tuples; either way the result is freshly built and shared with no one.

Each tagged type has an encoder here and a decoder in ``deserialize``, paired by
name. Leaves pass through unencoded.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path, PurePath
from types import (
    BuiltinFunctionType,
    FunctionType,
    MappingProxyType,
    ModuleType,
)
from typing import (
    Final,
    Literal,
    cast,
    overload,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import base64
import json
import math
import weakref

from treekle.plain import mutable
from treekle.reflect import (
    PENDING,
    SKIPPED_ATTRIBUTES,
    resolve_import,
    slot_names,
)
from treekle.types import (
    Dialect,
    FieldPath,
    Hooks,
    MutablePlainTree,
    PlainTree,
)


__all__ = [
    "to_plain",
]


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: Literal[True] = ...,
    hooks: Hooks = ...,
) -> MutablePlainTree: ...


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
) -> PlainTree: ...


def to_plain(
    obj: object,
    *,
    dialect: Dialect = "json",
    mutable: bool = True,
    hooks: Hooks = MappingProxyType({}),
) -> PlainTree:
    """Convert a Python value to plain data, tagging what ``dialect`` lacks.

    Args:
      obj: The value to convert; any object graph, shared references and cycles
        included.
      dialect: The target format, which decides which values need a tag.
      mutable: Build dicts and lists when true; ``MappingProxyType`` and tuples
        when false.
      hooks: Codecs for leaf types with no built-in conversion, such as tensors.

    Returns:
      tree: Fresh plain data that ``from_plain`` turns back into ``obj``.

    Raises:
      TypeError: A leaf has no conversion and no hook.

    """
    state = _Encoding(
        dialect=dialect,
        hooks=hooks,
        make_list=_thawed_list if mutable else _frozen_list,
        make_dict=_thawed_dict if mutable else _frozen_dict,
    )
    return _encode(obj, state)


@dataclass(slots=True, kw_only=True)
class _Encoding:
    """State of one ``to_plain`` walk.

    Attributes:
      dialect: The target format.
      hooks: Codecs for leaf types with no built-in conversion.
      make_list: Builds a sequence node: ``list`` or ``tuple``.
      make_dict: Builds a mapping node: ``dict`` or ``MappingProxyType``.
      at: Where the node being encoded sits.
      seen: The path of each registered object's first occurrence, by ``id``.
      alive: Every registered object, so its ``id`` is not reused mid-walk.

    """

    dialect: Dialect
    hooks: Hooks
    make_list: Callable[[list[PlainTree]], PlainTree]
    make_dict: Callable[[dict[str, PlainTree]], PlainTree]
    at: list[str | int] = field(default_factory=list[str | int])
    seen: dict[int, FieldPath] = field(default_factory=dict[int, FieldPath])
    alive: list[object] = field(default_factory=list[object])


type _Encoder = Callable[[object, _Encoding], PlainTree]
"""Encode one value into whichever kind of tree ``state`` builds."""


def _thawed_list(items: list[PlainTree]) -> PlainTree:
    """Return a sequence node as the list it was built in."""
    return items


def _frozen_list(items: list[PlainTree]) -> PlainTree:
    """Return a sequence node as a tuple."""
    return tuple(items)


def _thawed_dict(items: dict[str, PlainTree]) -> PlainTree:
    """Return a mapping node as the dict it was built in."""
    return items


def _frozen_dict(items: dict[str, PlainTree]) -> PlainTree:
    """Return a mapping node as a ``MappingProxyType``."""
    return MappingProxyType(items)


def _encode(value: object, state: _Encoding) -> PlainTree:
    """Encode one node: plain passthrough, ``py/ref`` repeat, else its encoder."""
    kind = type(value)
    if value is None or (isinstance(value, (str, int)) and kind in _NATIVE_LEAVES):
        return value
    if kind is float:
        return _encode_float(value, state)
    seen = state.seen.get(id(value))
    if seen is not None:
        return state.make_dict({"py/ref": state.make_list(list(seen))})
    if kind in state.hooks:
        return _encode_hooked(value, state)
    return _encoder_for(kind)(value, state)


def _encode_at(value: object, key: str | int, state: _Encoding) -> PlainTree:
    """Encode a container's part, which sits at ``key`` below the current node."""
    state.at.append(key)
    try:
        return _encode(value, state)
    finally:
        state.at.pop()


def _encoder_for(kind: type) -> _Encoder:
    """Return the encoder for ``kind``, choosing once per class and caching it."""
    encoder = _ENCODERS.get(kind) or _ENCODER_CACHE.get(kind)
    if encoder is not None:
        return encoder
    bases = kind.__mro__
    if type in bases:
        encoder = _encode_type
    elif Enum in bases:
        encoder = _encode_enum
    elif kind is Path or kind is _CONCRETE_PATH:
        encoder = _encode_path
    elif kind is UUID:
        encoder = _encode_uuid
    elif kind is datetime:
        encoder = _encode_datetime
    elif is_dataclass(kind):
        encoder = _encode_object
    else:
        encoder = _encode_reduce
    _ENCODER_CACHE[kind] = encoder
    return encoder


# Registering before encoding children is what makes cycles terminate: a
# back-reference reached below finds the parent already placed.
def _register(value: object, state: _Encoding) -> None:
    """Record where ``value`` first sits, before its children are encoded."""
    state.seen[id(value)] = tuple(state.at)
    state.alive.append(value)


def _rollback(state: _Encoding, mark: int) -> None:
    """Forget every number assigned since ``len(state.alive)`` was ``mark``."""
    for value in state.alive[mark:]:
        del state.seen[id(value)]
    del state.alive[mark:]


def _encode_float(value: object, state: _Encoding) -> PlainTree:
    """Encode a float, as ``py/float`` when non-finite and the dialect must."""
    number = cast("float", value)
    if math.isfinite(number) or state.dialect == "python":
        return number
    return state.make_dict({"py/float": repr(number)})


def _encode_complex(value: object, state: _Encoding) -> PlainTree:
    """Encode a complex as ``py/complex`` of its real and imaginary floats."""
    number = cast("complex", value)
    parts = [_encode_float(number.real, state), _encode_float(number.imag, state)]
    return state.make_dict({"py/complex": state.make_list(parts)})


def _encode_bytes(value: object, state: _Encoding) -> PlainTree:
    """Encode bytes as ``py/b64``."""
    text = base64.b64encode(cast("bytes", value)).decode()
    return state.make_dict({"py/b64": text})


def _encode_path(value: object, state: _Encoding) -> PlainTree:
    """Encode a path as ``py/path``."""
    return state.make_dict({"py/path": str(value)})


def _encode_uuid(value: object, state: _Encoding) -> PlainTree:
    """Encode a UUID as ``py/uuid``."""
    return state.make_dict({"py/uuid": str(value)})


def _encode_datetime(value: object, state: _Encoding) -> PlainTree:
    """Encode a datetime as ``py/datetime``, keeping a named zone."""
    moment = cast("datetime", value)
    text = moment.isoformat()
    if isinstance(moment.tzinfo, ZoneInfo):
        text = f"{text}[{moment.tzinfo.key}]"
    return state.make_dict({"py/datetime": text})


def _encode_enum(value: object, state: _Encoding) -> PlainTree:
    """Encode an enum member as ``py/reduce`` of its class and value."""
    return _encode_reduce(value, state)


def _encode_list(value: object, state: _Encoding) -> PlainTree:
    """Encode a list, registering it for ``py/ref``."""
    items = cast("Sequence[object]", value)
    _register(value, state)
    return state.make_list(
        [_encode_at(item, index, state) for index, item in enumerate(items)],
    )


def _encode_tuple(value: object, state: _Encoding) -> PlainTree:
    """Encode a tuple as ``py/tuple``; a tuple is a value, never referenced."""
    items = [
        _encode_at(item, index, state)
        for index, item in enumerate(cast("tuple[object, ...]", value))
    ]
    return state.make_dict({"py/tuple": state.make_list(items)})


def _encode_set(value: object, state: _Encoding) -> PlainTree:
    """Encode a set as ``py/set`` with members in a deterministic order."""
    members = cast("AbstractSet[object]", value)
    _register(value, state)
    ordered = sorted(members, key=lambda member: _set_order_key(member, state))
    items = [_encode_at(member, index, state) for index, member in enumerate(ordered)]
    return state.make_dict({"py/set": state.make_list(items)})


def _encode_frozenset(value: object, state: _Encoding) -> PlainTree:
    """Encode a frozenset as ``py/frozenset``; a frozenset is never referenced."""
    members = cast("AbstractSet[object]", value)
    ordered = sorted(members, key=lambda member: _set_order_key(member, state))
    items = [_encode_at(member, index, state) for index, member in enumerate(ordered)]
    return state.make_dict({"py/frozenset": state.make_list(items)})


def _set_order_key(value: object, state: _Encoding) -> tuple[str, str]:
    """Return a set member's sort key without registering anything it reaches."""
    mark = len(state.alive)
    try:
        encoded = _encode(value, state)
    finally:
        _rollback(state, mark)
    text = json.dumps(mutable(encoded), sort_keys=True, separators=(",", ":"))
    return repr(value), text


def _encode_dict(value: object, state: _Encoding) -> PlainTree:
    """Encode a dict, escaping keys through ``_encode_key``."""
    mapping = cast("Mapping[object, object]", value)
    _register(value, state)
    encoded: dict[str, PlainTree] = {}
    for key, member in mapping.items():
        name = _encode_key(key, state)
        encoded[name] = _encode_at(member, name, state)
    return state.make_dict(encoded)


def _encode_mappingproxy(value: object, state: _Encoding) -> PlainTree:
    """Encode a ``MappingProxyType`` as ``py/mappingproxy`` around its dict."""
    copied = dict(cast("Mapping[object, object]", value))
    return state.make_dict({"py/mappingproxy": _encode_dict(copied, state)})


def _encode_key(key: object, state: _Encoding) -> str:
    """Return a dict key, ``json://``-escaping non-strings and tag lookalikes."""
    if isinstance(key, str) and not key.startswith(("py/", "json://")):
        return key
    # A key is text, so nothing inside it has a path a later ``py/ref`` can name.
    mark = len(state.alive)
    try:
        encoded = _encode(key, state)
    finally:
        _rollback(state, mark)
    return "json://" + json.dumps(mutable(encoded))


def _encode_type(value: object, state: _Encoding) -> PlainTree:
    """Encode a class as ``py/type``."""
    return state.make_dict({"py/type": _import_path(cast("type", value))})


def _encode_function(value: object, state: _Encoding) -> PlainTree:
    """Encode a module-level function as ``py/function``."""
    if not (isinstance(getattr(value, "__self__", None), (ModuleType, type(None)))):
        return _encode_reduce(value, state)
    path = _import_path(cast("Callable[..., object]", value))
    return state.make_dict({"py/function": path})


def _encode_object(value: object, state: _Encoding) -> PlainTree:
    """Encode a dataclass or slotted object as ``py/object`` with its state."""
    _register(value, state)
    payload: dict[str, PlainTree] = {"py/object": _import_path(type(value))}
    for name in _attribute_names(value):
        member: object = getattr(value, name, PENDING)
        if member is not PENDING:
            payload[name] = _encode_at(member, name, state)
    return state.make_dict(payload)


def _encode_hooked(value: object, state: _Encoding) -> PlainTree:
    """Encode a value through its caller-supplied hook as ``py/hook``."""
    encode_hook = state.hooks[type(value)][0]
    _register(value, state)
    # The payload is arbitrary caller data, so it takes the same walk as any
    # other value: it may hold non-finite floats or tag-like keys.
    path = _import_path(type(value))
    payload = _encode_at(encode_hook(value), 1, state)
    return state.make_dict({"py/hook": state.make_list([path, payload])})


def _encode_reduce(value: object, state: _Encoding) -> PlainTree:
    """Encode a value by its pickle reduce recipe as ``py/reduce``."""
    recipe = _canonical_reduce(value)
    if recipe is None:
        return _encode_unreduced(value, state)
    # A bare-name reduce names a module global, written as an import path.
    if isinstance(recipe, str):
        path = _verified_path(f"{type(value).__module__}.{recipe}", value)
        return state.make_dict({"py/type": path})
    func, args, *rest = recipe
    mark = len(state.alive)
    try:
        if any(part is not None for part in rest):
            _register(value, state)
        elements = [
            _encode_at(part, index, state)
            for index, part in enumerate((func, args, *rest))
        ]
    except TypeError:
        _rollback(state, mark)
        return _encode_unreduced(value, state)
    while len(elements) > 2 and elements[-1] is None:
        elements.pop()
    return state.make_dict({"py/reduce": state.make_list(elements)})


def _encode_unreduced(value: object, state: _Encoding) -> PlainTree:
    """Encode a value with no usable reduce recipe by the protocols it has."""
    kind = type(value)
    name = kind.__name__
    if hasattr(kind, "__dataclass_fields__") or "__slots__" in kind.__dict__:
        return _encode_object(value, state)
    if issubclass(kind, Mapping):
        return _encode_dict(value, state)
    if issubclass(kind, Sequence) and not issubclass(kind, (str, bytes)):
        return _encode_list(value, state)
    if issubclass(kind, AbstractSet):
        return _encode_set(value, state)
    raise TypeError(
        f"Cannot serialize leaf of type {name!r}. "
        f"Pass hooks={{{name}: (encode, decode)}} to to_plain().",
    )


# A path's arguments are restated as one joined string. CPython 3.12 reduces
# ``PurePath`` to one argument per segment and 3.14 to a single string, so the
# raw recipe would differ per interpreter; every version rebuilds from the join.
def _canonical_reduce(
    value: object,
) -> tuple[Callable[..., object], tuple[object, ...], *tuple[object, ...]] | str | None:
    """Return ``value``'s reduce recipe with interpreter-independent arguments."""
    try:
        reduced = cast("object", value.__reduce_ex__(2))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    if isinstance(reduced, str):
        return reduced
    if not isinstance(reduced, tuple):
        return None
    parts = list(cast("tuple[object, ...]", reduced))
    if len(parts) < 2 or len(parts) > 5:
        return None
    func, args = parts[0], parts[1]
    if not callable(func) or not isinstance(args, tuple):
        return None
    arguments = (
        (str(value),) if isinstance(value, PurePath) else cast("tuple[object]", args)
    )
    rest = [
        list(cast("Iterable[object]", part))
        if index in (1, 2) and part is not None
        else part
        for index, part in enumerate(parts[2:])
    ]
    # A partial's state ends with its instance ``__dict__``, which CPython creates,
    # empty, the first time anything reads the attribute; empty rebuilds as none.
    if isinstance(value, partial) and rest and isinstance(rest[0], tuple):
        *fields, instance_dict = cast("tuple[object, ...]", rest[0])
        if instance_dict == {}:
            rest[0] = (*fields, None)
    return (func, arguments, *rest)


def _import_path(value: type | Callable[..., object]) -> str:
    """Return the verified dotted import path of a class or function, cached."""
    cached = _IMPORT_PATHS.get(id(value))
    if cached is not None and cached[0] is value:
        return cached[1]
    module: object = getattr(value, "__module__", None)
    qualname: object = getattr(value, "__qualname__", None)
    if (
        not isinstance(module, str)
        or not isinstance(qualname, str)
        or "<locals>" in qualname
    ):
        raise TypeError(
            f"Cannot serialize {value!r}: it has no importable path "
            "(module-level __qualname__). Local/lambda callables and local "
            "classes/subclasses cannot be deserialized.",
        )
    path = _verified_path(f"{module}.{qualname}", value)
    _IMPORT_PATHS[id(value)] = (value, path)
    return path


def _verified_path(path: str, value: object) -> str:
    """Return ``path`` after proving it imports as ``value`` itself."""
    message = (
        f"Cannot serialize {value!r}: import path {path!r} does not resolve to "
        "the same object."
    )
    try:
        resolved = resolve_import(path)
    except (AttributeError, ImportError) as error:
        raise TypeError(message) from error
    if resolved is not value:
        raise TypeError(message)
    return path


def _attribute_names(value: object) -> Iterator[str]:
    """Yield state attributes: slots, then sorted ``__dict__`` keys."""
    slots = slot_names(type(value))
    yield from slots
    if hasattr(value, "__dict__"):
        seen = set(slots)
        for key in sorted(vars(value)):
            if key not in seen and key not in SKIPPED_ATTRIBUTES:
                seen.add(key)
                yield key


_NATIVE_LEAVES: Final = frozenset((str, int, bool))


_CONCRETE_PATH: Final = type(Path())


_ENCODERS: Final[Mapping[type, _Encoder]] = MappingProxyType(
    {
        float: _encode_float,
        complex: _encode_complex,
        bytes: _encode_bytes,
        list: _encode_list,
        tuple: _encode_tuple,
        set: _encode_set,
        frozenset: _encode_frozenset,
        dict: _encode_dict,
        MappingProxyType: _encode_mappingproxy,
        FunctionType: _encode_function,
        BuiltinFunctionType: _encode_function,
    },
)
"""Encoders by exact type; ``_encoder_for`` resolves and caches every other class."""


_ENCODER_CACHE: Final[weakref.WeakKeyDictionary[type, _Encoder]] = (
    weakref.WeakKeyDictionary()
)


# Keyed by ``id`` rather than weakly: a builtin function such as ``max`` cannot
# be weakly referenced. Each entry keeps its object, so the ``id`` is never
# reused while the entry stands.
_IMPORT_PATHS: Final[dict[int, tuple[object, str]]] = {}
