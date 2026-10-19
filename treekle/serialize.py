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
    Mapping,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from math import isfinite
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
    get_args,
    overload,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import base64
import json
import weakref

from treekle.native import native
from treekle.plain import immutable
from treekle.reflect import (
    PENDING,
    attribute_names,
    is_mutable,
    resolve_import,
)
from treekle.types import (
    Dialect,
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
    sort: bool = ...,
) -> MutablePlainTree: ...


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
    sort: bool = ...,
) -> PlainTree: ...


def to_plain(
    obj: object,
    *,
    dialect: Dialect = "json",
    mutable: bool = True,
    hooks: Hooks = MappingProxyType({}),
    sort: bool = True,
) -> PlainTree:
    """Convert a Python value to plain data, tagging what ``dialect`` lacks.

    Args:
      obj: The value to convert; any object graph, shared references and cycles
        included, except a cycle inside a dict key. An object repeated inside a
        dict key reads back as an equal copy, not the same object.
      dialect: The target format, which decides which values need a tag.
      mutable: Build dicts and lists when true; ``MappingProxyType`` and tuples
        when false.
      hooks: Codecs for leaf types with no built-in conversion, such as tensors.
      sort: Order each object's attributes by name; when false, keep slots in
        MRO order then ``__dict__`` keys in insertion order. A stored form that
        must stay byte-stable sets this false.

    Returns:
      tree: Fresh plain data that ``from_plain`` turns back into ``obj``.

    Raises:
      ValueError: ``dialect`` names no known format.
      TypeError: A leaf has no conversion and no hook, or a dict key holds a
        cycle.

    """
    if dialect not in get_args(cast("object", Dialect.__value__)):
        raise ValueError(f"unknown dialect {dialect!r}; expected 'json' or 'python'")
    tree = cast(
        "MutablePlainTree",
        _encode(obj, _Encoding(dialect=dialect, hooks=hooks, sort=sort)),
    )
    return tree if mutable else immutable(tree)


# A place links to its parent's rather than holding the whole path, so placing a
# node costs one tuple however deep it sits; a ``py/ref`` spells it out.
type _Here = tuple[_Here, str | int] | None
"""Where a node sits: its parent's place and its own key; ``None`` is the root."""


type _Node = list[object] | dict[str, object]
"""A container node: raw parts at first, each replaced in place by its plain form."""


type _Entry = tuple[object, _Node, str | int | _RawKey, _Here]
"""A part still to encode: the value, its node and key, and the node's place."""


@dataclass(slots=True, kw_only=True)
class _Encoding:
    """State of one ``to_plain`` walk.

    Attributes:
      dialect: The target format.
      hooks: Codecs for leaf types with no built-in conversion.
      sort: Order each object's attributes by name, or keep declaration order.
      here: Where the value being encoded sits.
      node: The node the value being encoded is written into.
      key: Where in ``node`` it is written.
      seen: Where each registered object first sits, by ``id``.
      alive: Every registered object, so its ``id`` is not reused mid-walk.
      stack: Every part still to encode, the next one last.

    """

    dialect: Dialect
    hooks: Hooks
    sort: bool = True
    here: _Here = None
    node: _Node = field(default_factory=list[object])
    key: str | int = 0
    seen: dict[int, _Here] = field(default_factory=dict[int, "_Here"])
    alive: list[object] = field(default_factory=list[object])
    stack: list[_Entry] = field(default_factory=list["_Entry"])


@dataclass(frozen=True, slots=True, kw_only=True)
class _RawKey:
    """A dict key that is not written as itself, escaped when its entry is reached."""

    key: object


@dataclass(frozen=True, slots=True, kw_only=True)
class _Checkpoint:
    """Where a reduce restarts by protocol when one of its parts fails.

    Attributes:
      value: The value being reduced.
      mark: ``len(alive)`` before the reduce registered anything.
      node: The node the reduce is written into.
      key: Where in ``node`` it is written.
      here: Where the reduced value sits.

    """

    value: object
    mark: int
    node: _Node
    key: str | int
    here: _Here


type _Encoder = Callable[[object, _Encoding], object]
"""Encode one value; a container pushes the parts it has yet to encode."""


# Parts wait on ``state.stack`` rather than the call stack, so a tree of any
# depth encodes; recursion stopped near a thousand levels. Each container is
# built at once with its leaves in place, and each other part is written into
# it when popped, in the order a depth-first walk reaches it.
def _encode(value: object, state: _Encoding) -> object:
    """Encode ``value`` and everything it holds."""
    outer = state.stack, state.here, state.node, state.key
    root: list[object] = [None]
    stack: list[_Entry] = []
    state.stack, state.node, state.key = stack, root, 0
    try:
        root[0] = _encode_node(value, state)
        while stack:
            item, node, key, parent = stack.pop()
            if isinstance(item, _Checkpoint):
                continue
            try:
                if isinstance(key, _RawKey):
                    key = _encode_key(key.key, state)
                state.here, state.node, state.key = (parent, key), node, key
                done = _encode_node(item, state)
            except TypeError as error:
                node, key, done = _recover(error, state)
            if isinstance(node, list):
                assert isinstance(key, int)
                node[key] = done
            else:
                assert isinstance(key, str)
                node[key] = done
        return root[0]
    finally:
        state.stack, state.here, state.node, state.key = outer


def _encode_node(value: object, state: _Encoding) -> object:
    """Encode a leaf or a ``py/ref`` repeat, else run the value's encoder."""
    kind = type(value)
    if value is None or kind in _NATIVE_LEAVES:
        return value
    if kind is float:
        return _encode_float(value, state)
    if id(value) in state.seen:
        place = state.seen[id(value)]
        reference = _reference(place)
        if reference is not None:
            return reference
        # Inside a dict key, write a repeated value in full again. A key is a
        # value, and a ``py/ref`` to a place inside a key cannot be read back.
        if _encloses(place, state.here):
            raise TypeError(
                f"Cannot serialize a dict key holding a cycle through {value!r}.",
            )
    if kind in state.hooks:
        return _encode_hooked(value, state)
    return _encoder_for(kind)(value, state)


def _recover(
    error: TypeError,
    state: _Encoding,
) -> tuple[_Node, str | int, object]:
    """Drop parts down to the innermost reduce and encode it by protocol instead."""
    stack = state.stack
    while stack:
        checkpoint = stack.pop()[0]
        if not isinstance(checkpoint, _Checkpoint):
            continue
        _rollback(state, checkpoint.mark)
        state.here = checkpoint.here
        state.node, state.key = checkpoint.node, checkpoint.key
        try:
            done = _encode_unreduced(checkpoint.value, state)
        except TypeError as again:
            error = again
            continue
        return checkpoint.node, checkpoint.key, done
    raise error


def _push_items(node: list[object], state: _Encoding) -> None:
    """Queue each part of ``node`` that is not yet plain, the first last."""
    stack, here, python = state.stack, state.here, state.dialect == "python"
    for index in range(len(node) - 1, -1, -1):
        item = node[index]
        if item is None or type(item) in _NATIVE_LEAVES:
            continue
        if type(item) is float and (python or isfinite(item)):
            continue
        stack.append((item, node, index, here))


def _push_values(node: dict[str, object], state: _Encoding) -> None:
    """Queue each value of ``node`` that is not yet plain, the first last."""
    stack, here, python = state.stack, state.here, state.dialect == "python"
    for key in reversed(node):
        item = node[key]
        if item is None or type(item) in _NATIVE_LEAVES:
            continue
        if type(item) is float and (python or isfinite(item)):
            continue
        stack.append((item, node, key, here))


def _reference(place: _Here) -> dict[str, object] | None:
    """Return a ``py/ref`` naming ``place``, or ``None`` if it is inside a key."""
    keys: list[str | int] = []
    while place is not None:
        if place is _IN_KEY:
            return None
        place, key = place
        keys.append(key)
    keys.reverse()
    return {"py/ref": keys}


def _encloses(place: _Here, here: _Here) -> bool:
    """Return whether ``place`` is ``here`` or one of its parents in the same key."""
    while here is not None:
        if here is place:
            return True
        if here is _IN_KEY:
            return False
        here = here[0]
    return False


def _encoder_for(kind: type) -> _Encoder:
    """Return the encoder for ``kind``, choosing once per class and caching it."""
    encoder = _ENCODERS.get(kind) or _ENCODER_CACHE.get(kind)
    if encoder is not None:
        return encoder
    bases = kind.__mro__
    if type in bases:
        encoder = _encode_type
    elif Enum in bases:
        encoder = _encode_reduce
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
    state.seen[id(value)] = state.here
    state.alive.append(value)


def _rollback(state: _Encoding, mark: int) -> None:
    """Forget every number assigned since ``len(state.alive)`` was ``mark``."""
    # A value repeated inside a dict key is registered once per copy.
    for value in state.alive[mark:]:
        state.seen.pop(id(value), None)
    del state.alive[mark:]


def _encode_float(value: object, state: _Encoding) -> object:
    """Encode a float, as ``py/float`` when non-finite and the dialect must."""
    number = cast("float", value)
    if isfinite(number) or state.dialect == "python":
        return number
    return {"py/float": repr(number)}


def _encode_complex(value: object, state: _Encoding) -> object:
    """Encode a complex as ``py/complex`` of its real and imaginary floats."""
    number = cast("complex", value)
    parts = [_encode_float(number.real, state), _encode_float(number.imag, state)]
    return {"py/complex": parts}


def _encode_bytes(value: object, state: _Encoding) -> object:
    """Encode bytes as ``py/b64``."""
    del state
    return {"py/b64": base64.b64encode(cast("bytes", value)).decode()}


def _encode_path(value: object, state: _Encoding) -> object:
    """Encode a path as ``py/path``."""
    del state
    return {"py/path": str(value)}


def _encode_uuid(value: object, state: _Encoding) -> object:
    """Encode a UUID as ``py/uuid``."""
    del state
    return {"py/uuid": str(value)}


def _encode_datetime(value: object, state: _Encoding) -> object:
    """Encode a datetime as ``py/datetime``, keeping a named zone."""
    del state
    moment = cast("datetime", value)
    text = moment.isoformat()
    if isinstance(moment.tzinfo, ZoneInfo):
        text = f"{text}[{moment.tzinfo.key}]"
    return {"py/datetime": text}


def _encode_list(value: object, state: _Encoding) -> object:
    """Encode a list, registering it for ``py/ref``."""
    _register(value, state)
    node = list(cast("Iterable[object]", value))
    _push_items(node, state)
    return node


def _encode_tuple(value: object, state: _Encoding) -> object:
    """Encode a tuple as ``py/tuple``; a tuple is a value, never referenced."""
    node = list(cast("tuple[object, ...]", value))
    _push_items(node, state)
    return {"py/tuple": node}


def _encode_set(value: object, state: _Encoding) -> object:
    """Encode a set as ``py/set`` with members in a deterministic order."""
    _register(value, state)
    return {"py/set": _ordered(cast("AbstractSet[object]", value), state)}


def _encode_frozenset(value: object, state: _Encoding) -> object:
    """Encode a frozenset as ``py/frozenset``; a frozenset is never referenced."""
    return {"py/frozenset": _ordered(cast("AbstractSet[object]", value), state)}


def _ordered(members: AbstractSet[object], state: _Encoding) -> object:
    """Return a set's members as a node in a deterministic order, queued."""
    node = sorted(members, key=lambda member: _set_order_key(member, state))
    _push_items(node, state)
    return node


def _set_order_key(value: object, state: _Encoding) -> tuple[str, str]:
    """Return a set member's sort key without registering anything it reaches."""
    mark = len(state.alive)
    try:
        encoded = _encode(value, state)
    finally:
        _rollback(state, mark)
    return repr(value), json.dumps(encoded, sort_keys=True)


def _encode_dict(value: object, state: _Encoding) -> object:
    """Encode a dict, escaping keys through ``_encode_key``."""
    mapping = cast("Mapping[object, object]", value)
    _register(value, state)
    if all(map(_is_plain_key, mapping)):
        node = {cast("str", key): item for key, item in mapping.items()}
        _push_values(node, state)
        return node
    # An escaped key is encoded when its entry is reached, after every earlier
    # value, so a key naming an object written earlier becomes a ``py/ref``.
    # Every entry waits its turn, plain ones too, to keep the keys in order.
    escaped: dict[str, object] = {}
    here = state.here
    for key, item in reversed(list(mapping.items())):
        name = cast("str", key) if _is_plain_key(key) else _RawKey(key=key)
        state.stack.append((item, escaped, name, here))
    return escaped


def _encode_mappingproxy(value: object, state: _Encoding) -> object:
    """Encode a ``MappingProxyType`` as ``py/mappingproxy`` around its dict."""
    _register(value, state)
    node: dict[str, object] = {"py/mappingproxy": None}
    copied = dict(cast("Mapping[object, object]", value))
    state.stack.append((copied, node, "py/mappingproxy", state.here))
    return node


def _is_plain_key(key: object) -> bool:
    """Return whether a dict key is written as itself, needing no escape."""
    return type(key) is str and not key.startswith(("py/", "json://"))


def _encode_key(key: object, state: _Encoding) -> str:
    """Return a dict key, ``json://``-escaping non-strings and tag lookalikes."""
    if _is_plain_key(key):
        return cast("str", key)
    # A key is text, so nothing inside it has a path a later ``py/ref`` can name.
    # A ``py/ref`` to one of its parts could not resolve either: the reader finds
    # those parts under the key's own text, which is not known until the key is
    # written. So the parts get places under ``_IN_KEY``, which ``_reference``
    # never names. A ``py/ref`` to a value written before the key still works.
    mark, here = len(state.alive), state.here
    state.here = _IN_KEY
    try:
        encoded = _encode(key, state)
    finally:
        _rollback(state, mark)
        state.here = here
    return "json://" + json.dumps(encoded)


def _encode_type(value: object, state: _Encoding) -> object:
    """Encode a class as ``py/type``."""
    del state
    return {"py/type": _import_path(cast("type", value))}


def _encode_function(value: object, state: _Encoding) -> object:
    """Encode a module-level function as ``py/function``."""
    if not (isinstance(getattr(value, "__self__", None), (ModuleType, type(None)))):
        return _encode_reduce(value, state)
    try:
        return {"py/function": _import_path(cast("Callable[..., object]", value))}
    except TypeError:
        # A static builtin method (``torch.relu``) has no importable path of its
        # own; its reduce recipe, ``getattr(owner, name)``, names one that imports.
        if type(value) is not BuiltinFunctionType:
            raise
        return _encode_reduce(value, state)


def _encode_object(value: object, state: _Encoding) -> object:
    """Encode a dataclass or slotted object as ``py/object`` with its state."""
    _register(value, state)
    node: dict[str, object] = {"py/object": _import_path(type(value))}
    for name in attribute_names(value, sort=state.sort):
        member: object = getattr(value, name, PENDING)
        if member is not PENDING:
            node[name] = member
    _push_values(node, state)
    return node


def _encode_hooked(value: object, state: _Encoding) -> object:
    """Encode a value through its caller-supplied hook as ``py/hook``."""
    encode_hook = state.hooks[type(value)][0]
    _register(value, state)
    # The payload is arbitrary caller data, so it takes the same walk as any
    # other value: it may hold non-finite floats or tag-like keys.
    path = _import_path(type(value))
    node: list[object] = [path, encode_hook(value)]
    _push_items(node, state)
    return {"py/hook": node}


def _encode_reduce(value: object, state: _Encoding) -> object:
    """Encode a value by its ``__reduce__`` recipe as ``py/reduce``."""
    recipe = _canonical_reduce(value)
    if recipe is None:
        return _encode_unreduced(value, state)
    # A bare-name reduce names a module global, written as an import path.
    if isinstance(recipe, str):
        return {"py/type": _verified_path(f"{type(value).__module__}.{recipe}", value)}
    mark = len(state.alive)
    # A hashable value with no state beyond its arguments (an enum member, a
    # Decimal) is a value: each occurrence is written in full, so the output
    # does not depend on sharing.
    if is_mutable(value) or any(part is not None for part in recipe[2:]):
        _register(value, state)
    parts: list[object] = list(recipe)
    # ``parts[1]`` is the argument tuple, never ``None``, so this stops.
    while parts[-1] is None:
        parts.pop()
    checkpoint = _Checkpoint(
        value=value,
        mark=mark,
        node=state.node,
        key=state.key,
        here=state.here,
    )
    state.stack.append((checkpoint, state.node, state.key, None))
    _push_items(parts, state)
    return {"py/reduce": parts}


def _encode_unreduced(value: object, state: _Encoding) -> object:
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


_NATIVE_LEAVES: Final = frozenset((str, int, bool))


_IN_KEY: Final[_Here] = (None, "")
"""Where an escaped key sits, as ``None`` is where the whole tree sits."""


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


_python_to_plain: Final = to_plain
"""The Python walker, used when the optional Rust core is unavailable."""


def _walker[F](python: F) -> F:
    """Return the Rust core's ``to_plain`` when it serves, else ``python``."""
    return native(
        python,
        name="to_plain",
        setup="setup_encode",
        module=globals(),
    )


to_plain = _walker(_python_to_plain)
