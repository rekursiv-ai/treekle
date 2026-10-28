"""Convert Python values to plain trees.

``to_plain`` takes a ``dialect`` naming the target format, which decides what
must be tagged; ``"json"`` tags non-finite floats, which ``"python"`` keeps
native. Both tag everything else alike.
``mutable`` picks dicts and lists, the default, or ``MappingProxyType`` and
tuples; either way the result is freshly built and shared with no one.

Each tagged type has an encoder here and a decoder in ``deserialize``, paired by
name. Leaves pass through unencoded.

A container is written untagged where its declared type implies its class, as
``from_plain`` reads it. A record's ``py/object`` tag names its class, so its
field types declare its parts; ``target``, the type ``from_plain`` will be
given, declares the root. ``default_concrete_types`` says what an abstract
declared type implies. Under the default ``object`` target, the tree reads back
with no target at all.

The tag rules and their example are in ``treekle``'s package docstring;
``_implied_by`` is rule 1.
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
    TypeGuard,
    cast,
    get_args,
    get_origin,
    is_typeddict,
    overload,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import base64
import json
import weakref

from treekle.deserialize import (
    _decode_any,
    _decode_complex,
    _decode_datetime,
    _decode_dict,
    _decode_enum,
    _decode_frozenset,
    _decode_list,
    _decode_object,
    _decode_path,
    _decode_set,
    _decode_typeddict,
    _decode_untagged_proxy,
    _decode_untagged_tuple,
    _decode_uuid,
    _Decoder,
    _dispatch,
    _element_type,
    _entry_types,
    _field_types,
    _fits,
    _is_any,
    _is_protocol,
    _is_union,
    _reads_tags_as_keys,
    _resolve_alias,
    _shallow_fit,
    _tuple_types,
    _union_candidates,
    _untagged_decoder,
)
from treekle.native import native
from treekle.plain import immutable
from treekle.reflect import (
    PENDING,
    attribute_names,
    class_tags,
    concrete_overrides,
    is_mutable,
    resolve_import,
    resolves_to,
)
from treekle.types import (
    Classes,
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
    classes: Classes = ...,
    sort: bool = ...,
    target: object = ...,
    default_concrete_types: Mapping[type, type] = ...,
) -> MutablePlainTree: ...


@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
    classes: Classes = ...,
    sort: bool = ...,
    target: object = ...,
    default_concrete_types: Mapping[type, type] = ...,
) -> PlainTree: ...


def to_plain(
    obj: object,
    *,
    dialect: Dialect = "json",
    mutable: bool = True,
    hooks: Hooks = MappingProxyType({}),
    classes: Classes = MappingProxyType({}),
    sort: bool = True,
    target: object = object,
    default_concrete_types: Mapping[type, type] = MappingProxyType({}),
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
      classes: Short names to write in ``py/object`` in place of import paths;
        when given, no other class is written as ``py/object``.
      sort: Order each object's attributes by name; when false, keep slots in
        MRO order then ``__dict__`` keys in insertion order. A class with no
        instance dict has no ``__dict__`` keys, whatever its ``__getattr__`` or
        ``__getattribute__`` answers. A stored form that must stay
        byte-stable sets this false.
      target: The type ``from_plain`` will read the tree as. A container whose
        declared type there builds its class from untagged data is written
        untagged; ``object`` declares nothing, so the tree reads by tags alone.
      default_concrete_types: What untagged data reads as under an abstract
        declared type, over the baseline, as ``from_plain`` will be given it.
        A dict or list where another class is concrete is written as
        ``py/dict`` or ``py/list``.

    Returns:
      tree: Fresh plain data that ``from_plain`` turns back into ``obj``, read
        as ``target`` with the same ``default_concrete_types``.

    Raises:
      ValueError: ``dialect`` names no known format, ``classes`` names one
        class twice, or ``default_concrete_types`` names a type it cannot
        apply.
      TypeError: A leaf has no conversion and no hook, a hook returns the type
        it encodes, a dict key holds a cycle, or ``classes`` is given and lacks
        a ``py/object`` class.

    """
    if dialect not in get_args(cast("object", Dialect.__value__)):
        raise ValueError(f"unknown dialect {dialect!r}; expected 'json' or 'python'")
    state = _Encoding(
        dialect=dialect,
        hooks=hooks,
        tags=class_tags(classes),
        sort=sort,
        default_concrete_types=concrete_overrides(default_concrete_types),
    )
    tree = cast("MutablePlainTree", _encode(obj, state, target))
    return tree if mutable else immutable(tree)


@dataclass(frozen=True, slots=True, kw_only=True)
class _DeclaredParts:
    """A container's parts' declared types: one for each, or one per name or index.

    Attributes:
      each: Every part's type, where neither ``named`` nor ``indexed`` is given.
      named: Each named part's type; a name absent here is ``object``.
      indexed: Each indexed part's type; an index past the end is ``object``.

    """

    each: object = object
    named: Mapping[str, object] | None = None
    indexed: tuple[object, ...] | None = None

    def of(self, key: str | int) -> object:
        """Return the declared type of the part at ``key``.

        Args:
          key: The part's name in a mapping or record, or its index in a sequence.

        Returns:
          declared: The type ``from_plain`` reads that part as.

        """
        # A fixed tuple's parts are declared only for a tuple of its arity.
        if self.indexed is not None:
            return self.indexed[cast("int", key)]
        if self.named is not None:
            return self.named.get(key, object) if isinstance(key, str) else object
        return self.each


# A place links to its parent's rather than holding the whole path, so placing a
# node costs one tuple however deep it sits; a ``py/ref`` spells it out.
type _Here = tuple[_Here, str | int] | None
"""Where a node sits: its parent's place and its own key; ``None`` is the root."""


type _Node = list[object] | dict[str, object]
"""A container node: raw parts at first, each replaced in place by its plain form."""


type _Entry = tuple[object, _Node, str | int | _RawKey, _Here, object]
"""A part still to encode: the value, its node and key, the node's place, and the
value's declared type."""


@dataclass(slots=True, kw_only=True)
class _Encoding:
    """State of one ``to_plain`` walk.

    Attributes:
      dialect: The target format.
      hooks: Codecs for leaf types with no built-in conversion.
      tags: Each class's short name for ``py/object``; empty writes import paths.
      sort: Order each object's attributes by name, or keep declaration order.
      default_concrete_types: The class untagged data reads as under each
        abstract declared origin, where not the baseline.
      declared: The type ``from_plain`` reads the value being encoded as.
      here: Where the value being encoded sits.
      node: The node the value being encoded is written into.
      key: Where in ``node`` it is written.
      seen: Where each registered object first sits, by ``id``.
      alive: Every registered object, so its ``id`` is not reused mid-walk.
      stack: Every part still to encode, the next one last.

    """

    dialect: Dialect
    hooks: Hooks
    tags: Mapping[type, str] = field(default_factory=dict[type, str])
    sort: bool = True
    default_concrete_types: Mapping[object, type] = field(
        default_factory=dict[object, type],
    )
    declared: object = object
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
    declared: object = object


@dataclass(frozen=True, slots=True, kw_only=True)
class _Checkpoint:
    """Where a reduce restarts by protocol when one of its parts fails.

    Attributes:
      value: The value being reduced.
      mark: ``len(alive)`` before the reduce registered anything.
      node: The node the reduce is written into.
      key: Where in ``node`` it is written.
      here: Where the reduced value sits.
      declared: The reduced value's declared type.

    """

    value: object
    mark: int
    node: _Node
    key: str | int
    here: _Here
    declared: object


type _Encoder = Callable[[object, _Encoding], object]
"""Encode one value; a container pushes the parts it has yet to encode."""


# Parts wait on ``state.stack`` rather than the call stack, so a tree of any
# depth encodes; recursion stopped near a thousand levels. Each container is
# built at once with its leaves in place, and each other part is written into
# it when popped, in the order a depth-first walk reaches it.
def _encode(value: object, state: _Encoding, declared: object = object) -> object:
    """Encode ``value``, declared ``declared``, and everything it holds."""
    outer = state.stack, state.here, state.node, state.key, state.declared
    root: list[object] = [None]
    stack: list[_Entry] = []
    state.stack, state.node, state.key = stack, root, 0
    state.declared = declared
    try:
        root[0] = _encode_node(value, state)
        while stack:
            item, node, key, parent, declared = stack.pop()
            if isinstance(item, _Checkpoint):
                continue
            try:
                if isinstance(key, _RawKey):
                    key = _encode_key(key.key, state, key.declared)
                state.here, state.node, state.key = (parent, key), node, key
                state.declared = declared
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
        state.stack, state.here, state.node, state.key, state.declared = outer


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
        state.here, state.declared = checkpoint.here, checkpoint.declared
        state.node, state.key = checkpoint.node, checkpoint.key
        try:
            done = _encode_unreduced(checkpoint.value, state)
        except TypeError as again:
            error = again
            continue
        return checkpoint.node, checkpoint.key, done
    raise error


def _push_items(
    node: list[object],
    state: _Encoding,
    declared: _DeclaredParts,
) -> None:
    """Queue each part of ``node`` that is not yet plain, the first last."""
    stack, here, python = state.stack, state.here, state.dialect == "python"
    for index in range(len(node) - 1, -1, -1):
        item = node[index]
        if item is None or type(item) in _NATIVE_LEAVES:
            continue
        if type(item) is float and (python or isfinite(item)):
            continue
        stack.append((item, node, index, here, declared.of(index)))


def _push_values(
    node: dict[str, object],
    state: _Encoding,
    declared: _DeclaredParts,
    here: _Here | object = PENDING,
) -> None:
    """Queue each value of ``node`` that is not yet plain, the first last."""
    stack, python = state.stack, state.dialect == "python"
    parent = state.here if here is PENDING else cast("_Here", here)
    for key in reversed(node):
        item = node[key]
        if item is None or type(item) in _NATIVE_LEAVES:
            continue
        if type(item) is float and (python or isfinite(item)):
            continue
        stack.append((item, node, key, parent, declared.of(key)))


def _reference(place: _Here) -> dict[str, object] | None:
    """Return a ``py/ref`` naming ``place``, or ``None`` if it is inside a key."""
    keys: list[str | int] = []
    while place is not None:
        if place is _IN_KEY:
            return None
        place, key = place  # pyrefly: ignore[not-iterable] -- pyrefly drops the loop's `is not None` narrowing of a recursive alias.
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
        here = here[0]  # pyrefly: ignore[unsupported-operation] -- pyrefly drops the loop's `is not None` narrowing of a recursive alias.
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
    """Encode a path as ``py/path``, untagged where it reads so."""
    text = _untagged_leaf(value, state.declared)
    return {"py/path": str(value)} if text is None else text


def _encode_uuid(value: object, state: _Encoding) -> object:
    """Encode a UUID as ``py/uuid``, untagged where it reads so."""
    text = _untagged_leaf(value, state.declared)
    return {"py/uuid": str(value)} if text is None else text


def _encode_datetime(value: object, state: _Encoding) -> object:
    """Encode a datetime as ``py/datetime``, keeping a named zone; untagged where it reads so."""
    text = _untagged_leaf(value, state.declared)
    if text is None:
        return {"py/datetime": _datetime_text(cast("datetime", value))}
    return text


def _datetime_text(moment: datetime) -> str:
    """Return ``moment`` in ISO 8601, a named zone in brackets after it."""
    text = moment.isoformat()
    if isinstance(moment.tzinfo, ZoneInfo):
        text = f"{text}[{moment.tzinfo.key}]"
    return text


# The Rust core asks this of a leaf it holds under a declared type.
def _untagged_leaf(value: object, declared: object) -> str | int | float | None:
    """Return the text ``declared`` reads back as leaf ``value``, else ``None``."""
    if declared is object:
        return None
    if isinstance(value, Enum):
        text = cast("object", value.value)
        # A non-finite value has no plain form; its member is written by reduce.
        if type(text) not in (str, int, float) or (
            isinstance(text, float) and not isfinite(text)
        ):
            return None
    elif isinstance(value, datetime):
        text = _datetime_text(value)
    elif isinstance(value, (PurePath, UUID)):
        text = str(value)
    else:
        return None
    plain = cast("str | int | float", text)
    return plain if _settled_leaf(type(value), type(plain), declared) else None


# A leaf's verdict never reads the value: its first union member decides, and a
# member that builds the leaf's class is a class, whose fit the class settles.
# The Rust core asks this once per leaf class, text class and declared type.
def _settled_leaf(
    kind: type,
    text: type[str | int | float],
    declared: object,
) -> bool:
    """Return whether ``declared`` reads every ``kind`` leaf written as ``text`` back."""
    implied, _ = _implied_by(text(), kind, declared, {}, _KindFit(kind=kind))
    return bool(implied)


def _leaf_class(decoder: _Decoder, member: object) -> object:
    """Return the class leaf ``decoder`` builds from text under ``member``, else ``object``."""
    if decoder is _decode_path:
        return type((member if _is_path_class(member) else Path)())
    if decoder is _decode_enum:
        return member
    if decoder is _decode_uuid:
        return UUID
    return datetime if decoder is _decode_datetime else object


def _is_path_class(member: object) -> TypeGuard[type[PurePath]]:
    """Return whether ``member`` is a ``PurePath`` class."""
    return isinstance(member, type) and issubclass(member, PurePath)


def _encode_list(value: object, state: _Encoding) -> object:
    """Encode a list, registering it for ``py/ref``; ``py/list`` where it reads otherwise."""
    _register(value, state)
    untagged, _, items = _form(
        value,
        list,
        state.declared,
        state.default_concrete_types,
    )
    node = list(cast("Iterable[object]", value))
    _push_items(node, state, items)
    return node if untagged else {"py/list": node}


def _encode_tuple(value: object, state: _Encoding) -> object:
    """Encode a tuple as ``py/tuple``, untagged where it reads so; it is never referenced."""
    untagged, _, items = _form(
        value,
        tuple,
        state.declared,
        state.default_concrete_types,
    )
    node = list(cast("tuple[object, ...]", value))
    _push_items(node, state, items)
    return node if untagged else {"py/tuple": node}


def _encode_set(value: object, state: _Encoding) -> object:
    """Encode a set as ``py/set`` with members in a deterministic order."""
    _register(value, state)
    untagged, _, items = _form(value, set, state.declared, state.default_concrete_types)
    node = _ordered(cast("AbstractSet[object]", value), state, items)
    return node if untagged else {"py/set": node}


def _encode_frozenset(value: object, state: _Encoding) -> object:
    """Encode a frozenset as ``py/frozenset``; a frozenset is never referenced."""
    untagged, _, items = _form(
        value,
        frozenset,
        state.declared,
        state.default_concrete_types,
    )
    node = _ordered(cast("AbstractSet[object]", value), state, items)
    return node if untagged else {"py/frozenset": node}


def _ordered(
    members: AbstractSet[object],
    state: _Encoding,
    items: _DeclaredParts,
) -> object:
    """Return a set's members, declared ``items``, as a node in a deterministic order."""
    node = sorted(members, key=lambda member: _set_order_key(member, state, items.each))
    _push_items(node, state, items)
    return node


# The encoding leads: a default ``repr`` holds the object's address, so ordering
# by it first made a set's text differ from run to run.
def _set_order_key(
    value: object,
    state: _Encoding,
    declared: object,
) -> tuple[str, str]:
    """Return a set member's sort key, as written, registering nothing it reaches."""
    mark = len(state.alive)
    try:
        encoded = _encode(value, state, declared)
    finally:
        _rollback(state, mark)
    return json.dumps(encoded, sort_keys=True), repr(value)


def _encode_dict(value: object, state: _Encoding) -> object:
    """Encode a dict, escaping keys through ``_encode_key``; ``py/dict`` where it reads otherwise."""
    _register(value, state)
    untagged, keys, values = _form(
        value,
        dict,
        state.declared,
        state.default_concrete_types,
    )
    node = _entries(
        cast("Mapping[object, object]", value),
        state,
        keys,
        values,
        state.here,
    )
    return node if untagged else {"py/dict": node}


def _encode_mappingproxy(value: object, state: _Encoding) -> object:
    """Encode a ``MappingProxyType`` as ``py/mappingproxy`` around its dict, untagged where it reads so."""
    _register(value, state)
    untagged, keys, values = _form(
        value,
        MappingProxyType,
        state.declared,
        state.default_concrete_types,
    )
    mapping = cast("Mapping[object, object]", value)
    if untagged:
        return _entries(mapping, state, keys, values, state.here)
    # Its entries sit below the dict the tag holds, as ``from_plain`` places them.
    inner = _entries(mapping, state, keys, values, (state.here, "py/mappingproxy"))
    return {"py/mappingproxy": inner}


def _entries(
    mapping: Mapping[object, object],
    state: _Encoding,
    keys: object,
    values: _DeclaredParts,
    here: _Here,
) -> dict[str, object]:
    """Return ``mapping``'s node, its entries at ``here`` queued, keys escaped."""
    if all(map(_is_plain_key, mapping)):
        node = {cast("str", key): item for key, item in mapping.items()}
        _push_values(node, state, values, here)
        return node
    # An escaped key is encoded when its entry is reached, after every earlier
    # value, so a key naming an object written earlier becomes a ``py/ref``.
    # Every entry waits its turn, plain ones too, to keep the keys in order.
    escaped: dict[str, object] = {}
    for key, item in reversed(list(mapping.items())):
        if _is_plain_key(key):
            plain = cast("str", key)
            state.stack.append((item, escaped, plain, here, values.of(plain)))
        else:
            name = _RawKey(key=key, declared=keys)
            part = values.each if values.named is None else object
            state.stack.append((item, escaped, name, here, part))
    return escaped


def _is_plain_key(key: object) -> bool:
    """Return whether a dict key is written as itself, needing no escape."""
    return type(key) is str and not key.startswith(("py/", "json://"))


def _encode_key(key: object, state: _Encoding, declared: object = object) -> str:
    """Return a dict key, declared ``declared``, ``json://``-escaping non-strings and tag lookalikes."""
    if _is_plain_key(key):
        return cast("str", key)
    # A key is text, so nothing inside it has a path a later ``py/ref`` can name.
    # A ``py/ref`` to one of its parts could not resolve either: the reader finds
    # those parts under the key's own text, which is not known until the key is
    # written. So the parts get places under ``_IN_KEY``, which ``_reference``
    # never names. A ``py/ref`` to a value written before the key still works.
    # The caller sets ``state.here`` again before anything reads it: ``_encode``
    # for the entry's value, or ``_recover`` for a reduce restarted.
    mark = len(state.alive)
    state.here = _IN_KEY
    try:
        encoded = _encode(key, state, declared)
    finally:
        _rollback(state, mark)
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
    kind = type(value)
    _register(value, state)
    members: dict[str, object] = {}
    for name in attribute_names(value, sort=state.sort):
        member: object = getattr(value, name, PENDING)
        if member is not PENDING:
            members[name] = member
    record = is_dataclass(kind)
    implied = record and _record_implied(
        kind,
        state.declared,
        state.default_concrete_types,
    )
    # Field types declare only parts that are not plain; reading them is most of
    # a first write of a class, so a record of plain members skips it.
    types = (
        _field_types(kind)
        if record
        and (
            implied
            or any(type(member) not in _UNDECLARED for member in members.values())
        )
        else None
    )
    if implied and types is not None and members.keys() <= types.keys():
        node = members
    else:
        if state.tags and kind not in state.tags:
            raise TypeError(f"Cannot serialize {kind!r}: it is not in classes.")
        tag = state.tags[kind] if state.tags else _import_path(kind)
        node = {"py/object": tag, **members}
    _push_values(node, state, _DeclaredParts(named=types))
    return node


def _encode_hooked(value: object, state: _Encoding) -> object:
    """Encode a value through its caller-supplied hook as ``py/hook``."""
    kind = type(value)
    encode_hook = state.hooks[kind][0]
    _register(value, state)
    # The payload is arbitrary caller data, so it takes the same walk as any
    # other value: it may hold non-finite floats or tag-like keys.
    path = _import_path(kind)
    payload = encode_hook(value)
    # A payload of the hooked type is hooked again, and a new one each time, so
    # no ``py/ref`` stops it: the walk would never end.
    if type(payload) is kind:
        raise TypeError(
            f"Cannot serialize {path}: its encode hook returned a {kind.__name__}.",
        )
    node: list[object] = [path, payload]
    _push_items(node, state, _DeclaredParts())
    return {"py/hook": node}


def _encode_reduce(value: object, state: _Encoding) -> object:
    """Encode a value by its ``__reduce__`` recipe as ``py/reduce``, a leaf untagged where it reads so."""
    text = _untagged_leaf(value, state.declared)
    if text is not None:
        return text
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
        declared=state.declared,
    )
    state.stack.append((checkpoint, state.node, state.key, None, object))
    _push_items(parts, state, _DeclaredParts())
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
    if cached is not None and cached[0] is value and resolves_to(cached[1], value):
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


type _Fit = Callable[[object, bool], bool | None]
"""Whether the value fits a member, judged whole or not; ``None`` where it depends
on more than the value's class."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _ValueFit:
    """``from_plain``'s fit check of one value."""

    value: object

    def __call__(self, member: object, whole: bool) -> bool:
        if whole:
            return _fits(self.value, member, whole=True)
        return _shallow_fit(self.value, member) is not False


@dataclass(frozen=True, slots=True, kw_only=True)
class _KindFit:
    """``from_plain``'s fit check of every value of one class, where they agree."""

    kind: type
    size: int | None = None

    def __call__(self, member: object, whole: bool) -> bool | None:
        # Each answer mirrors ``_shallow_fit`` for a container or record of
        # ``kind``; a check that reads the value itself answers ``None``.
        if whole:
            return None
        target = _resolve_alias(member)
        origin: object = get_origin(target) or target
        if origin is Literal or _is_union(origin):
            return None
        if origin is type:
            return None if issubclass(self.kind, type) else False
        if not isinstance(origin, type) or origin is object or _is_any(origin):
            return True
        if is_typeddict(origin) or _is_protocol(origin):
            return None
        if not issubclass(self.kind, origin):
            return False
        if origin is tuple:
            fixed, _ = _tuple_types(target)
            if fixed is None:
                return True
            return None if self.size is None else len(fixed) == self.size
        return True


# Rule 1 of the contract in ``treekle``'s package docstring. It asks
# ``from_plain``'s own dispatch what untagged data builds, rather than restating
# it: a union reads untagged data by the first member its kind selects, a leaf's
# text passing over ``None``.
# ``fit`` judges whether the value fits the member that reads it; ``None`` where only
# the value can tell. Also returns that member.
def _implied_by(
    plain: PlainTree,
    kind: type,
    declared: object,
    default_concrete_types: Mapping[object, type],
    fit: _Fit,
) -> tuple[bool | None, object]:
    """Return whether ``plain``, written from a ``kind``, goes untagged under ``declared``."""
    if isinstance(plain, Mapping):
        builds = _MAPPING_BUILDS
    elif isinstance(plain, (str, int, float)):
        builds = None
    else:
        builds = _SEQUENCE_BUILDS
    chosen: list[tuple[object, object]] = []
    for member in _union_candidates(plain, _members(declared)):
        decoder = _untagged_decoder(member, default_concrete_types)
        if builds is None:
            if member is None or member is type(None):
                continue
            chosen.append((member, _leaf_class(decoder, member)))
            break
        built = builds.get(decoder)
        if built is not None:
            chosen.append((member, member if decoder is _decode_object else built))
    if not chosen:
        return False, object
    (member, built), *rest = chosen
    if built is not kind:
        return False, member
    # A union reads untagged data as the next member that builds it when a part
    # does not read under this one; where that member builds another class, the
    # whole value must fit this one. Judging only the node keeps a write linear.
    return fit(member, any(other is not kind for _, other in rest)), member


def _record_implied(
    kind: type,
    declared: object,
    default_concrete_types: Mapping[object, type],
) -> bool:
    """Return whether ``declared`` implies record class ``kind``, the same for every record."""
    if declared is object:
        return False
    built = _decided(
        ("record", declared),
        default_concrete_types,
        lambda: _record_built(declared, default_concrete_types),
    )
    return built is kind


# ``_implied_by`` for a record: the first member that builds from a mapping
# decides, and every record fits the class its member names.
def _record_built(
    declared: object,
    default_concrete_types: Mapping[object, type],
) -> object:
    """Return the class an untagged record under ``declared`` reads back as, else ``None``."""
    for member in _union_candidates(_NO_ENTRIES, _members(declared)):
        decoder = _untagged_decoder(member, default_concrete_types)
        built = _MAPPING_BUILDS.get(decoder)
        if built is not None:
            return member if decoder is _decode_object else built
    return None


type _Form = tuple[bool, object, _DeclaredParts]
"""Whether a container goes untagged, its keys' type, and its parts' types."""


def _form(
    value: object,
    kind: type,
    declared: object,
    default_concrete_types: Mapping[object, type],
) -> _Form:
    """Return whether ``value`` written as a ``kind`` goes untagged, and its parts' types."""
    size = len(cast("tuple[object, ...]", value)) if kind is tuple else None
    settled = _settled_form(kind, declared, default_concrete_types, size)
    if settled is not None:
        return settled
    return cast(
        "_Form",
        _form_by(kind, declared, default_concrete_types, _ValueFit(value=value)),
    )


# The Rust core asks this once per container class and declared type.
def _settled_form(
    kind: type,
    declared: object,
    default_concrete_types: Mapping[object, type],
    size: int | None = None,
) -> _Form | None:
    """Return ``_form``'s answer for every ``kind`` under ``declared``, else ``None``."""
    if declared is object:
        return kind is dict or kind is list, object, _DeclaredParts()
    found = _decided(
        ("form", kind, declared, size),
        default_concrete_types,
        lambda: _form_by(
            kind,
            declared,
            default_concrete_types,
            _KindFit(kind=kind, size=size),
        ),
    )
    return cast("_Form | None", found)


# A union reads tagged data by the first member the value fits, mapping members
# last for a value tag, and by tags alone when none fits.
def _form_by(
    kind: type,
    declared: object,
    default_concrete_types: Mapping[object, type],
    fit: _Fit,
) -> _Form | None:
    """Return ``_form``'s answer judging fits by ``fit``; ``None`` where one is ``None``."""
    mapping = kind is dict or kind is MappingProxyType
    shape = _NO_ENTRIES if mapping else ()
    implied, member = _implied_by(shape, kind, declared, default_concrete_types, fit)
    if implied is None:
        return None
    if implied:
        if is_typeddict(member):
            named = _field_types(cast("type", member))
            return True, object, _DeclaredParts(named=named)
        return True, *_part_types(member, kind)
    valued = not mapping
    for member in sorted(
        _members(declared),
        key=lambda member: (
            valued and _reads_tags_as_keys(member, default_concrete_types)
        ),
    ):
        fits = fit(member, False)
        if fits is None:
            return None
        if fits:
            return False, *_part_types(member, kind)
    return False, object, _DeclaredParts()


# Decisions name classes and types, which live as long as the code that made
# them; the bound only stops a process that makes types without end.
def _decided(
    key: tuple[object, ...],
    default_concrete_types: Mapping[object, type],
    decide: Callable[[], object],
) -> object:
    """Return ``decide()``, kept by ``key`` while ``from_plain``'s dispatch stands."""
    try:
        full = (*key, tuple(default_concrete_types.items()))
        known = _DECISIONS.get(full, _DECISIONS)
    except TypeError:
        return decide()
    dispatch = _dispatch()
    if known is not _DECISIONS:
        held, answer = cast("tuple[object, object]", known)
        if held is dispatch:
            return answer
    answer = decide()
    if len(_DECISIONS) >= _DECISION_LIMIT:
        _DECISIONS.clear()
    _DECISIONS[full] = (dispatch, answer)
    return answer


def _part_types(member: object, kind: type) -> tuple[object, _DeclaredParts]:
    """Return the key and part types a ``kind`` decoder reads under ``member``."""
    if kind is dict or kind is MappingProxyType:
        key_type, value_type = _entry_types(member)
        return key_type, _DeclaredParts(each=value_type)
    if kind is tuple:
        fixed, each = _tuple_types(member)
        return object, _DeclaredParts(each=each, indexed=fixed)
    return object, _DeclaredParts(each=_element_type(member))


def _members(declared: object) -> tuple[object, ...]:
    """Return ``declared``'s union members, nested unions spelled out, aliases resolved."""
    flat: list[object] = []
    pending = [declared]
    while pending:
        member = _resolve_alias(pending.pop())
        if _is_union(get_origin(member) or member):
            pending.extend(reversed(cast("tuple[object, ...]", get_args(member))))
        else:
            flat.append(member)
    return tuple(flat)


_NATIVE_LEAVES: Final = frozenset((str, int, bool))


_UNDECLARED: Final = frozenset((str, int, bool, float, type(None)))
"""The classes written alike under every declared type."""


_DECISIONS: Final[dict[tuple[object, ...], object]] = {}
"""Each decision ``_decided`` keeps, with the decoder table it was made under."""


_DECISION_LIMIT: Final = 4096
"""How many decisions ``_DECISIONS`` holds before it starts over."""


_NO_ENTRIES: Final = MappingProxyType[str, PlainTree]({})
"""The shape of an untagged mapping or record, whatever its entries."""


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
# reused while the entry stands. A hit is trusted only while its path still
# names the object, so a rebound or reloaded name is verified again.
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


# ``object`` marks a decoder that takes the data but builds no container of its kind.
_MAPPING_BUILDS: Final = MappingProxyType[_Decoder, type](
    {
        _decode_dict: dict,
        _decode_untagged_proxy: MappingProxyType,
        _decode_typeddict: dict,
        _decode_any: dict,
        _decode_object: object,
    },
)
"""The class each decoder builds from an untagged mapping."""


_SEQUENCE_BUILDS: Final = MappingProxyType[_Decoder, type](
    {
        _decode_list: list,
        _decode_untagged_tuple: tuple,
        _decode_set: set,
        _decode_frozenset: frozenset,
        _decode_any: list,
        _decode_complex: object,
    },
)
"""The class each decoder builds from an untagged sequence."""
