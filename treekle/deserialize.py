"""Convert plain trees to Python values.

``from_plain`` takes no dialect: it accepts the tagged and native form of every
value alike, mutable or not.

``from_plain(data, T)`` lets a tag decide where one is present and ``T``
everywhere else, so a Protocol-typed field reads back as the class its tag
names. A value that fits neither raises :class:`ReadError`; there are no silent
defaults. ``allow_imports`` gates every tag that imports or calls code, except
a tag naming the very class ``T`` asks for: the caller already holds it.

A tag only selects which decoder runs; the decoder is the same one untagged
data reaches through ``T``. Some targets (``Literal``, unions, ``object``) only
decode.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from functools import partial
from itertools import islice, repeat
from numbers import Real
from pathlib import Path, PurePath
from types import GenericAlias, MappingProxyType, SimpleNamespace, UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Literal,
    NotRequired,
    Required,
    TypeGuard,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    overload,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import base64
import binascii
import importlib
import inspect
import json
import reprlib
import sys
import typing
import weakref

from treekle.absent import ABSENT
from treekle.native import native
from treekle.reflect import (
    PENDING,
    is_mutable,
    resolve_import,
    resolves_to,
    slot_names,
)
from treekle.types import (
    FieldPath,
    Hooks,
    Invalid,
    Plain,
    PlainTree,
    ReadError,
)


if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = [
    "from_plain",
]


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: None,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T | None: ...


@overload
def from_plain[T](
    data: object,
    target: type[T],
    *,
    default: T,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> T: ...


@overload
def from_plain(
    data: object,
    target: object,
    *,
    default: object = ...,
    hooks: Hooks = ...,
    allow_imports: bool = ...,
    strict: bool = ...,
) -> object: ...


def from_plain(
    data: object,
    target: object,
    *,
    default: object = ABSENT,
    hooks: Hooks = MappingProxyType({}),
    allow_imports: bool = False,
    strict: bool = True,
) -> object:
    """Convert plain data to ``target``; a ``py/*`` tag decides where present.

    Read one field as ``from_plain(row.get("name"), str, default="")``.

    Args:
      data: A plain tree, mutable or not, as any reader or ``to_plain`` made it.
        A non-plain leaf already of its target type, such as a ``datetime``
        from ``tomllib``, passes through.
      target: The type to produce; ``object`` reads by tags alone.
      default: Returned when ``data`` is ``None`` (a missing or null field);
        omit to read ``None`` like any other value.
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
        A ``py/object`` tag naming the target dataclass itself needs none.
      strict: When false, also read numeric and boolean text (``"3"`` as
        ``3``) and integral numbers across ``int``, ``float``, and ``bool``.

    Returns:
      value: ``data`` as a ``target``, or ``default``.

    Raises:
      ReadError: ``data`` does not fit ``target``, or a tag needs imports while
        ``allow_imports`` is false.

    """
    if data is None and default is not ABSENT:
        return default
    state = _Decoding(
        hooks=hooks,
        allow_imports=allow_imports,
        strict=strict,
        untagged=not _holds_tags(data),
    )
    # A non-plain leaf is checked where it is read, by ``_passes_through``.
    tree = cast("PlainTree", data)
    try:
        value = _decode(tree, target, (), state)
    except ReadError as error:
        whole = Invalid(raw=tree, reason=str(error))
        raise ReadError(
            str(error),
            partial=whole,
            bad=MappingProxyType({**state.bad, (): whole}),
        ) from error
    if state.bad:
        at, first = next(iter(state.bad.items()))
        raise ReadError(
            f"{len(state.bad)} part(s) do not read; first at {list(at)}: "
            f"{first.reason}",
            partial=value,
            bad=MappingProxyType(state.bad),
        )
    return value


@dataclass(slots=True, kw_only=True)
class _Decoding:
    """State of one ``from_plain`` walk.

    Attributes:
      hooks: The codecs ``to_plain`` used.
      allow_imports: Whether tags may import modules and call code they name.
      strict: Whether scalars must arrive as their own plain kind.
      untagged: Whether the input holds no tag, so an ``object`` part is
        returned as the input itself rather than copied.
      built: Every numbered object, by encounter order, for ``py/id``.
      placed: Every registered object, by where it sits, for ``py/ref``.
      bad: Each part that did not read, by where it sits.
      references: How many references have resolved so far.
      unread: The data of each ``(id(data), id(union))`` that failed with no
        reference read, held so its ``id`` is not reused.
      reading: The ``id`` of each input container a frame is reading now.

    """

    hooks: Hooks
    allow_imports: bool
    strict: bool = True
    untagged: bool = False
    built: list[object] = field(default_factory=list[object])
    placed: dict[FieldPath, object] = field(default_factory=dict[FieldPath, object])
    bad: dict[FieldPath, Invalid] = field(default_factory=dict[FieldPath, Invalid])
    references: int = 0
    unread: dict[tuple[int, int], PlainTree] = field(
        default_factory=dict[tuple[int, int], PlainTree],
    )
    reading: set[int] = field(default_factory=set[int])


type _Decoder = Callable[[PlainTree, object, FieldPath, _Decoding], object]
"""Start one node, untagged or a tag's payload, as ``target``: its value or a frame."""


# Containers wait on a stack of frames rather than the call stack, so a tree of any
# depth reads; recursion stopped near a thousand levels.
def _decode(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode ``data`` as ``target``, one frame per container still being read."""
    root = _start(data, target, at, state)
    if not isinstance(root, _Frame):
        return root
    stack = [root]
    while True:
        frame = stack[-1]
        try:
            child = frame.step(state)
            if child is not None:
                stack.append(child)
                continue
            value = _finished(frame, state)
            _close(stack.pop(), state)
            if not stack:
                return value
            stack[-1].take(value, state)
        except ReadError as error:
            _unwind(stack, error, state)


def _finished(frame: _Frame, state: _Decoding) -> object:
    """Build ``frame``'s container, then convert and check it."""
    value = frame.finish(state)
    if frame.then is not None:
        value = frame.then(value)
    if frame.check is not _UNCHECKED and not _fits(value, frame.check):
        raise ReadError(f"cannot read {_shown(value)} as {frame.check}")
    return value


def _unwind(stack: list[_Frame], error: ReadError, state: _Decoding) -> None:
    """Hand the top frame's ``error`` to the nearest frame that recovers, else raise."""
    _close(stack.pop(), state)
    while stack:
        try:
            if stack[-1].fail(error, state):
                return
        except ReadError as again:
            error = again
        _close(stack.pop(), state)
    raise error


def _close(frame: _Frame, state: _Decoding) -> None:
    """Forget that ``frame``'s input container is being read."""
    if frame.reading is not None:
        state.reading.discard(frame.reading)


# A union rereads its data through each member, so only the member's frame counts.
def _start(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Start ``data`` as ``target``: its value, or a frame that reads its parts."""
    value = _start_node(data, target, at, state)
    if isinstance(value, _Frame) and not isinstance(value, _Union):
        if id(data) in state.reading:
            raise ReadError("plain data cannot hold a cycle")
        state.reading.add(id(data))
        value.reading = id(data)
    return value


def _start_node(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Start one node by its target and tags, before cycle bookkeeping."""
    # Every node under a config reads as ``object``, so this path is the hot
    # one: a leaf passes through and a tag picks its decoder with no alias,
    # dispatch, or fit check, each of which is the identity for ``object``.
    kind = type(data)
    # A leaf of exactly its declared kind reads as itself under every decoder.
    if target is kind and kind in _PLAIN_LEAVES:
        return data
    if target is object:
        if kind in _PLAIN_LEAVES:
            return data
        if kind is dict and not state.untagged:
            return _decode_tagged(cast("dict[str, PlainTree]", data), at, state)
        if kind is list and not state.untagged:
            return _decode_items(cast("list[PlainTree]", data), object, at, state)
    target = _resolve_alias(target)
    if not _is_plain(data):
        if _fits(data, target):
            return data
        # ``json.loads(parse_float=Decimal)`` and similar readers yield numbers
        # that are not ``float``; ``custom_json.convert`` read them as floats.
        if target is float and isinstance(data, (Real, Decimal)):
            return float(data)
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    decoder = _decoder_for(target)
    if decoder is _decode_union or not isinstance(data, Mapping):
        return decoder(data, target, at, state)
    if decoder is _decode_object and kind is dict:
        frame = _decode_exact_object(
            cast("dict[str, PlainTree]", data),
            target,
            at,
            state,
        )
        if frame is not None:
            return frame
    if "py/ref" in data:
        value = _decode_path_reference(data, state)
    elif "py/id" in data:
        value = _decode_reference(data, state)
    else:
        tag = _tag_of(data)
        # A mapping target reads a tag key as data, as ``custom_json.convert``
        # did; callers then pick the payload by key. Without imports no class
        # can be built, so an untyped ``py/object`` record stays a record too.
        if (
            tag is None
            or (decoder is _decode_dict and tag != "py/mappingproxy")
            or (
                tag == "py/object"
                and decoder is _decode_any
                and not state.allow_imports
            )
        ):
            return decoder(data, target, at, state)
        if tag != "py/object" and len(data) != 1:
            raise ReadError(f"invalid {tag} envelope: {_shown(dict(data))}")
        payload = data if tag == "py/object" else data[tag]
        value = _TAG_DECODERS[tag](payload, target, at, state)
        if isinstance(value, _Frame):
            value.check = target
            return value
    if not _fits(value, target):
        raise ReadError(f"cannot read {_shown(value)} as {target}")
    return value


def _decode_tagged(
    data: dict[str, PlainTree],
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Start a dict read as ``object``: by its tag, else as a plain dict."""
    # References come first, in ``_start``'s order, so both paths read one
    # malformed node alike.
    if "py/ref" in data:
        return _decode_path_reference(data, state)
    if "py/id" in data:
        return _decode_reference(data, state)
    if "py/object" in data:
        if not state.allow_imports:
            return _decode_entries(data, object, object, at, state)
        return _decode_stored(data, at, state)
    if len(data) == 1:
        decoder = _TAG_DECODERS.get(next(iter(data)))
        if decoder is not None:
            return decoder(next(iter(data.values())), object, at, state)
    tag = _tag_of(data)
    if tag is None:
        return _decode_entries(data, object, object, at, state)
    raise ReadError(f"invalid {tag} envelope: {_shown(data)}")


def _is_plain(data: object) -> bool:
    """Return whether ``data``'s top node is plain: a leaf, mapping, or sequence."""
    return data is None or (
        isinstance(data, (str, int, float, Mapping, Sequence))
        and not isinstance(data, (bytes, bytearray))
    )


def _decoder_for(target: object) -> _Decoder:
    """Return the decoder for ``target``, keyed on its origin."""
    origin: object = get_origin(target) or target
    decoder = _DECODERS.get(origin)
    if decoder is not None:
        return decoder
    if not _is_class(origin):
        return _decode_any
    if Enum in origin.__mro__:
        return _decode_enum
    if PurePath in origin.__mro__:
        return _decode_path
    if is_dataclass(origin):
        return _decode_object
    if is_typeddict(origin):
        return _decode_typeddict
    if _is_protocol(origin) or inspect.isabstract(origin) or _is_any(origin):
        return _decode_any
    return _decode_object


def _tag_of(node: Mapping[str, PlainTree]) -> str | None:
    """Return the highest-precedence ``py/*`` tag on ``node``, if any."""
    if not _has_tag_key(node):
        return None
    for tag in _TAGS:
        if tag in node:
            return tag
    raise ReadError(f"unknown tag among {sorted(map(str, node))}")


def _holds_tags(data: object) -> bool:
    """Return whether any mapping in ``data`` has a tag or escaped key."""
    stack = [data]
    # Keep each node, not just its id: a custom container may build new parts on
    # every read, and a dropped part frees its id for the next one.
    seen: dict[int, object] = {}
    while stack:
        node = stack.pop()
        if isinstance(node, (str, int, float)) or node is None:
            continue
        if id(node) in seen:
            continue
        seen[id(node)] = node
        if isinstance(node, Mapping):
            mapping = cast("Mapping[object, object]", node)
            if any(
                isinstance(key, str) and key.startswith(("py/", "json://"))
                for key in mapping
            ):
                return True
            stack.extend(mapping.values())
        elif isinstance(node, Sequence) and not isinstance(
            node,
            (str, bytes, bytearray),
        ):
            stack.extend(node)
    return False


def _has_tag_key(keys: Iterable[object]) -> bool:
    """Return whether any key reads as a tag; a YAML reader may yield non-str keys."""
    return any(isinstance(key, str) and key.startswith("py/") for key in keys)


type _Parts = tuple[bool, Iterator[tuple[object, object]]]
"""Pairs of ``(value, target)`` left to judge: all must fit when ``True``, else one."""


type _Judging = tuple[bool, Iterator[tuple[object, object]], tuple[int, int]]
"""Parts left to judge, with the ``(id(value), id(target))`` they belong to."""


# Judged with a stack, as decoding is: a referenced value is as deep as its data. A
# pair met again while still being judged fits; the rest of the walk decides.
def _fits(value: object, target: object) -> bool:
    """Return whether a tag-decoded or referenced ``value`` is acceptable as ``target``."""
    key = (id(value), id(target))
    verdict = _shallow_fit(value, target)
    stack: list[_Judging] = []
    judging: set[tuple[int, int]] = set()
    while True:
        if isinstance(verdict, bool):
            while stack and verdict != stack[-1][0]:
                judging.discard(stack.pop()[2])
            if not stack:
                return verdict
        else:
            stack.append((*verdict, key))
            judging.add(key)
        needs_all, parts, _ = stack[-1]
        pair = next(parts, None)
        if pair is None:
            judging.discard(stack.pop()[2])
            verdict = needs_all
        else:
            key = (id(pair[0]), id(pair[1]))
            verdict = key in judging or _shallow_fit(*pair)


# ``isinstance`` alone refuses to judge a ``TypedDict`` or a Protocol that is not
# runtime-checkable, so those are checked by their declared structure. A referenced
# value was decoded as whatever its first occurrence was, and a reduce, hook, or
# ``py/object`` builds its items untyped, so items are checked against ``target``'s
# element types too. Only the containers a reference can name are walked: lists,
# sets, and mappings; a tuple or frozenset was read item by item already.
def _shallow_fit(value: object, target: object) -> bool | _Parts:
    """Judge ``value`` against ``target`` alone, or return the parts left to judge."""
    # A part that did not read is already recorded where it sits.
    if isinstance(value, Invalid):
        return True
    target = _resolve_alias(target)
    origin: object = get_origin(target) or target
    args = cast("tuple[object, ...]", get_args(target))
    if _is_union(origin):
        return False, iter([(value, member) for member in args])
    if origin is Literal:
        return value in args
    if origin is type:
        return isinstance(value, type) and (not args or _is_subclass(value, args[0]))
    if not _is_class(origin) or origin is object or _is_any(origin):
        return True
    if is_typeddict(origin):
        return isinstance(value, Mapping) and _required_keys(origin) <= set(
            cast("Mapping[object, object]", value),
        )
    if _is_protocol(origin):
        return _has_members(value, origin)
    if not isinstance(value, origin):
        return False
    if isinstance(value, Mapping) and len(args) == 2:
        mapping = cast("Mapping[object, object]", value)
        key_type, item_type = args
        return True, (
            pair
            for key, member in mapping.items()
            for pair in ((key, key_type), (member, item_type))
        )
    if isinstance(value, (list, set)) and args:
        items = cast("list[object] | set[object]", value)
        return True, ((item, args[0]) for item in items)
    return True


def _is_subclass(kind: type, bound: object) -> bool:
    """Return whether class ``kind`` lies inside ``type[bound]``."""
    bound = _resolve_alias(bound)
    origin: object = get_origin(bound) or bound
    if _is_union(origin):
        return any(
            _is_subclass(kind, member)
            for member in cast("tuple[object, ...]", get_args(bound))
        )
    if not _is_class(origin) or origin is object or _is_any(origin):
        return True
    if _is_protocol(origin):
        return _has_members(kind, origin)
    return issubclass(kind, origin)


def _is_union(origin: object) -> bool:
    """Return whether ``origin`` is a union's origin."""
    return any(origin is kind for kind in _UNION_ORIGINS)


def _is_protocol(kind: type) -> bool:
    """Return whether ``kind`` is a Protocol class."""
    return kind.__dict__.get("_is_protocol") is True


def _is_any(kind: type) -> bool:
    """Return whether ``kind`` is ``typing.Any``, a class ``isinstance`` refuses."""
    return kind.__module__ == "typing" and kind.__qualname__ == "Any"


def _has_members(value: object, protocol: type) -> bool:
    """Return whether ``value`` has every member ``protocol`` declares."""
    members = cast("frozenset[str]", vars(protocol)["__protocol_attrs__"])
    return all(hasattr(value, name) for name in members)


def _decode_reference(
    node: Mapping[str, PlainTree],
    state: _Decoding,
) -> object:
    """Return the object a ``py/id`` envelope names."""
    index = node["py/id"]
    if (
        len(node) != 1
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(state.built)
    ):
        raise ReadError(f"invalid py/id reference: {_shown(dict(node))}")
    state.references += 1
    value = state.built[index]
    if value is PENDING:
        raise ReadError(f"py/id {index} names a value still being built")
    return value


def _decode_path_reference(
    node: Mapping[str, PlainTree],
    state: _Decoding,
) -> object:
    """Return the object a ``py/ref`` envelope names by its first path."""
    path = node["py/ref"]
    if (
        len(node) != 1
        or isinstance(path, (str, Mapping))
        or not isinstance(path, Sequence)
        or not all(type(step) in (str, int) for step in path)
    ):
        raise ReadError(f"invalid py/ref reference: {_shown(dict(node))}")
    state.references += 1
    value = state.placed.get(tuple(cast("Sequence[str | int]", path)), _ABSENT_REF)
    if value is _ABSENT_REF:
        raise ReadError(f"py/ref {_shown(list(path))} names no earlier value")
    if value is PENDING:
        raise ReadError(f"py/ref {_shown(list(path))} names a value still being built")
    return value


def _place(value: object, at: FieldPath, state: _Decoding) -> None:
    """Record ``value`` by encounter number for ``py/id`` and by path for ``py/ref``."""
    state.built.append(value)
    state.placed[at] = value


def _replace(index: int, value: object, at: FieldPath, state: _Decoding) -> None:
    """Fill a slot ``_place`` reserved with ``PENDING`` once ``value`` is built."""
    state.built[index] = value
    state.placed[at] = value


def _invalid(
    value: object,
    reason: str,
    at: FieldPath,
    state: _Decoding,
) -> Invalid:
    """Record ``value`` as bad at ``at`` and return its placeholder."""
    invalid = Invalid(raw=cast("PlainTree", value), reason=reason)
    state.bad[at] = invalid
    return invalid


def _start_part(
    raw: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Start a container's part; one that does not read becomes its :class:`Invalid`."""
    try:
        return _start(raw, target, at, state)
    except ReadError as error:
        return _invalid(raw, str(error), at, state)


def _decode_none(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> None:
    """Decode ``None``."""
    del target, at, state
    if data is not None:
        raise ReadError(f"cannot read {_shown(data)} as None")


def _decode_bool(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> bool:
    """Decode a bool; lax also reads ``0``/``1`` and true/false text."""
    del target, at
    if isinstance(data, bool):
        return data
    if not state.strict:
        if isinstance(data, str):
            data = _LAX_BOOLS.get(data.lower(), data)
        if isinstance(data, int) and data in (0, 1):
            return bool(data)
    raise ReadError(f"cannot read {_shown(data)} as bool")


def _decode_int(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> int:
    """Decode an int; lax also reads integral floats and numeric text."""
    del target, at
    if isinstance(data, int) and not isinstance(data, bool):
        return data
    if not state.strict and not isinstance(data, bool):
        # Parse integer text directly. Through a float, it would round past 2**53,
        # and text past the float range would turn into infinity and be refused.
        if isinstance(data, str):
            try:
                return int(data)
            except ValueError:
                pass
        number = _lax_number(data)
        # ``is_integer`` is false for a non-finite float, so ``int`` cannot raise.
        if number is not None and number.is_integer():
            # Parse float-form text, such as ``"9007199254740993.0"``, again as a
            # ``Decimal`` to get its exact value. ``Decimal`` accepts any text
            # ``float`` did. Text that is integral only after rounding to a float,
            # such as ``"1.0000000000000000001"``, keeps the rounded value.
            if isinstance(data, str):
                exact = Decimal(data)
                if exact == exact.to_integral_value():
                    return int(exact)
            return int(number)
    raise ReadError(f"cannot read {_shown(data)} as int")


def _lax_number(data: PlainTree) -> float | None:
    """Return the number ``data`` is or spells, else ``None``."""
    if isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            return None
    return data if isinstance(data, float) else None


def _decode_str(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> str:
    """Decode a str."""
    del target, at, state
    if isinstance(data, str):
        return data
    raise ReadError(f"cannot read {_shown(data)} as str")


def _decode_float(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> float:
    """Decode a float from an int or a float; lax also reads float text."""
    del target, at
    if isinstance(data, float):
        return data
    if isinstance(data, int) and not isinstance(data, bool):
        return float(data)
    if isinstance(data, str) and not state.strict:
        try:
            return float(data.strip())
        except ValueError:
            pass
    raise ReadError(f"cannot read {_shown(data)} as float")


def _decode_float_tag(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> float:
    """Decode a ``py/float`` payload: the ``repr`` of a non-finite float."""
    del target, at, state
    # The tag is the one place strict data spells a float as text, so it reads
    # only what ``to_plain`` writes there; lax text is ``_decode_float``'s.
    if isinstance(data, str) and data in _NON_FINITE_REPRS:
        return float(data)
    raise ReadError(f"invalid py/float payload: {_shown(data)}")


def _decode_complex(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a complex: a real number, or a ``py/complex`` pair of floats."""
    del target, state
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return complex(data)
    if isinstance(data, Sequence) and not isinstance(data, str) and len(data) == 2:
        # Each part is a tree of its own: a non-finite one arrives as ``py/float``.
        jobs = [(part, float, (*at, index)) for index, part in enumerate(data)]
        return _Steps(at, jobs=jobs, done=_complex_of)
    raise ReadError(f"cannot read {_shown(data)} as complex")


def _complex_of(parts: list[object]) -> object:
    """Return the complex number of a decoded real and imaginary part."""
    real, imag = parts
    return complex(cast("float", real), cast("float", imag))


def _decode_bytes(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> bytes:
    """Decode bytes from base64 text."""
    del target, at, state
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ReadError(f"cannot read {_shown(data)} as bytes: {error}") from error
    raise ReadError(f"cannot read {_shown(data)} as bytes")


def _decode_path(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> PurePath:
    """Decode text as the ``PurePath`` class ``target`` names, else as a ``Path``."""
    del at, state
    kind = target if isinstance(target, type) and issubclass(target, PurePath) else Path
    if isinstance(data, str):
        return kind(data)
    raise ReadError(f"cannot read {_shown(data)} as {kind.__name__}")


def _decode_uuid(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> UUID:
    """Decode a ``UUID`` from its canonical text."""
    del target, at, state
    if isinstance(data, str):
        try:
            return UUID(data)
        except ValueError as error:
            raise ReadError(f"cannot read {_shown(data)} as UUID") from error
    raise ReadError(f"cannot read {_shown(data)} as UUID")


def _decode_datetime(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> datetime:
    """Decode an ISO 8601 ``datetime``, restoring a bracketed zone name."""
    del target, at, state
    if not isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as datetime")
    text, bracket, zone = data.partition("[")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ReadError(f"cannot read {_shown(data)} as datetime") from error
    if not bracket:
        return moment
    if moment.utcoffset() is None:
        raise ReadError(f"cannot read {_shown(data)} as datetime: timestamp is naive")
    if not zone.endswith("]"):
        raise ReadError(f"cannot read {_shown(data)} as datetime: invalid named zone")
    try:
        named = ZoneInfo(zone.removesuffix("]"))
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ReadError(
            f"cannot read {_shown(data)} as datetime: invalid named zone",
        ) from error
    return moment.astimezone(named)


def _decode_enum(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Enum:
    """Decode an untagged value as a member of ``target``; tagged ones are reduces."""
    del at, state
    # ``_decoder_for`` picks this decoder only for a class with ``Enum`` in its MRO.
    return _enum_member(data, cast("type[Enum]", target))


def _enum_member(data: PlainTree, kind: type[Enum]) -> Enum:
    """Return ``kind``'s member whose value ``data`` is; a bool names none."""
    if isinstance(data, (str, int, float)) and not isinstance(data, bool):
        try:
            return kind(data)
        except ValueError:
            pass
    raise ReadError(f"cannot read {_shown(data)} as {kind}")


def _decode_literal(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Plain | Enum:
    """Decode one of a ``Literal``'s values, matching bool and int apart."""
    del at, state
    for option in cast("tuple[object, ...]", get_args(target)):
        if isinstance(option, Enum):
            try:
                member = _enum_member(data, type(option))
            except ReadError:
                continue
            if member is option:
                return option
        elif type(option) is type(data) and option == data:
            return cast("Plain", option)
    raise ReadError(f"cannot read {_shown(data)} as {target}")


def _decode_list(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a list, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    return _decode_items(data, _element_type(target), at, state)


def _decode_items(
    data: Sequence[PlainTree],
    item: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Start a sequence's items as ``item`` into a list numbered for ``py/id``."""
    if item is object and all(type(raw) in _PLAIN_LEAVES for raw in data):
        leaves = list[object](data)
        _place(leaves, at, state)
        return leaves
    result: list[object] = []
    _place(result, at, state)
    return _Items(at, raws=data, types=repeat(item), values=result, done=_as_built)


def _decode_tuple(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a fixed or variadic tuple; tuples take no ``py/id`` number."""
    del state
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    if get_origin(target) is not tuple:
        types = repeat(_element_type(target))
    elif len(args) == 2 and args[1] is Ellipsis:
        types = repeat(args[0])
    elif len(args) != len(data):
        raise ReadError(f"expected {len(args)} items for {target}, got {len(data)}")
    else:
        types = iter(args)
    return _Items(at, raws=data, types=types, values=[], done=tuple)


def _decode_set(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a set, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    result: set[object] = set()
    _place(result, at, state)
    return _SetItems(at, raws=data, item=_element_type(target), members=result)


def _decode_frozenset(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a frozenset; frozensets take no ``py/id`` number."""
    del state
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    types = repeat(_element_type(target))
    return _Items(at, raws=data, types=types, values=[], done=_frozenset_of)


def _frozenset_of(members: list[object]) -> object:
    """Return ``members`` as a frozenset, refusing an unhashable one."""
    try:
        return frozenset(members)
    except TypeError as error:
        raise ReadError(
            f"frozenset members {_shown(members)} are unhashable",
        ) from error


def _as_built(values: list[object]) -> object:
    """Return a list node as the list its parts were read into."""
    return values


def _decode_dict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a dict, typing keys and values, numbering it for ``py/id``."""
    if not isinstance(data, Mapping):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    key_type, value_type = args if len(args) == 2 else (object, object)
    return _decode_entries(data, key_type, value_type, at, state)


def _decode_entries(
    data: Mapping[str, PlainTree],
    key_type: object,
    value_type: object,
    at: FieldPath,
    state: _Decoding,
) -> _Entries:
    """Start a mapping's keys and values into a dict numbered for ``py/id``."""
    result: dict[Hashable, object] = {}
    _place(result, at, state)
    return _Entries(
        at,
        data=data,
        key_type=key_type,
        value_type=value_type,
        result=result,
    )


# ``to_plain`` registers the proxy at its own path and the copied dict one step below
# it, under ``py/mappingproxy``; a ``py/ref`` may name either.
def _decode_mappingproxy(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a ``MappingProxyType`` around a decoded dict, placed before its dict."""
    inner: dict[Hashable, object] = {}
    value = MappingProxyType(inner)
    state.placed[at] = value
    if not isinstance(data, Mapping):
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    key_type, value_type = args if len(args) == 2 else (object, object)
    frame = _decode_entries(data, key_type, value_type, (*at, "py/mappingproxy"), state)
    frame.then = partial(_fill_proxy, inner=inner, proxy=value)
    return frame


def _fill_proxy(
    decoded: object,
    *,
    inner: dict[Hashable, object],
    proxy: MappingProxyType[Hashable, object],
) -> object:
    """Fill the dict behind ``proxy`` with ``decoded`` entries and return ``proxy``."""
    inner.update(cast("dict[Hashable, object]", decoded))
    return proxy


def _bad_key(key: Hashable, target: object) -> Hashable:
    """Raise for a non-str key that does not fit ``target``."""
    raise ReadError(f"cannot read key {_shown(key)} as {target}")


# A YAML reader yields non-str keys such as ``True``; they read as is.
def _decode_key(
    key: Hashable,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Hashable:
    """Return a dict key as ``target``, unescaping a ``json://`` key."""
    if not isinstance(key, str):
        return key if _fits(key, target) else _bad_key(key, target)
    raw: PlainTree = key
    if key.startswith("json://"):
        try:
            raw = cast(
                "PlainTree",
                json.loads(key.removeprefix("json://")),
            )
        except json.JSONDecodeError as error:
            raise ReadError(f"invalid escaped key {_shown(key)}") from error
    decoded = _decode(raw, target, at, state)
    # Call ``hash`` because the dict insert does. The ``Hashable`` check is not
    # enough: it passes a tuple holding a list, and any class registered with it.
    try:
        hash(decoded)
    except TypeError as error:
        raise ReadError(f"key {_shown(decoded)} is unhashable") from error
    return cast("Hashable", decoded)


def _decode_type(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a ``py/type`` path to the class or module global it names."""
    del at
    if not isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as a class path")
    resolved = _resolve(data, state)
    if (get_origin(target) or target) is type and not isinstance(resolved, type):
        raise ReadError(f"{_shown(data)} does not name a class")
    return resolved


def _decode_function(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Callable[..., object]:
    """Decode a ``py/function`` path to the function it names."""
    del target, at
    if not isinstance(data, str):
        raise ReadError(f"cannot read {_shown(data)} as a function path")
    resolved = _resolve(data, state)
    if not callable(resolved):
        raise ReadError(f"{_shown(data)} does not name a callable")
    return resolved


def _decode_typeddict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a ``TypedDict``: declared keys by their types, unknown ones dropped."""
    kind = cast("type", target)
    if not isinstance(data, Mapping):
        raise ReadError(f"expected object for {kind.__name__}, got {_shown(data)}")
    hints = _field_types(kind)
    missing = sorted(_required_keys(kind) - set(data))
    if missing:
        raise ReadError(f"{kind.__name__}: missing required field(s) {missing}")
    result: dict[str, object] = {}
    _place(result, at, state)
    return _Fields(at, data=data, types=hints, members=result)


# ``__required_keys__`` applies each declaring class's own totality, but under
# ``from __future__ import annotations`` it cannot see a ``Required`` or
# ``NotRequired`` written as a string; the resolved hints carry the qualifier.
def _required_keys(kind: type) -> frozenset[str]:
    """Return a TypedDict's required keys: by totality, unless a hint says."""
    by_totality = cast("frozenset[str]", vars(kind)["__required_keys__"])
    hints, _ = _resolvable_hints(kind, include_extras=True)
    return frozenset(
        key
        for key, hint in hints.items()
        if get_origin(hint) is Required
        or (get_origin(hint) is not NotRequired and key in by_totality)
    )


def _decode_object(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode an object: the ``py/object`` class, else ``target``'s dataclass."""
    name = target.__name__ if isinstance(target, type) else str(target)
    if not isinstance(data, Mapping):
        raise ReadError(f"expected object for {name}, got {_shown(data)}")
    stored = "py/object" in data
    if _names_dataclass(data, target, state):
        fields_only: dict[str, PlainTree] = {
            key: raw for key, raw in data.items() if key != "py/object"
        }
        data = fields_only
    if "py/object" in data:
        return _decode_stored(data, at, state)
    if not isinstance(target, type) or not is_dataclass(target):
        raise ReadError(f"cannot read {_shown(dict(data))} as {name}")
    types = _field_types(target)
    unknown = sorted(set(data) - set(types))
    if unknown:
        raise ReadError(
            f"{name}: unknown field(s) {unknown}; valid: {sorted(types)}",
        )
    index = len(state.built)
    # A ``py/object`` record is state ``to_plain`` took from a built object, so it
    # is restored as is: running ``__init__`` again would rerun a ``__post_init__``
    # that already transformed it. Allocated before its fields read, it is there
    # for a field that refers back to it. An untagged record is constructor input,
    # so it goes through ``__init__``, which validates it. A part that did not
    # read leaves ``__init__`` unrunnable either way.
    value = _allocate(target) if stored else PENDING
    _place(value, at, state)
    return _Object(
        at,
        data=data,
        types=types,
        target=target,
        stored=stored,
        value=value,
        index=index,
        before=len(state.bad),
    )


# Taken for a dict whose keys are all ``str``, each a field of ``target`` but a
# ``py/object`` naming ``target`` itself. Such a record reads as ``_start_node`` and
# ``_decode_object`` would read it, minus their reference and tag lookups and the copy
# without the tag; anything else, refusals included, goes the full path, which words
# every error.
def _decode_exact_object(
    data: dict[str, PlainTree],
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> _Object | None:
    """Start a dataclass record in one pass, else ``None`` for the full path."""
    if (
        not isinstance(target, type)
        or not is_dataclass(target)
        # ``_names_dataclass`` asks ``not allow_imports`` of each record, which
        # a flag that is not a bool answers with its own ``__bool__``.
        or type(cast("object", state.allow_imports)) is not bool
    ):
        return None
    types = _field_types(target)
    stored = False
    for key, raw in data.items():
        if type(key) is not str:
            return None
        if key == "py/object":
            if (
                type(raw) is not str
                or raw != f"{target.__module__}.{target.__qualname__}"
            ):
                return None
            stored = True
        elif key not in types:
            return None
    if stored and "py/object" in types:
        return None
    index = len(state.built)
    value = _allocate(target) if stored else PENDING
    _place(value, at, state)
    # The full path also has a stored record's built object checked against
    # ``target``; that object is exactly ``target``, which ``isinstance``
    # answers without any ``__instancecheck__``, so the check is left out.
    return _Object(
        at,
        data=data,
        types=types,
        target=target,
        stored=stored,
        value=value,
        index=index,
        before=len(state.bad),
    )


def _decode_stored(
    data: Mapping[str, PlainTree],
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Rebuild the class a ``py/object`` record names, its attributes untyped."""
    path = data["py/object"]
    if not isinstance(path, str):
        raise ReadError(f"invalid py/object path: {_shown(path)}")
    kind = _resolve(path, state)
    if not isinstance(kind, type):
        raise ReadError(f"{_shown(path)} does not name a class")
    value = _allocate(kind)
    _place(value, at, state)
    return _Stored(at, data=data, value=value)


def _with_defaults(
    declared: tuple[Field[object], ...],
    members: dict[str, object],
    name: str,
) -> dict[str, object]:
    """Return ``members`` with each absent defaulted field filled from its default."""
    filled = dict(members)
    missing: list[str] = []
    for item in declared:
        if item.name in filled:
            continue
        if item.default is not MISSING:
            filled[item.name] = item.default
        elif item.default_factory is not MISSING:
            filled[item.name] = item.default_factory()
        elif item.init:
            missing.append(item.name)
    if missing:
        raise ReadError(f"cannot build {name}: missing field(s) {missing}")
    return filled


# It does when the tag names ``target``, or when imports are off: the tag's class cannot
# be built then, and data written before a class moved still names its old path.
def _names_dataclass(
    data: Mapping[str, PlainTree],
    target: object,
    state: _Decoding,
) -> bool:
    """Return whether dataclass ``target`` reads ``data`` with its own fields."""
    return (
        isinstance(target, type)
        and is_dataclass(target)
        and (
            not state.allow_imports
            or data.get("py/object") == f"{target.__module__}.{target.__qualname__}"
        )
    )


def _decode_hooked(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Rebuild a ``py/hook`` value through its registered decode callback."""
    del target
    path, payload = _pair(data, "py/hook")
    hook = next(
        (
            pair
            for kind, pair in state.hooks.items()
            if f"{kind.__module__}.{kind.__qualname__}" == path
        ),
        None,
    )
    if hook is None:
        raise ReadError(f"hook {_shown(path)} is not registered")
    # The encoder numbers the hooked value before its payload, so the slot is
    # reserved in that order and filled once the hook rebuilds it.
    index = len(state.built)
    _place(PENDING, at, state)
    return _Steps(
        at,
        jobs=[(payload, object, (*at, 1))],
        done=partial(
            _hooked,
            decode=hook[1],
            path=path,
            index=index,
            at=at,
            state=state,
        ),
    )


def _hooked(
    parts: list[object],
    *,
    decode: Callable[..., object],
    path: PlainTree,
    index: int,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Rebuild a hooked value from its decoded payload and fill its reserved slot."""
    # A hook refusing its payload is bad data, as a reduce call refusing its
    # arguments is: the rest of the tree still reads.
    try:
        value = decode(parts[0])
    except (TypeError, ValueError) as error:
        raise ReadError(f"hook {_shown(path)} failed: {error}") from error
    _replace(index, value, at, state)
    return value


def _decode_reduce(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Replay a ``py/reduce`` recipe; one naming ``target`` itself needs no import."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"invalid py/reduce payload: {_shown(data)}")
    if len(data) < 2 or len(data) > 5:
        raise ReadError("py/reduce requires two to five elements")
    stateful = any(part is not None for part in data[2:])
    index = len(state.built)
    if stateful:
        _place(PENDING, at, state)
    return _Reduce(at, data=data, target=target, stateful=stateful, index=index)


def _apply_state(value: object, attributes: object) -> None:
    """Apply reduce state through ``__setstate__`` or direct attributes."""
    setstate: object = getattr(value, "__setstate__", None)
    if callable(setstate):
        setstate(attributes)
        return
    chunks: tuple[object, ...] = (attributes, None)
    if isinstance(attributes, tuple):
        pair = cast("tuple[object, ...]", attributes)
        if len(pair) == 2:
            chunks = pair
    for chunk in chunks:
        if isinstance(chunk, dict):
            for key, member in cast("dict[str, object]", chunk).items():
                object.__setattr__(value, key, member)


def _decode_any(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode untagged data for ``object``, ``Any``, or a Protocol, as is."""
    del target
    # Callers edit a read record's untyped members in place, so tag-free data
    # must come back as the very objects passed in.
    if state.untagged or data is None or isinstance(data, (str, int, float)):
        return data
    if isinstance(data, Mapping):
        return _decode_entries(data, object, object, at, state)
    return _decode_items(data, object, at, state)


def _decode_union(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode the member the data's kind selects, else the first that fits."""
    # Each member a parent union tries reaches this node again; remembering the
    # failure keeps nested unions linear instead of exponential in depth. A read
    # that resolved a reference may succeed elsewhere, where it names another
    # value, so it is never remembered.
    key = (id(data), id(target))
    if key in state.unread:
        raise ReadError(f"cannot read {_shown(data)} as {target}")
    members = cast("tuple[object, ...]", get_args(target))
    if isinstance(data, Mapping) and _has_tag_key(data):
        # The member a ``py/object`` tag names goes first, so member order
        # cannot pick a sibling whose fields happen to fit too.
        path = cast("Mapping[str, PlainTree]", data).get("py/object")
        members = tuple(
            sorted(members, key=lambda member: _import_name(member) != path),
        )
    else:
        members = _union_candidates(data, members)
    return _Union(
        at,
        data=data,
        target=target,
        members=members,
        key=key,
        references=state.references,
    )


class _Frame:
    """A container being read: ``step`` starts its parts, ``finish`` builds it.

    Attributes:
      at: Where the container sits.
      check: The type a tag-decoded value must fit, else ``_UNCHECKED``.
      then: Applied to the built value before it is checked and handed up.
      pending_raw: The part ``step`` last handed out as a frame of its own.
      pending_at: Where that part sits.
      reading: The ``id`` of the input container this frame reads, if any.

    """

    __slots__ = ("at", "check", "pending_at", "pending_raw", "reading", "then")

    def __init__(self, at: FieldPath) -> None:
        self.at = at
        self.check: object = _UNCHECKED
        self.then: Callable[[object], object] | None = None
        self.pending_raw: PlainTree = None
        self.pending_at: FieldPath = at
        self.reading: int | None = None

    def step(self, state: _Decoding) -> _Frame | None:
        """Read parts until one needs a frame of its own; return it, or ``None``."""
        del state
        return None

    def take(self, value: object, state: _Decoding) -> None:
        """Accept the value the part handed out last was read as."""
        del value, state
        raise NotImplementedError

    def fail(self, error: ReadError, state: _Decoding) -> bool:
        """Recover from the part handed out last failing; ``False`` passes it on."""
        del error, state
        return False

    def finish(self, state: _Decoding) -> object:
        """Build the container from its parts."""
        del state
        raise NotImplementedError


class _PartFrame(_Frame):
    """A container whose parts each read on their own: a bad one is recorded."""

    __slots__ = ()

    @override
    def fail(self, error: ReadError, state: _Decoding) -> bool:
        self.take(_invalid(self.pending_raw, str(error), self.pending_at, state), state)
        return True


class _Items(_PartFrame):
    """A sequence read part by part, each part as the next of ``types``."""

    __slots__ = ("done", "parts", "values")

    def __init__(
        self,
        at: FieldPath,
        *,
        raws: Sequence[PlainTree],
        types: Iterator[object],
        values: list[object],
        done: Callable[[list[object]], object],
    ) -> None:
        super().__init__(at)
        self.parts = enumerate(zip(raws, types, strict=False))
        self.values = values
        self.done = done

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        at = self.at
        for index, (raw, target) in self.parts:
            # ``_start_node``'s first answer: a plain leaf read as ``object``
            # or as exactly its own type is itself.
            if (target is object or target is type(raw)) and type(raw) in _PLAIN_LEAVES:
                self.take(raw, state)
                continue
            where = (*at, index)
            value = _start_part(raw, target, where, state)
            if isinstance(value, _Frame):
                self.pending_raw, self.pending_at = raw, where
                return value
            self.take(value, state)
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        del state
        self.values.append(value)

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.done(self.values)


class _SetItems(_Items):
    """A set read part by part into ``members``, refusing an unhashable part."""

    __slots__ = ("members",)

    def __init__(
        self,
        at: FieldPath,
        *,
        raws: Sequence[PlainTree],
        item: object,
        members: set[object],
    ) -> None:
        super().__init__(
            at,
            raws=raws,
            types=repeat(item),
            values=[],
            done=_as_built,
        )
        self.members = members

    @override
    def take(self, value: object, state: _Decoding) -> None:
        del state
        try:
            self.members.add(value)
        except TypeError as error:
            raise ReadError(f"set member {_shown(value)} is unhashable") from error

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.members


class _Entries(_PartFrame):
    """A mapping read entry by entry: each key as ``key_type``, value as ``value_type``."""

    __slots__ = ("key_type", "parts", "pending_key", "result", "value_type")

    def __init__(
        self,
        at: FieldPath,
        *,
        data: Mapping[str, PlainTree],
        key_type: object,
        value_type: object,
        result: dict[Hashable, object],
    ) -> None:
        super().__init__(at)
        self.parts = iter(data.items())
        self.key_type = key_type
        self.value_type = value_type
        self.result = result
        self.pending_key: Hashable = None

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        at, key_type, value_type, result = (
            self.at,
            self.key_type,
            self.value_type,
            self.result,
        )
        for key, raw in self.parts:
            where = (*at, key)
            if key_type is object and type(key) is str and key[:7] != "json://":
                decoded: Hashable = key
            else:
                try:
                    decoded = _decode_key(key, key_type, where, state)
                except ReadError as error:
                    _invalid(key, str(error), where, state)
                    continue
            if (value_type is object or value_type is type(raw)) and type(
                raw,
            ) in _PLAIN_LEAVES:
                result[decoded] = raw
                continue
            value = _start_part(raw, value_type, where, state)
            if isinstance(value, _Frame):
                self.pending_raw, self.pending_at = raw, where
                self.pending_key = decoded
                return value
            result[decoded] = value
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        del state
        self.result[self.pending_key] = value

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.result


class _Fields(_PartFrame):
    """A record read field by field, each as its declared type, else ``object``."""

    __slots__ = ("members", "parts", "pending_name", "types")

    def __init__(
        self,
        at: FieldPath,
        *,
        data: Mapping[str, PlainTree],
        types: Mapping[str, object] | None,
        members: dict[str, object],
    ) -> None:
        super().__init__(at)
        self.parts = iter(data.items())
        self.types = types
        self.members = members
        self.pending_name = ""

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        at, types, members = self.at, self.types, self.members
        for name, raw in self.parts:
            if types is None:
                if name == "py/object":
                    continue
                target: object = object
            elif name in types:
                target = types[name]
            else:
                continue
            if (target is object or target is type(raw)) and type(raw) in _PLAIN_LEAVES:
                members[name] = raw
                continue
            where = (*at, name)
            value = _start_part(raw, target, where, state)
            if isinstance(value, _Frame):
                self.pending_raw, self.pending_at = raw, where
                self.pending_name = name
                return value
            members[name] = value
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        del state
        self.members[self.pending_name] = value

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.members


class _Object(_Fields):
    """A dataclass record, built from its fields once they have all read."""

    __slots__ = ("before", "index", "stored", "target", "value")

    def __init__(
        self,
        at: FieldPath,
        *,
        data: Mapping[str, PlainTree],
        types: Mapping[str, object],
        target: type[DataclassInstance],
        stored: bool,
        value: object,
        index: int,
        before: int,
    ) -> None:
        super().__init__(at, data=data, types=types, members={})
        self.target = target
        self.stored = stored
        self.value = value
        self.index = index
        self.before = before

    @override
    def finish(self, state: _Decoding) -> object:
        target, members, value = self.target, self.members, self.value
        name = target.__name__
        if self.stored or len(state.bad) > self.before:
            if value is PENDING:
                value = _allocate(target)
            rest = _with_defaults(fields(target), members, name)
        else:
            init = {item.name for item in fields(target) if item.init}
            try:
                value = target(**{k: v for k, v in members.items() if k in init})
            except (TypeError, ValueError) as error:
                raise ReadError(f"cannot build {name}: {error}") from error
            rest = {k: v for k, v in members.items() if k not in init}
        for key, member in rest.items():
            object.__setattr__(value, key, member)
        _replace(self.index, value, self.at, state)
        return value


class _Stored(_Fields):
    """A ``py/object`` record's attributes, set on its allocated object once read."""

    __slots__ = ("value",)

    def __init__(
        self,
        at: FieldPath,
        *,
        data: Mapping[str, PlainTree],
        value: object,
    ) -> None:
        super().__init__(at, data=data, types=None, members={})
        self.value = value

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        for attribute, member in self.members.items():
            object.__setattr__(self.value, attribute, member)
        return self.value


class _Steps(_Frame):
    """A fixed list of parts whose failure is the container's own."""

    __slots__ = ("done", "jobs", "values")

    def __init__(
        self,
        at: FieldPath,
        *,
        jobs: Sequence[tuple[PlainTree, object, FieldPath]],
        done: Callable[[list[object]], object],
    ) -> None:
        super().__init__(at)
        self.jobs = iter(jobs)
        self.done = done
        self.values: list[object] = []

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        for raw, target, where in self.jobs:
            value = _start(raw, target, where, state)
            if isinstance(value, _Frame):
                self.pending_raw, self.pending_at = raw, where
                return value
            self.values.append(value)
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        del state
        self.values.append(value)

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.done(self.values)


class _Reduce(_Frame):
    """A ``py/reduce`` recipe: its callable and arguments, the call, then its state."""

    __slots__ = ("data", "func", "index", "part", "stateful", "target", "value")

    def __init__(
        self,
        at: FieldPath,
        *,
        data: Sequence[PlainTree],
        target: object,
        stateful: bool,
        index: int,
    ) -> None:
        super().__init__(at)
        self.data = data
        self.target = target
        self.stateful = stateful
        self.index = index
        self.part = 0
        self.func: object = None
        self.value: object = None

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        data, at, target = self.data, self.at, self.target
        while self.part < len(data):
            part = self.part
            raw = data[part]
            if (
                part == 0
                and isinstance(target, type)
                and raw == {"py/type": _import_name(target)}
            ):
                self.take(target, state)
                continue
            if part >= 2 and raw is None:
                self.part += 1
                continue
            where = (*at, part)
            value = _start(raw, object, where, state)
            if isinstance(value, _Frame):
                self.pending_raw, self.pending_at = raw, where
                return value
            self.take(value, state)
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        part = self.part
        self.part = part + 1
        if part == 0:
            self.func = value
        elif part == 1:
            self._call(value, state)
        elif part == 2:
            _apply_state(self.value, value)
        elif part == 3:
            self._extend(value)
        else:
            self._set_items(value)

    @override
    def finish(self, state: _Decoding) -> object:
        del state
        return self.value

    def _call(self, args: object, state: _Decoding) -> None:
        """Call the recipe's callable on ``args`` and place the value it builds."""
        func = self.func
        if not callable(func) or not isinstance(args, tuple):
            raise ReadError("py/reduce needs a callable and an argument tuple")
        try:
            value = func(*cast("tuple[object, ...]", args))
        except (TypeError, ValueError) as error:
            raise ReadError(f"py/reduce call failed: {error}") from error
        if self.stateful:
            _replace(self.index, value, self.at, state)
        elif is_mutable(value):
            # ``to_plain`` names a mutable stateless value by path only; it wrote no
            # ``py/id`` for one, so no encounter number is reserved for it.
            state.placed[self.at] = value
        self.value = value

    def _extend(self, items: object) -> None:
        """Extend the built value with the recipe's list items."""
        extend: object = getattr(self.value, "extend", None)
        if not callable(extend) or not isinstance(items, list):
            raise ReadError(
                f"reduce target cannot take list items: {_shown(self.value)}",
            )
        extend(items)

    def _set_items(self, pairs: object) -> None:
        """Set the recipe's dict items on the built value."""
        setitem: object = getattr(self.value, "__setitem__", None)
        if not callable(setitem) or not isinstance(pairs, list):
            raise ReadError(
                f"reduce target cannot take dict items: {_shown(self.value)}",
            )
        for pair in cast("list[object]", pairs):
            if not isinstance(pair, tuple):
                raise ReadError(f"invalid reduce dict item: {_shown(pair)}")
            entry = cast("tuple[object, ...]", pair)
            if len(entry) != 2:
                raise ReadError(f"invalid reduce dict item: {_shown(entry)}")
            key, member = entry
            setitem(key, member)


class _Union(_Frame):
    """A union read member by member until one reads with no bad part."""

    __slots__ = ("data", "key", "marks", "members", "references", "target", "value")

    def __init__(
        self,
        at: FieldPath,
        *,
        data: PlainTree,
        target: object,
        members: tuple[object, ...],
        key: tuple[int, int],
        references: int,
    ) -> None:
        super().__init__(at)
        self.data = data
        self.target = target
        self.members = iter(members)
        self.key = key
        self.references = references
        self.marks = (0, 0, 0)
        self.value: object = PENDING

    @override
    def step(self, state: _Decoding) -> _Frame | None:
        for member in self.members:
            if self.value is not PENDING:
                return None
            self.marks = (len(state.built), len(state.bad), len(state.placed))
            try:
                value = _start(self.data, member, self.at, state)
            except ReadError:
                self._rollback(state)
                continue
            if isinstance(value, _Frame):
                return value
            self.take(value, state)
        return None

    @override
    def take(self, value: object, state: _Decoding) -> None:
        if len(state.bad) == self.marks[1]:
            self.value = value
            self.members = iter(())
        else:
            self._rollback(state)

    @override
    def fail(self, error: ReadError, state: _Decoding) -> bool:
        del error
        self._rollback(state)
        return True

    @override
    def finish(self, state: _Decoding) -> object:
        if self.value is not PENDING:
            return self.value
        if state.references == self.references:
            state.unread[self.key] = self.data
        raise ReadError(f"cannot read {_shown(self.data)} as {self.target}")

    def _rollback(self, state: _Decoding) -> None:
        """Forget everything the member last tried numbered, placed, or found bad."""
        built, bad, placed = self.marks
        del state.built[built:]
        while len(state.bad) > bad:
            state.bad.popitem()
        while len(state.placed) > placed:
            state.placed.popitem()


def _import_name(member: object) -> str | None:
    """Return a class member's ``module.qualname``, else ``None``."""
    if isinstance(member, type):
        return f"{member.__module__}.{member.__qualname__}"
    return None


def _union_candidates(
    data: PlainTree,
    members: tuple[object, ...],
) -> tuple[object, ...]:
    """Return the union members to try: the data's own kind first."""
    if data is None or isinstance(data, (str, int, float)):
        kind: type = type(data)
    elif isinstance(data, Mapping):
        kind = dict
    else:
        kind = list
    origins = _NATIVE_ORIGINS[kind]
    exact = tuple(
        member
        for member in members
        if (get_origin(_resolve_alias(member)) or _resolve_alias(member)) in origins
    )
    return exact + tuple(member for member in members if member not in exact)


def _element_type(target: object) -> object:
    """Return a one-parameter container's element type, else ``object``."""
    args = cast("tuple[object, ...]", get_args(target))
    return args[0] if len(args) == 1 else object


def _pair(data: PlainTree, tag: str) -> tuple[PlainTree, ...]:
    """Return a two-element tag payload's parts, naming the tag when malformed."""
    if not isinstance(data, Sequence) or isinstance(data, str) or len(data) != 2:
        raise ReadError(f"invalid {tag} payload: {_shown(data)}")
    return tuple(cast("Sequence[PlainTree]", data))


def _is_class(value: object) -> TypeGuard[type[object]]:
    """Return whether ``value`` is a class."""
    return isinstance(value, type)


def _allocate(kind: type) -> object:
    """Allocate ``kind`` without running ``__init__``."""
    allocate = cast("Callable[[type], object]", kind.__new__)
    return allocate(kind)


def _resolve(path: str, state: _Decoding) -> object:
    """Import the object ``path`` names, refusing unless imports are allowed."""
    if not state.allow_imports:
        raise ReadError(f"{path!r} needs allow_imports=True")
    resolved = _RESOLVED.get(path, PENDING)
    if resolved is PENDING or not resolves_to(path, resolved):
        try:
            resolved = resolve_import(path)
        except (AttributeError, ImportError) as error:
            raise ReadError(f"cannot import {path!r}: {error}") from error
        _RESOLVED[path] = resolved
    return resolved


def _resolve_alias(target: object) -> object:
    """Unwrap PEP 695 alias chains and ``Annotated`` to the underlying type."""
    resolved = target
    seen: set[int] = set()
    while (identity := id(resolved)) not in seen:
        seen.add(identity)
        if get_origin(resolved) is Annotated:
            resolved = cast("tuple[object, ...]", get_args(resolved))[0]
            continue
        value: object = getattr(resolved, "__value__", None)
        if value is None:
            break
        resolved = value
    return resolved


# A cached value naming its own class would keep the weak key alive forever, so
# a class is cached only if its module leads to it by qualname and no hint names
# it. That includes nested classes (such as configgle's ``Outer.Config``), since
# evaluating their hints again is most of a typed read. An unresolved hint is not
# cached either: its import may only be pending.
def _field_types(kind: type) -> Mapping[str, object]:
    """Return a dataclass's state types (fields, then slots) or a TypedDict's, cached."""
    cached = _FIELD_TYPES.get(kind)
    if cached is not None:
        return cached
    owner: type = kind
    reached = _module_reaches(kind)
    plain = _plain_hints(kind)
    hints, resolved = (
        (plain, True)
        if plain is not None
        else _resolvable_hints(kind, include_extras=False)
    )
    names = (
        dict.fromkeys([item.name for item in fields(kind)] + list(slot_names(kind)))
        if is_dataclass(kind)
        else hints
    )
    result = MappingProxyType(
        {name: _resolved_hint(hints.get(name, object)) for name in names},
    )
    if (
        resolved
        and reached
        and not any(_names(hint, owner) for hint in result.values())
    ):
        _FIELD_TYPES[kind] = result
    return result


def _names(hint: object, kind: object) -> bool:
    """Return whether ``hint`` is ``kind`` or holds it as an argument."""
    if hint is kind:
        return True
    # A plain class has no arguments; skip asking ``get_args`` for them.
    if type(hint) is type:
        return False
    return kind in cast("tuple[object, ...]", get_args(hint))


# ``get_type_hints`` evaluates each annotation and strips ``Annotated``; for a
# class whose annotations are all plain classes, ``X | Y`` of them, or builtin
# generics of them, each already is its evaluated, stripped self. Reading them
# straight off each base is about three times cheaper, and any other
# annotation leaves the full evaluation to ``get_type_hints``.
def _plain_hints(kind: type) -> dict[str, object] | None:
    """Return ``kind``'s hints as ``get_type_hints`` would, if all are plain, else ``None``."""
    if getattr(kind, "__no_type_check__", None):
        return None
    hints: dict[str, object] = {}
    for base in reversed(kind.__mro__):
        # A deferred annotation naming what does not exist yet raises here;
        # ``get_type_hints`` decides what that means.
        try:
            annotations = cast("dict[str, object]", inspect.get_annotations(base))
        except (AttributeError, NameError, TypeError):
            return None
        for name, hint in annotations.items():
            if not _is_plain_hint(hint):
                return None
            hints[name] = _rebuilt_hint(hint)
    return hints


# ``get_type_hints`` hands back a builtin generic as a new alias over the same
# arguments, and any other plain hint as itself, ``None`` as its class.
def _rebuilt_hint(hint: object) -> object:
    """Return ``hint`` as ``get_type_hints`` hands it back."""
    if hint is None:
        return type(None)
    if isinstance(hint, GenericAlias) and type(hint) is GenericAlias:
        origin = cast("type", hint.__origin__)
        return GenericAlias(origin, hint.__args__)
    return hint


def _is_plain_hint(hint: object) -> bool:
    """Return whether ``hint`` is its own evaluated, ``Annotated``-free form."""
    # ``None`` reads as its class; ``...`` closes a variadic tuple's arguments.
    if hint is None or hint is Ellipsis:
        return True
    if type(hint) is type:
        return True
    if isinstance(hint, UnionType):
        return all(map(_is_plain_hint, cast("tuple[object, ...]", hint.__args__)))
    if isinstance(hint, GenericAlias) and type(hint) is GenericAlias:
        return not hint.__unpacked__ and all(
            map(_is_plain_hint, cast("tuple[object, ...]", hint.__args__)),
        )
    return False


def _resolved_hint(hint: object) -> object:
    """``_resolve_alias``, skipped for a plain class, which is no alias."""
    return hint if type(hint) is type else _resolve_alias(hint)


def _module_reaches(kind: type) -> bool:
    """Return whether following ``kind``'s qualname from its module finds ``kind``."""
    module = sys.modules.get(kind.__module__)
    if module is None:
        return False
    first, *rest = kind.__qualname__.split(".")
    found: object = vars(module).get(first)
    # Only classes are followed, so a class defined in a function, such as
    # ``f.<locals>.C``, is not found.
    for name in rest:
        if not isinstance(found, type):
            return False
        found = vars(found).get(name)
    return found is kind


# A hint naming what its module imports only for type checking does not resolve at
# run time. Its field reads by its tags instead, so a value still reads back as the
# type it was written from.
#
# Such a class is not cached, since the import may only be pending. But later
# reads retry only the hints that failed. The ones that resolved are kept, under
# the rules ``_field_types`` uses to cache a class, and once a failed hint
# resolves, the class is read in full again.
def _resolvable_hints(
    kind: type,
    *,
    include_extras: bool,
) -> tuple[dict[str, object], bool]:
    """Return ``kind``'s hints, ``object`` for each that does not resolve, and if all did."""
    known = _PARTIAL_HINTS.get(kind, {}).get(include_extras)
    if known is not None:
        kept, retry = known
        if not any(
            _resolves(kind.__mro__[at], name, include_extras=include_extras)
            for name, at in retry.items()
        ):
            return dict(kept), False
        # Another thread may have dropped the entry since it was read.
        _PARTIAL_HINTS.pop(kind, None)
    try:
        return get_type_hints(kind, include_extras=include_extras), True
    except (AttributeError, NameError, TypeError):
        pass
    hints: dict[str, object] = {}
    failed: dict[str, int] = {}
    for at in reversed(range(len(kind.__mro__))):
        base = kind.__mro__[at]
        globalns, localns = _namespaces(base)
        for name, hint in _own_annotations(base).items():
            hints[name] = _single_hint(
                name,
                hint,
                globalns,
                localns,
                include_extras=include_extras,
            )
            failed.pop(name, None)
            if hints[name] is PENDING:
                hints[name] = object
                failed[name] = at
    if _module_reaches(kind) and not any(
        hint is kind or kind in cast("tuple[object, ...]", get_args(hint))
        for hint in hints.values()
    ):
        _PARTIAL_HINTS.setdefault(kind, {})[include_extras] = (dict(hints), failed)
    return hints, False


def _resolves(base: type, name: str, *, include_extras: bool) -> bool:
    """Return whether ``base``'s own annotation ``name`` resolves now, or is gone."""
    hint = _own_annotations(base).get(name, PENDING)
    if hint is PENDING:
        return True
    globalns, localns = _namespaces(base)
    resolved = _single_hint(
        name,
        hint,
        globalns,
        localns,
        include_extras=include_extras,
    )
    return resolved is not PENDING


def _namespaces(base: type) -> tuple[dict[str, object], dict[str, object]]:
    """Return the globals and locals ``base``'s annotations are evaluated in."""
    module = sys.modules.get(base.__module__)
    return (dict(vars(module)) if module is not None else {}), dict(vars(base))


def _single_hint(
    name: str,
    hint: object,
    globalns: dict[str, object],
    localns: dict[str, object],
    *,
    include_extras: bool,
) -> object:
    """Resolve one annotation by itself, or return ``PENDING`` if it fails."""
    single = SimpleNamespace(__annotations__={name: hint})
    try:
        return get_type_hints(
            single,
            globalns,
            localns,
            include_extras=include_extras,
        )[name]
    except (AttributeError, NameError, TypeError):
        return PENDING


def _own_annotations(base: type) -> dict[str, object]:
    """Return the annotations ``base`` itself declares, unevaluated where possible."""
    if sys.version_info >= (3, 14):
        return dict(  # pyright: ignore[reportUnreachable] -- The public checker targets 3.12; this branch runs on the monorepo's 3.14.
            inspect.get_annotations(
                base,
                format=importlib.import_module("annotationlib").Format.FORWARDREF,
            ),
        )
    return dict(inspect.get_annotations(base))


class _Shown(reprlib.Repr):
    """A ``repr`` cut short past a depth, keeping dict and set members in order."""

    @override
    def repr_dict(self, x: dict[object, object], level: int) -> str:
        if not x:
            return "{}"
        if level <= 0:
            return "{" + self.fillvalue + "}"
        pieces = [
            f"{self.repr1(key, level - 1)}: {self.repr1(item, level - 1)}"
            for key, item in islice(x.items(), self.maxdict)
        ]
        if len(x) > self.maxdict:
            pieces.append(self.fillvalue)
        return "{" + ", ".join(pieces) + "}"

    @override
    def repr_set(self, x: set[object], level: int) -> str:
        if not x:
            return "set()"
        return self._repr_members(x, level, "{", "}")

    @override
    def repr_frozenset(self, x: frozenset[object], level: int) -> str:
        if not x:
            return "frozenset()"
        return self._repr_members(x, level, "frozenset({", "})")

    def _repr_members(
        self,
        x: AbstractSet[object],
        level: int,
        left: str,
        right: str,
    ) -> str:
        """Return a set's members in iteration order between ``left`` and ``right``."""
        if level <= 0:
            return left + self.fillvalue + right
        pieces = [self.repr1(member, level - 1) for member in islice(x, self.maxset)]
        if len(x) > self.maxset:
            pieces.append(self.fillvalue)
        return left + ", ".join(pieces) + right


def _shown(value: object) -> str:
    """Return ``value``'s repr for an error message, elided past a fixed depth."""
    return _SHOWN.repr(value)


_TAGS: Final = (
    "py/type",
    "py/function",
    "py/tuple",
    "py/set",
    "py/frozenset",
    "py/mappingproxy",
    "py/b64",
    "py/float",
    "py/complex",
    "py/path",
    "py/uuid",
    "py/datetime",
    "py/reduce",
    "py/hook",
    "py/object",
)
"""Every tag, in the precedence a node carrying several resolves by."""


_PLAIN_LEAVES: Final = frozenset[object](
    cast("tuple[type, ...]", get_args(_resolve_alias(Plain))),
)
"""The exact :data:`Plain` types, which read as ``object`` unchanged."""


_NON_FINITE_REPRS: Final = frozenset(("inf", "-inf", "nan"))
"""What ``repr`` writes for a non-finite float, the only ``py/float`` payloads."""


_LAX_BOOLS: Final[Mapping[str, int]] = MappingProxyType(
    {"true": 1, "false": 0, "1": 1, "0": 0},
)
"""Text a lax read takes as a bool, by its lowercased spelling."""


_UNION_ORIGINS: Final = (UnionType, typing.Union)  # pyright: ignore[reportDeprecated] -- A runtime origin, not an annotation.
"""A union's origin: ``typing.Union`` before 3.14 for ``Optional``/``Union``, else
``UnionType``; from 3.14 the two are one class."""


_ABSENT_REF: Final = object()
"""Marks a ``py/ref`` path that names no placed value."""


_UNCHECKED: Final = object()
"""Marks a frame whose built value needs no fit check."""


_SHOWN: Final = _Shown(
    maxlevel=8,
    maxtuple=100,
    maxlist=100,
    maxarray=100,
    maxdict=100,
    maxset=100,
    maxfrozenset=100,
    maxdeque=100,
    maxstring=10_000,
    maxlong=10_000,
    maxother=10_000,
)
"""An error message's repr: identical to ``repr`` for small values, bounded for deep."""


_NATIVE_ORIGINS: Final[Mapping[type, tuple[object, ...]]] = MappingProxyType(
    {
        type(None): (type(None),),
        bool: (bool,),
        int: (int,),
        float: (float,),
        str: (str,),
        dict: (dict, Mapping, MutableMapping),
        list: (list, Sequence, MutableSequence),
    },
)
"""The union-member origins each plain kind reads as without conversion."""


_DECODERS: Final[Mapping[object, _Decoder]] = MappingProxyType(
    {
        object: _decode_any,
        None: _decode_none,
        type(None): _decode_none,
        bool: _decode_bool,
        int: _decode_int,
        float: _decode_float,
        complex: _decode_complex,
        str: _decode_str,
        bytes: _decode_bytes,
        Path: _decode_path,
        UUID: _decode_uuid,
        datetime: _decode_datetime,
        Literal: _decode_literal,
        list: _decode_list,
        Sequence: _decode_list,
        MutableSequence: _decode_list,
        tuple: _decode_tuple,
        set: _decode_set,
        MutableSet: _decode_set,
        frozenset: _decode_frozenset,
        AbstractSet: _decode_frozenset,
        dict: _decode_dict,
        Mapping: _decode_dict,
        MutableMapping: _decode_dict,
        MappingProxyType: _decode_mappingproxy,
        type: _decode_type,
        UnionType: _decode_union,
        # Before 3.14, ``Optional``/``Union`` have this origin, not ``UnionType``.
        typing.Union: _decode_union,  # pyright: ignore[reportDeprecated] -- A runtime origin key, not an annotation.
    },
)
"""Decoders by origin; ``_decoder_for`` adds Enum, path, dataclass, and Protocol targets."""


_TAG_DECODERS: Final[Mapping[str, _Decoder]] = MappingProxyType(
    {
        "py/float": _decode_float_tag,
        "py/complex": _decode_complex,
        "py/b64": _decode_bytes,
        "py/path": _decode_path,
        "py/uuid": _decode_uuid,
        "py/datetime": _decode_datetime,
        "py/tuple": _decode_tuple,
        "py/set": _decode_set,
        "py/frozenset": _decode_frozenset,
        "py/mappingproxy": _decode_mappingproxy,
        "py/type": _decode_type,
        "py/function": _decode_function,
        "py/object": _decode_object,
        "py/hook": _decode_hooked,
        "py/reduce": _decode_reduce,
    },
)
"""The decoder each tag selects; the same decoder untagged data reaches by type."""


_RESOLVED: Final[dict[str, object]] = {}
"""What each path last resolved to, trusted only while the path still names it."""


_FIELD_TYPES: Final[weakref.WeakKeyDictionary[type, Mapping[str, object]]] = (
    weakref.WeakKeyDictionary()
)


_PARTIAL_HINTS: Final[
    weakref.WeakKeyDictionary[
        type,
        dict[bool, tuple[dict[str, object], dict[str, int]]],
    ]
] = weakref.WeakKeyDictionary()
"""For each class with failed hints, keyed by ``include_extras``: its hints, and for
each failed name, the MRO index of the base that declared it."""


_python_from_plain: Final = from_plain
"""The Python walker, used when the optional Rust core is unavailable."""


def _walker[F](python: F) -> F:
    """Return the Rust core's ``from_plain`` when it serves, else ``python``."""
    return native(
        python,
        name="from_plain",
        setup="setup_decode",
        module=globals(),
    )


from_plain = _walker(_python_from_plain)
