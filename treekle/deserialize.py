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
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
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
)
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import base64
import binascii
import inspect
import json
import math
import sys
import typing
import weakref

from treekle.absent import ABSENT
from treekle.reflect import PENDING, resolve_import, slot_names
from treekle.types import (
    FieldPath,
    Hooks,
    Invalid,
    Plain,
    PlainTree,
    ReadError,
)


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

    """

    hooks: Hooks
    allow_imports: bool
    strict: bool = True
    untagged: bool = False
    built: list[object] = field(default_factory=list[object])
    placed: dict[FieldPath, object] = field(default_factory=dict[FieldPath, object])
    bad: dict[FieldPath, Invalid] = field(default_factory=dict[FieldPath, Invalid])


type _Decoder = Callable[[PlainTree, object, FieldPath, _Decoding], object]
"""Decode one node, untagged or a tag's payload, as ``target``."""


def _decode(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode one node: a reference, else its tag's decoder, else ``target``'s."""
    # Every node under a config reads as ``object``, so this path is the hot
    # one: a leaf passes through and a tag picks its decoder with no alias,
    # dispatch, or fit check, each of which is the identity for ``object``.
    if target is object:
        kind = type(data)
        if kind in _PLAIN_LEAVES:
            return data
        if kind is dict and not state.untagged:
            return _decode_tagged(cast("dict[str, PlainTree]", data), at, state)
    target = _resolve_alias(target)
    if not _is_plain(data):
        if _fits(data, target):
            return data
        # ``json.loads(parse_float=Decimal)`` and similar readers yield numbers
        # that are not ``float``; ``custom_json.convert`` read them as floats.
        if target is float and isinstance(data, (Real, Decimal)):
            return float(data)
        raise ReadError(f"cannot read {data!r} as {target}")
    decoder = _decoder_for(target)
    if decoder is _decode_union or not isinstance(data, Mapping):
        return decoder(data, target, at, state)
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
            raise ReadError(f"invalid {tag} envelope: {dict(data)!r}")
        payload = data if tag == "py/object" else data[tag]
        value = _TAG_DECODERS[tag](payload, target, at, state)
    if not _fits(value, target):
        raise ReadError(f"cannot read {value!r} as {target}")
    return value


def _decode_tagged(
    data: dict[str, PlainTree],
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a dict read as ``object``: by its tag, else as a plain dict."""
    if "py/object" in data:
        if not state.allow_imports:
            return _decode_dict(data, object, at, state)
        return _decode_object(data, object, at, state)
    if len(data) == 1:
        tag = next(iter(data))
        if tag == "py/ref":
            return _decode_path_reference(data, state)
        if tag == "py/id":
            return _decode_reference(data, state)
        decoder = _TAG_DECODERS.get(tag)
        if decoder is not None:
            return decoder(data[tag], object, at, state)
    tag = _tag_of(data)
    if tag is None:
        return _decode_dict(data, object, at, state)
    if "py/ref" in data:
        return _decode_path_reference(data, state)
    if "py/id" in data:
        return _decode_reference(data, state)
    raise ReadError(f"invalid {tag} envelope: {data!r}")


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
    if is_dataclass(origin):
        return _decode_object
    if is_typeddict(origin):
        return _decode_typeddict
    if (
        origin.__dict__.get("_is_protocol") is True
        or inspect.isabstract(origin)
        or (origin.__module__ == "typing" and origin.__qualname__ == "Any")
    ):
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
    if isinstance(data, Mapping):
        node = cast("Mapping[object, object]", data)
        return any(
            isinstance(key, str) and key.startswith(("py/", "json://")) for key in node
        ) or any(_holds_tags(item) for item in node.values())
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return any(_holds_tags(item) for item in data)
    return False


def _has_tag_key(keys: Iterable[object]) -> bool:
    """Return whether any key reads as a tag; a YAML reader may yield non-str keys."""
    return any(isinstance(key, str) and key.startswith("py/") for key in keys)


def _fits(value: object, target: object) -> bool:
    """Return whether a tag-decoded ``value`` is acceptable as ``target``."""
    target = _resolve_alias(target)
    origin: object = get_origin(target) or target
    if origin is UnionType or origin is typing.Union:  # pyright: ignore[reportDeprecated] -- A runtime origin test for pre-3.14 ``Optional``/``Union``, not an annotation.
        return any(
            _fits(value, member)
            for member in cast("tuple[object, ...]", get_args(target))
        )
    if origin is Literal:
        return value in cast("tuple[object, ...]", get_args(target))
    if not isinstance(origin, type):
        return True
    try:
        return isinstance(value, origin)
    except TypeError:
        return True


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
        raise ReadError(f"invalid py/id reference: {dict(node)!r}")
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
        raise ReadError(f"invalid py/ref reference: {dict(node)!r}")
    value = state.placed.get(tuple(cast("Sequence[str | int]", path)), _ABSENT_REF)
    if value is _ABSENT_REF:
        raise ReadError(f"py/ref {list(path)!r} names no earlier value")
    if value is PENDING:
        raise ReadError(f"py/ref {list(path)!r} names a value still being built")
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
    value: PlainTree,
    reason: str,
    at: FieldPath,
    state: _Decoding,
) -> Invalid:
    """Record ``value`` as bad at ``at`` and return its placeholder."""
    invalid = Invalid(raw=value, reason=reason)
    state.bad[at] = invalid
    return invalid


def _decode_part(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a container's part, recording an :class:`Invalid` if it fails."""
    try:
        return _decode(data, target, at, state)
    except ReadError as error:
        return _invalid(data, str(error), at, state)


def _decode_none(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> None:
    """Decode ``None``."""
    del target, at, state
    if data is not None:
        raise ReadError(f"cannot read {data!r} as None")


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
    raise ReadError(f"cannot read {data!r} as bool")


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
        number = _lax_number(data)
        if number is not None and number.is_integer():
            return int(number)
    raise ReadError(f"cannot read {data!r} as int")


def _lax_number(data: PlainTree) -> float | None:
    """Return the finite number ``data`` is or spells, else ``None``."""
    if isinstance(data, str):
        try:
            data = float(data)
        except ValueError:
            return None
    if isinstance(data, (int, float)) and math.isfinite(data):
        return float(data)
    return None


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
    raise ReadError(f"cannot read {data!r} as str")


def _decode_float(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> float:
    """Decode a float from an int, a float, or a ``py/float`` payload."""
    del target, at
    if isinstance(data, float):
        return data
    if isinstance(data, int) and not isinstance(data, bool):
        return float(data)
    if isinstance(data, str):
        try:
            number = float(data.strip())
        except ValueError:
            number = None
        if number is not None and (not math.isfinite(number) or not state.strict):
            return number
    raise ReadError(f"cannot read {data!r} as float")


def _decode_complex(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> complex:
    """Decode a complex: a real number, or a ``py/complex`` pair."""
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return complex(data)
    if isinstance(data, Sequence) and not isinstance(data, str) and len(data) == 2:
        return complex(
            _decode_float(data[0], float, (*at, 0), state),
            _decode_float(data[1], float, (*at, 1), state),
        )
    del target
    raise ReadError(f"cannot read {data!r} as complex")


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
            raise ReadError(f"cannot read {data!r} as bytes: {error}") from error
    raise ReadError(f"cannot read {data!r} as bytes")


def _decode_path(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Path:
    """Decode a ``Path`` from text."""
    del target, at, state
    if isinstance(data, str):
        return Path(data)
    raise ReadError(f"cannot read {data!r} as Path")


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
            raise ReadError(f"cannot read {data!r} as UUID") from error
    raise ReadError(f"cannot read {data!r} as UUID")


def _decode_datetime(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> datetime:
    """Decode an ISO 8601 ``datetime``, restoring a bracketed zone name."""
    del target, at, state
    if not isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as datetime")
    text, bracket, zone = data.partition("[")
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as error:
        raise ReadError(f"cannot read {data!r} as datetime") from error
    if not bracket:
        return moment
    if moment.utcoffset() is None:
        raise ReadError(f"cannot read {data!r} as datetime: timestamp is naive")
    if not zone.endswith("]") or len(zone) == 1:
        raise ReadError(f"cannot read {data!r} as datetime: invalid named zone")
    try:
        named = ZoneInfo(zone.removesuffix("]"))
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise ReadError(
            f"cannot read {data!r} as datetime: invalid named zone",
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
    if (
        isinstance(target, type)
        and issubclass(target, Enum)
        and isinstance(data, (str, int, float))
        and not isinstance(data, bool)
    ):
        try:
            return target(data)
        except ValueError:
            pass
    raise ReadError(f"cannot read {data!r} as {target}")


def _decode_literal(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> Plain | Enum:
    """Decode one of a ``Literal``'s values, matching bool and int apart."""
    for option in cast("tuple[object, ...]", get_args(target)):
        if isinstance(option, Enum):
            try:
                member = _decode_enum(data, type(option), at, state)
            except ReadError:
                continue
            if member is option:
                return option
        elif (
            (option is None or isinstance(option, (str, int, float)))
            and type(option) is type(data)
            and option == data
        ):
            return option
    raise ReadError(f"cannot read {data!r} as {target}")


def _decode_list(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> list[object]:
    """Decode a list, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    result: list[object] = []
    _place(result, at, state)
    result.extend(
        _decode_part(raw, item, (*at, index), state) for index, raw in enumerate(data)
    )
    return result


def _decode_tuple(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> tuple[object, ...]:
    """Decode a fixed or variadic tuple; tuples take no ``py/id`` number."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    if get_origin(target) is not tuple:
        types: tuple[object, ...] = (_element_type(target),) * len(data)
    elif len(args) == 2 and args[1] is Ellipsis:
        types = (args[0],) * len(data)
    elif len(args) != len(data):
        raise ReadError(f"expected {len(args)} items for {target}, got {len(data)}")
    else:
        types = args
    return tuple(
        _decode_part(raw, kind, (*at, index), state)
        for index, (raw, kind) in enumerate(zip(data, types, strict=True))
    )


def _decode_set(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> set[object]:
    """Decode a set, numbering it for ``py/id``."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    result: set[object] = set()
    _place(result, at, state)
    for index, raw in enumerate(data):
        member = _decode_part(raw, item, (*at, index), state)
        try:
            result.add(member)
        except TypeError as error:
            raise ReadError(f"set member {member!r} is unhashable") from error
    return result


def _decode_frozenset(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> frozenset[object]:
    """Decode a frozenset; frozensets take no ``py/id`` number."""
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as {target}")
    item = _element_type(target)
    members = [
        _decode_part(raw, item, (*at, index), state) for index, raw in enumerate(data)
    ]
    try:
        return frozenset(members)
    except TypeError as error:
        raise ReadError(f"frozenset members {members!r} are unhashable") from error


def _decode_dict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> dict[Hashable, object]:
    """Decode a dict, typing keys and values, numbering it for ``py/id``."""
    if not isinstance(data, Mapping):
        raise ReadError(f"cannot read {data!r} as {target}")
    args = cast("tuple[object, ...]", get_args(target))
    key_type, value_type = args if len(args) == 2 else (object, object)
    result: dict[Hashable, object] = {}
    _place(result, at, state)
    for key, raw in data.items():
        where = (*at, key)
        try:
            decoded = _decode_key(key, key_type, where, state)
        except ReadError as error:
            _invalid(key, str(error), where, state)
            continue
        result[decoded] = _decode_part(raw, value_type, where, state)
    return result


def _decode_mappingproxy(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> MappingProxyType[Hashable, object]:
    """Decode a ``MappingProxyType`` around a decoded dict."""
    return MappingProxyType(_decode_dict(data, target, at, state))


def _bad_key(key: Hashable, target: object) -> Hashable:
    """Raise for a non-str key that does not fit ``target``."""
    raise ReadError(f"cannot read key {key!r} as {target}")


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
            raise ReadError(f"invalid escaped key {key!r}") from error
    decoded = _decode(raw, target, at, state)
    if not isinstance(decoded, Hashable):
        raise ReadError(f"key {decoded!r} is unhashable")
    return decoded


def _decode_type(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode a ``py/type`` path to the class or module global it names."""
    del at
    if not isinstance(data, str):
        raise ReadError(f"cannot read {data!r} as a class path")
    resolved = _resolve(data, state)
    if (get_origin(target) or target) is type and not isinstance(resolved, type):
        raise ReadError(f"{data!r} does not name a class")
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
        raise ReadError(f"cannot read {data!r} as a function path")
    resolved = _resolve(data, state)
    if not callable(resolved):
        raise ReadError(f"{data!r} does not name a callable")
    return resolved


def _decode_typeddict(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> dict[str, object]:
    """Decode a ``TypedDict``: declared keys by their types, unknown ones dropped."""
    kind = cast("type", target)
    if not isinstance(data, Mapping):
        raise ReadError(f"expected object for {kind.__name__}, got {data!r}")
    hints = _field_types(kind)
    missing = sorted(_required_keys(kind) - set(data))
    if missing:
        raise ReadError(f"{kind.__name__}: missing required field(s) {missing}")
    return {
        key: _decode_part(raw, hints[key], (*at, key), state)
        for key, raw in data.items()
        if key in hints
    }


# Under ``from __future__ import annotations`` a class body's ``NotRequired``
# is a string, so ``__required_keys__`` lists every key; the resolved hints
# carry the qualifier.
def _required_keys(kind: type) -> frozenset[str]:
    """Return a TypedDict's required keys, read from its resolved hints."""
    total = cast("bool", getattr(kind, "__total__", True))
    hints = get_type_hints(kind, include_extras=True)
    return frozenset(
        key
        for key, hint in hints.items()
        if get_origin(hint) is not NotRequired
        and (total or get_origin(hint) is Required)
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
        raise ReadError(f"expected object for {name}, got {data!r}")
    if _names_dataclass(data, target, state):
        fields_only: dict[str, PlainTree] = {
            key: raw for key, raw in data.items() if key != "py/object"
        }
        data = fields_only
    if "py/object" in data:
        path = data["py/object"]
        if not isinstance(path, str):
            raise ReadError(f"invalid py/object path: {path!r}")
        kind = _resolve(path, state)
        if not isinstance(kind, type):
            raise ReadError(f"{path!r} does not name a class")
        value = _allocate(kind)
        _place(value, at, state)
        for attribute, raw in data.items():
            if attribute != "py/object":
                member = _decode_part(raw, object, (*at, attribute), state)
                object.__setattr__(value, attribute, member)
        return value
    if not isinstance(target, type) or not is_dataclass(target):
        raise ReadError(f"cannot read {dict(data)!r} as {name}")
    types = _field_types(target)
    unknown = sorted(set(data) - set(types))
    if unknown:
        raise ReadError(
            f"{name}: unknown field(s) {unknown}; valid: {sorted(types)}",
        )
    index = len(state.built)
    _place(PENDING, at, state)
    before = len(state.bad)
    members = {
        key: _decode_part(raw, types[key], (*at, key), state)
        for key, raw in data.items()
    }
    # A part that did not read leaves ``__init__`` unrunnable, so every member is
    # set directly; otherwise ``__init__`` takes its fields and the rest of the
    # state, such as a non-init slot, is set after it.
    init = {item.name for item in fields(target) if item.init}
    if len(state.bad) > before:
        value, rest = _allocate(target), members
    else:
        try:
            value = target(**{k: v for k, v in members.items() if k in init})
        except (TypeError, ValueError) as error:
            raise ReadError(f"cannot build {name}: {error}") from error
        rest = {k: v for k, v in members.items() if k not in init}
    for key, member in rest.items():
        object.__setattr__(value, key, member)
    _replace(index, value, at, state)
    return value


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
        raise ReadError(f"hook {path!r} is not registered")
    # The encoder numbers the hooked value before its payload, so the slot is
    # reserved in that order and filled once the hook rebuilds it.
    index = len(state.built)
    _place(PENDING, at, state)
    value = hook[1](_decode(payload, object, (*at, 1), state))
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
        raise ReadError(f"invalid py/reduce payload: {data!r}")
    if len(data) < 2 or len(data) > 5:
        raise ReadError("py/reduce requires two to five elements")
    stateful = any(part is not None for part in data[2:])
    index = len(state.built)
    if stateful:
        _place(PENDING, at, state)
    func = (
        target
        if isinstance(target, type) and data[0] == {"py/type": _import_name(target)}
        else _decode(data[0], object, (*at, 0), state)
    )
    args = _decode(data[1], object, (*at, 1), state)
    if not callable(func) or not isinstance(args, tuple):
        raise ReadError("py/reduce needs a callable and an argument tuple")
    try:
        value = func(*cast("tuple[object, ...]", args))
    except (TypeError, ValueError) as error:
        raise ReadError(f"py/reduce call failed: {error}") from error
    if stateful:
        _replace(index, value, at, state)
    if len(data) > 2 and data[2] is not None:
        _apply_state(value, _decode(data[2], object, (*at, 2), state))
    if len(data) > 3 and data[3] is not None:
        items = _decode(data[3], object, (*at, 3), state)
        extend: object = getattr(value, "extend", None)
        if not callable(extend) or not isinstance(items, list):
            raise ReadError(f"reduce target cannot take list items: {value!r}")
        extend(items)
    if len(data) > 4 and data[4] is not None:
        pairs = _decode(data[4], object, (*at, 4), state)
        setitem: object = getattr(value, "__setitem__", None)
        if not callable(setitem) or not isinstance(pairs, list):
            raise ReadError(f"reduce target cannot take dict items: {value!r}")
        for pair in cast("list[object]", pairs):
            entry = cast("tuple[object, ...]", pair) if isinstance(pair, tuple) else ()
            if len(entry) != 2:
                raise ReadError(f"invalid reduce dict item: {pair!r}")
            key, member = entry
            setitem(key, member)
    return value


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
        return _decode_dict(data, object, at, state)
    return _decode_list(data, object, at, state)


def _decode_union(
    data: PlainTree,
    target: object,
    at: FieldPath,
    state: _Decoding,
) -> object:
    """Decode the member the data's kind selects, else the first that fits."""
    members = cast("tuple[object, ...]", get_args(target))
    tagged = isinstance(data, Mapping) and _has_tag_key(data)
    if tagged:
        # The member a ``py/object`` tag names goes first, so member order
        # cannot pick a sibling whose fields happen to fit too.
        path = cast("Mapping[str, PlainTree]", data).get("py/object")
        members = tuple(
            sorted(members, key=lambda member: _import_name(member) != path),
        )
    for member in members if tagged else _union_candidates(data, members):
        built, bad, placed = len(state.built), set(state.bad), set(state.placed)
        try:
            value = _decode(data, member, at, state)
        except ReadError:
            value = PENDING
        if value is not PENDING and len(state.bad) == len(bad):
            return value
        del state.built[built:]
        for key in set(state.bad) - bad:
            del state.bad[key]
        for key in set(state.placed) - placed:
            del state.placed[key]
    raise ReadError(f"cannot read {data!r} as {target}")


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
        native: type = type(data)
    elif isinstance(data, Mapping):
        native = dict
    else:
        native = list
    origins = _NATIVE_ORIGINS[native]
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
        raise ReadError(f"invalid {tag} payload: {data!r}")
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
    if resolved is PENDING:
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
# only module-level classes whose hints do not name themselves are cached.
def _field_types(kind: type) -> Mapping[str, object]:
    """Return a dataclass's state types (fields, then slots) or a TypedDict's, cached."""
    cached = _FIELD_TYPES.get(kind)
    if cached is not None:
        return cached
    module_name: str = kind.__module__
    qualname: str = kind.__qualname__
    try:
        hints: dict[str, object] = get_type_hints(kind)
    except (AttributeError, NameError, TypeError) as error:
        raise ReadError(f"cannot resolve field types of {kind}: {error}") from error
    names = (
        dict.fromkeys([item.name for item in fields(kind)] + list(slot_names(kind)))
        if is_dataclass(kind)
        else hints
    )
    result = MappingProxyType(
        {name: _resolve_alias(hints.get(name, object)) for name in names},
    )
    module = sys.modules.get(module_name)
    if (
        module is not None
        and vars(module).get(qualname) is kind
        and not any(
            hint is kind or kind in cast("tuple[object, ...]", get_args(hint))
            for hint in result.values()
        )
    ):
        _FIELD_TYPES[kind] = result
    return result


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


_PLAIN_LEAVES: Final = frozenset(
    cast("tuple[type, ...]", get_args(_resolve_alias(Plain))),
)
"""The exact :data:`Plain` types, which read as ``object`` unchanged."""


_LAX_BOOLS: Final[Mapping[str, int]] = MappingProxyType(
    {"true": 1, "false": 0, "1": 1, "0": 0},
)
"""Text a lax read takes as a bool, by its lowercased spelling."""


_ABSENT_REF: Final = object()
"""Marks a ``py/ref`` path that names no placed value."""


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
"""Decoders by origin; ``_decoder_for`` adds Enum, dataclass, and Protocol targets."""


_TAG_DECODERS: Final[Mapping[str, _Decoder]] = MappingProxyType(
    {
        "py/float": _decode_float,
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


_FIELD_TYPES: Final[weakref.WeakKeyDictionary[type, Mapping[str, object]]] = (
    weakref.WeakKeyDictionary()
)
