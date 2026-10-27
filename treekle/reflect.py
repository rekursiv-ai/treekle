"""Import paths, object state, and concrete types, shared by both walks."""

from __future__ import annotations

from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set as AbstractSet,
)
from types import MappingProxyType, MemberDescriptorType
from typing import TYPE_CHECKING, Final, cast, get_origin

import importlib
import sys
import weakref


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


__all__ = [
    "BASELINE_CONCRETE",
    "PENDING",
    "SKIPPED_ATTRIBUTES",
    "attribute_names",
    "class_tags",
    "concrete_overrides",
    "is_mutable",
    "resolve_import",
    "resolves_to",
    "slot_names",
]


SKIPPED_ATTRIBUTES: Final = frozenset(("__weakref__", "__dict__"))
"""Slot names that hold no object state."""


PENDING: Final = object()
"""Marks a value not built yet, or an attribute a walk did not find."""


BASELINE_CONCRETE: Final = MappingProxyType[type, type](
    {
        Mapping: dict,
        MutableMapping: dict,
        Sequence: list,
        MutableSequence: list,
        AbstractSet: frozenset,
        MutableSet: set,
    },
)
"""What untagged data under each abstract declared type reads as by default."""


_BUILDABLE: Final[tuple[type, ...]] = (
    dict,
    MappingProxyType,
    list,
    tuple,
    set,
    frozenset,
)
"""The classes a walk builds from an untagged mapping or sequence."""


def resolve_import(path: str) -> object:
    """Walk ``path`` from its longest loaded prefix, else its longest importable one.

    Args:
      path: A dotted ``module.attr`` path, e.g. ``pkg.mod.Cls.Config``.

    Returns:
      resolved: The object ``path`` names.

    Raises:
      AttributeError: An importable prefix lacks the named attribute.
      ImportError: No prefix of ``path`` imports.

    """
    parts = path.split(".")
    # A nested class such as ``pkg.mod.Cls.Config`` names no module at
    # ``pkg.mod.Cls``; trying to import it first costs a failed finder search
    # per class, so a loaded ancestor is walked before anything is imported.
    for split in range(len(parts) - 1, 0, -1):
        module = sys.modules.get(".".join(parts[:split]))
        if module is not None:
            resolved = _walk(module, parts[split:])
            if resolved is not PENDING:
                return resolved
            break
    for split in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        resolved = _walk(module, parts[split:])
        if resolved is PENDING:
            raise AttributeError(f"{path!r} names no attribute of {module.__name__!r}")
        return resolved
    raise ImportError(f"Cannot resolve path: {path!r}")


def resolves_to(path: str, value: object) -> bool:
    """Return whether ``path`` names ``value`` now, as ``resolve_import`` walks it.

    Imports nothing: a path ``resolve_import`` once resolved has its module
    loaded, so this walk from the longest loaded prefix is the walk it would
    take. A cache of resolved paths asks it on every hit, so a name rebound
    after the first read is seen.

    Args:
      path: A dotted ``module.attr`` path.
      value: The object ``path`` named when it was cached.

    Returns:
      holds: Whether ``path`` still names ``value`` itself.

    """
    # Perf: pure-Python _resolve and _import_path hits, timeit: 370 -> 160 ns.
    splits = _SPLITS.get(path)
    if splits is None:
        splits = _SPLITS[path] = _splits(path)
    modules = sys.modules
    for prefix, names in splits:
        module = modules.get(prefix)
        if module is not None:
            return _walk(module, names) is value
    return False


def class_tags[K](classes: Mapping[K, object]) -> dict[type, str]:
    """Return each class in ``classes`` by its short name, checking the mapping.

    Args:
      classes: Short names to the classes they name.

    Returns:
      tags: Each class to its name.

    Raises:
      TypeError: A name is not a ``str``, or a value is not a class.
      ValueError: One class has two names.

    """
    tags: dict[type, str] = {}
    for name, kind in classes.items():
        if type(name) is not str:
            raise TypeError(f"classes keys must be str, not {type(name).__name__}")
        if not isinstance(kind, type):
            raise TypeError(f"classes[{name!r}] is not a class")
        if kind in tags:
            raise ValueError(
                f"classes names one class twice: {tags[kind]!r} and {name!r}",
            )
        tags[kind] = name
    return tags


def concrete_overrides(concrete: Mapping[type, type]) -> dict[object, type]:
    """Return each entry of ``concrete`` that differs from the baseline, checked.

    Args:
      concrete: Abstract declared types to the class untagged data under them
        reads as; a ``typing`` alias names its ``collections.abc`` class.

    Returns:
      overrides: The entries a walk must apply over ``BASELINE_CONCRETE``.

    Raises:
      ValueError: A key is not a ``BASELINE_CONCRETE`` key, or its value is
        not a buildable subclass of it.

    """
    overrides: dict[object, type] = {}
    for key, chosen in concrete.items():
        declared = cast("type", get_origin(key) or key)
        baseline = BASELINE_CONCRETE.get(declared)
        if baseline is None:
            raise ValueError(
                "default_concrete_types keys must be one of "
                f"{', '.join(kind.__name__ for kind in BASELINE_CONCRETE)}, "
                f"not {key!r}",
            )
        choices = (
            baseline,
            *(
                kind
                for kind in _BUILDABLE
                if kind is not baseline and issubclass(kind, declared)
            ),
        )
        if chosen not in choices:
            raise ValueError(
                f"default_concrete_types[{declared.__name__}] must be one of "
                f"{', '.join(choice.__name__ for choice in choices)}, not {chosen!r}",
            )
        if chosen is not baseline:
            overrides[declared] = chosen
    return overrides


# A class that sets ``__hash__ = None`` (``bytearray``, ``array``) is mutable by
# Python's own convention, so two occurrences of one are one object, not two values.
def is_mutable(value: object) -> bool:
    """Return whether ``value``'s class opts out of hashing."""
    kind: type = type(value)
    hasher: object = kind.__hash__
    return hasher is None


def slot_names(kind: type) -> tuple[str, ...]:
    """Return ``kind``'s state slots in MRO order, cached.

    Args:
      kind: The class to inspect.

    Returns:
      names: Each slot once, ``__dict__`` and ``__weakref__`` excluded.

    """
    cached = _SLOT_NAMES.get(kind)
    if cached is not None:
        return cached
    names: dict[str, None] = {}
    for base in kind.__mro__:
        slots = cast("object", vars(base).get("__slots__", ()))
        declared = (slots,) if isinstance(slots, str) else cast("Iterable[str]", slots)
        prefix = _mangling_prefix(base, declared)
        for slot in declared:
            if slot not in SKIPPED_ATTRIBUTES:
                names.setdefault(_mangled(slot, prefix))
    result = tuple(names)
    _SLOT_NAMES[kind] = result
    return result


# Sorted so moving a field between base classes leaves ``to_plain`` bytes unchanged.
def attribute_names(value: object, *, sort: bool = True) -> tuple[str, ...]:
    """Return the names of ``value``'s state: its slots and ``__dict__`` keys.

    Args:
      value: The object whose state to name.
      sort: Order names alphabetically; when false, slots in MRO order then
        ``__dict__`` keys in insertion order.

    Returns:
      names: Each name once; an unset slot is still named. A class with no
        instance dict contributes no ``__dict__`` keys.

    """
    kind = type(value)
    slots = slot_names(kind)
    if sort:
        slots = _SORTED_SLOT_NAMES.get(kind) or _sorted_slots(kind)
    if _dictless(kind):
        return slots
    state = getattr(value, "__dict__", None)
    if not state:
        return slots
    extra = [key for key in cast("dict[str, object]", state) if key not in slots]
    if not extra:
        return slots
    return tuple(sorted((*slots, *extra))) if sort else (*slots, *extra)


# A class with no dict storage and no ``__dict__`` in its MRO holds no
# ``__dict__`` state, whatever a ``__getattr__`` would answer for it. Read
# through ``type``'s own descriptors: a metaclass may answer ``__dictoffset__``
# or ``__mro__`` as it likes, and the instance dict is there all the same.
def _dictless(kind: type) -> bool:
    """Return whether ``kind``'s instances hold no ``__dict__`` state."""
    return _DICT_OFFSET(kind) == 0 and not any(
        "__dict__" in _OWN_ATTRIBUTES(base) for base in _MRO(kind)
    )


def _sorted_slots(kind: type) -> tuple[str, ...]:
    """Return and cache ``kind``'s slot names in alphabetical order."""
    result = tuple(sorted(slot_names(kind)))
    _SORTED_SLOT_NAMES[kind] = result
    return result


# ``__slots__`` lists a private name as written, but CPython stores the slot,
# as it compiles every access, under ``_<Class>__name``; read unmangled, the
# slot looks unset and its state is silently dropped. ``<Class>`` is the name
# ``type`` was given, which a later ``__name__`` change does not follow.
def _mangling_prefix(owner: type, declared: Iterable[str]) -> str:
    """Return the ``_<Class>`` CPython put before ``owner``'s private slots."""
    namespace = cast("Mapping[str, object]", vars(owner))
    names = frozenset(declared)
    for slot in names:
        if _is_private(slot):
            for key, entry in namespace.items():
                # A declared name is stored as written, so it is no mangling.
                if (
                    key.endswith(slot)
                    and key not in names
                    and isinstance(entry, MemberDescriptorType)
                    and entry.__objclass__ is owner
                ):
                    prefix = key.removesuffix(slot)
                    if prefix.startswith("_") and not prefix.startswith("__"):
                        return prefix
    # Every private slot's descriptor is gone; the current name is all there is.
    stripped = owner.__name__.lstrip("_")
    return f"_{stripped}" if stripped else ""


def _mangled(name: str, prefix: str) -> str:
    """Return slot ``name`` as CPython named it under mangling ``prefix``."""
    return f"{prefix}{name}" if _is_private(name) else name


def _is_private(name: str) -> bool:
    """Return whether CPython mangles ``name`` inside a class body."""
    return name.startswith("__") and not name.endswith("__")


def _walk(start: object, parts: Iterable[str]) -> object:
    """Return the attribute chain ``parts`` names from ``start``, else ``PENDING``."""
    resolved = start
    for part in parts:
        resolved = getattr(resolved, part, PENDING)
        if resolved is PENDING:
            break
    return resolved


_DICT_OFFSET: Final = cast(
    "Callable[[type], int]",
    vars(type)["__dictoffset__"].__get__,
)
"""``type.__dictoffset__`` of a class, read past its metaclass."""


_MRO: Final = cast("Callable[[type], tuple[type, ...]]", vars(type)["__mro__"].__get__)
"""``type.__mro__`` of a class, read past its metaclass."""


_OWN_ATTRIBUTES: Final = cast(
    "Callable[[type], Mapping[str, object]]",
    vars(type)["__dict__"].__get__,
)
"""``vars(base)`` of a class, read past its metaclass."""


_SPLITS: Final[dict[str, tuple[tuple[str, tuple[str, ...]], ...]]] = {}
"""``_splits`` of each path ``resolves_to`` was asked about.

One entry per distinct path asked, as the import-path caches that ask keep."""


_SLOT_NAMES: Final[weakref.WeakKeyDictionary[type, tuple[str, ...]]] = (
    weakref.WeakKeyDictionary()
)


_SORTED_SLOT_NAMES: Final[weakref.WeakKeyDictionary[type, tuple[str, ...]]] = (
    weakref.WeakKeyDictionary()
)


def _splits(path: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return each module prefix of ``path``, longest first, with the names after it."""
    parts = path.split(".")
    return tuple(
        (".".join(parts[:split]), tuple(parts[split:]))
        for split in range(len(parts) - 1, 0, -1)
    )
