"""Tests that the Rust core reads classes as the pure-Python walk does.

Each case builds a class whose import path, slot layout, or table entry
takes a less common path through the native class reader, then compares
``to_plain`` with ``_python_to_plain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, override

import sys

import pytest

from treekle import reflect, serialize


if TYPE_CHECKING:
    from collections.abc import Callable


def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result, or the type and text of what it raised."""
    try:
        return ("ok", call())
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either walker raises.
        return ("raise", type(error).__name__, str(error))


def _assert_encodes_alike(value: object, *, sort: bool = True) -> None:
    # Native first: Python's walk fills ``_IMPORT_PATHS``, which native reads.
    actual = _outcome(lambda: serialize.to_plain(value, sort=sort))
    assert actual == _outcome(lambda: serialize._python_to_plain(value, sort=sort))


def _module(monkeypatch: pytest.MonkeyPatch, name: str) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Slotted(Protocol):
    """A slotted dataclass with one field, ``x``."""

    x: int

    def __init__(self, *, x: int) -> None: ...


def _slotted(module: ModuleType, qualname: str, **namespace: object) -> type[_Slotted]:
    """Return a slotted dataclass named ``qualname`` in ``module``."""
    name = qualname.rsplit(".", 1)[-1]
    body: dict[str, object] = {
        "__module__": module.__name__,
        "__qualname__": qualname,
        "__annotations__": {"x": int},
    }
    body.update(namespace)
    return cast("type[_Slotted]", dataclass(slots=True)(type(name, (), body)))


@pytest.mark.parametrize("sort", [True, False])
def test_a_nested_class_path_walks_the_outer_class(
    monkeypatch: pytest.MonkeyPatch,
    sort: bool,
) -> None:
    module = _module(monkeypatch, "treekle_class_nested")
    inner = _slotted(module, "Outer.Inner")
    outer = type("Outer", (), {"__module__": module.__name__, "Inner": inner})
    vars(module).update(Outer=outer)

    _assert_encodes_alike(inner(x=1), sort=sort)


@pytest.mark.parametrize(
    "case",
    [
        "outer is not a class",
        "inner is missing",
        "inner is not the class",
        "inner is not a class",
        "outer is missing",
        "dotted module prefix is loaded",
    ],
)
def test_a_nested_class_path_that_does_not_resolve_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    module = _module(monkeypatch, "treekle_class_broken")
    inner = _slotted(module, "Outer.Inner")
    members: dict[str, object] = {"__module__": module.__name__}
    if case == "inner is not the class":
        members["Inner"] = int
    elif case == "inner is not a class":
        members["Inner"] = 5
    elif case == "dotted module prefix is loaded":
        members["Inner"] = inner
        _module(monkeypatch, f"{module.__name__}.Outer")
    if case != "outer is missing":
        outer = 3 if case == "outer is not a class" else type("Outer", (), members)
        vars(module).update(Outer=outer)

    _assert_encodes_alike(inner(x=1))


def test_a_class_whose_module_is_not_loaded_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("treekle_class_unloaded")
    kind = _slotted(module, "Lost")
    vars(module).update(Lost=kind)
    monkeypatch.delitem(sys.modules, module.__name__, raising=False)

    _assert_encodes_alike(kind(x=1))


def test_a_class_whose_module_is_not_a_module_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Namespace(ModuleType):
        pass

    module = _Namespace("treekle_class_subclassed")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    kind = _slotted(module, "Kept")
    vars(module).update(Kept=kind)

    _assert_encodes_alike(kind(x=1))


@pytest.mark.parametrize("module_name", ["treekle_class_é", 7])
def test_a_class_with_an_unusual_module_name_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    module_name: object,
) -> None:
    if isinstance(module_name, str):
        module = _module(monkeypatch, module_name)
        kind = _slotted(module, "Named")
        vars(module).update(Named=kind)
    else:
        module = _module(monkeypatch, "treekle_class_int")
        kind = _slotted(module, "Named")
        type.__setattr__(kind, "__module__", module_name)

    # Twice: the second read takes the walk's remembered module.
    _assert_encodes_alike([kind(x=1), kind(x=2)])


def test_classes_after_one_another_in_a_module_read_their_own_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch, "treekle_class_memo")
    first = _slotted(module, "First")
    second = _slotted(module, "Second")
    renamed = _slotted(module, "Renamed")
    ghost = _slotted(module, "Ghost")
    third = _slotted(module, "Third")
    fourth = _slotted(module, "Fourth")
    vars(module).update(
        First=first,
        Second=second,
        Third=third,
        Fourth=fourth,
        Renamed=int,
    )

    # Each walk remembers the last module read; a class it cannot resolve
    # ends the walk, so each such class closes its own.
    _assert_encodes_alike([first(x=1), second(x=2), first(x=3)])
    # Classes no walk has kept, so each path is read: ``sys.modules``
    # changes between the two, and the remembered module is read again.
    _assert_encodes_alike([third(x=1), _Importing(), fourth(x=2)])
    _assert_encodes_alike([first(x=1), renamed(x=2)])
    _assert_encodes_alike([first(x=1), ghost(x=2)])


@pytest.mark.parametrize("sort", [True, False])
def test_a_weakref_slot_matches_python(sort: bool) -> None:
    @dataclass(kw_only=True, slots=True, weakref_slot=True)
    class Weak:
        b: int
        a: int

    _assert_encodes_alike([Weak(b=1, a=2), Weak(b=3, a=4)], sort=sort)


@pytest.mark.parametrize(
    "namespace",
    [
        {"__slots__": ("x", "y")},
        {"__slots__": ["x"]},
        {"extra": 1},
    ],
    ids=["slots tuple too long", "slots not a tuple", "entry after the slots"],
)
def test_a_slot_layout_that_differs_from_type_s_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    namespace: dict[str, object],
) -> None:
    module = _module(monkeypatch, "treekle_class_layout")
    kind = _slotted(module, "Odd")
    for name, value in namespace.items():
        type.__setattr__(kind, name, value)
    vars(module).update(Odd=kind)

    _assert_encodes_alike([kind(x=1), kind(x=2)])


def test_more_classes_than_the_table_holds_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each class's path is kept; past the table's limit the oldest give way.
    module = _module(monkeypatch, "treekle_class_many")
    kinds = [
        type(f"K{index}", (), {"__module__": module.__name__}) for index in range(4200)
    ]
    vars(module).update({kind.__name__: kind for kind in kinds})

    _assert_encodes_alike([*kinds, *kinds[:8]])


# ``__doc__`` is given, so ``type`` adds nothing after the slots' members.
def _plain_slotted(
    module: ModuleType,
    slots: tuple[str, ...],
    **after: object,
) -> type[object]:
    """Return a class ``type`` lays out from ``slots``, then ``after`` set on it."""
    kind = type(
        "Laid",
        (),
        {
            "__module__": module.__name__,
            "__doc__": None,
            "__dataclass_fields__": {},
            "junk": None,
            "__slots__": slots,
        },
    )
    for name, value in after.items():
        type.__setattr__(kind, name, value)
    vars(module).update(Laid=kind)
    return kind


def _filled(kind: type[object], **members: object) -> object:
    value = object.__new__(kind)
    for name, member in members.items():
        setattr(value, name, member)
    return value


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "weakref slot",
        "descriptor replaced in place",
        "member of another class",
        "dunder slot",
        "entry after the slots",
        "entry after the slots, past a hole",
        "entry after the weakref slot",
        "weakref slot moved",
    ],
)
def test_a_slot_layout_read_from_the_class_dict_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    sort: bool,
) -> None:
    module = _module(monkeypatch, "treekle_class_laid")
    slots = ("b", "a", "__weakref__") if case.startswith("weakref") else ("b", "a")
    if case == "dunder slot":
        slots = ("b", "__a__")
    kind = _plain_slotted(module, slots)
    members: dict[str, object] = {"b": 1, "__a__" if case == "dunder slot" else "a": 2}
    value = _filled(kind, **members)
    if case == "descriptor replaced in place":

        def nine(value: object) -> int:
            del value
            return 9

        type.__setattr__(kind, "a", property(nine))
    elif case == "member of another class":
        donor = _plain_slotted(_module(monkeypatch, "treekle_class_donor"), ("a",))
        type.__setattr__(kind, "a", vars(donor)["a"])
    elif case.startswith("entry after the"):
        if case.endswith("past a hole"):
            # A deleted entry leaves the dict's live entries past ``used``.
            type.__delattr__(kind, "junk")
        type.__setattr__(kind, "marker", None)
    elif case == "weakref slot moved":
        # Another entry where ``__weakref__`` stood, the slots kept in place.
        weak = cast("object", vars(kind)["__weakref__"])
        type.__delattr__(kind, "__weakref__")
        type.__setattr__(kind, "marker", weak)

    _assert_encodes_alike([value, value], sort=sort)


def test_a_rebound_sys_modules_matches_python(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("treekle_class_sysmodules")
    kind = _slotted(module, "Rebound")
    vars(module).update(Rebound=kind)
    modules = {**sys.modules, module.__name__: module}
    monkeypatch.setattr(sys, "modules", _Modules(modules))

    _assert_encodes_alike(kind(x=1))


class _Modules(dict[str, ModuleType]):
    """A ``sys.modules`` that is not an exact dict."""


def test_a_local_class_matches_python() -> None:
    @dataclass(kw_only=True, slots=True)
    class Local:
        x: int

    _assert_encodes_alike(Local(x=1))


@pytest.mark.parametrize(
    "names",
    [("x", 3), ("x", "y")],
    ids=["a name that is not a str", "a name with no member"],
)
def test_slot_names_planted_in_reflect_s_cache_match_python(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[object, ...],
) -> None:

    module = _module(monkeypatch, "treekle_class_planted")
    kind = _slotted(module, "Planted")
    vars(module).update(Planted=kind)
    monkeypatch.setitem(reflect._SLOT_NAMES, kind, names)
    monkeypatch.setitem(reflect._SORTED_SLOT_NAMES, kind, names)

    _assert_encodes_alike(kind(x=1))
    _assert_encodes_alike(kind(x=1), sort=False)


class _FailureError(Exception):
    __slots__ = ("x",)
    __dataclass_fields__: ClassVar[dict[str, object]] = {}

    def __init__(self) -> None:
        super().__init__()
        self.x = 1


def test_a_slot_name_naming_a_builtin_member_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    # ``__suppress_context__`` is ``BaseException``'s bool member, not an
    # object slot, so its value is read by ``getattr``.
    names = ("x", "__suppress_context__")
    monkeypatch.setitem(reflect._SLOT_NAMES, _FailureError, names)
    monkeypatch.setitem(reflect._SORTED_SLOT_NAMES, _FailureError, names)
    value = _FailureError()

    _assert_encodes_alike(value, sort=False)


class _Importing:
    """A value whose reduce loads a module, so ``sys.modules`` changes mid-walk."""

    @override
    def __reduce__(self) -> tuple[object, ...]:
        sys.modules.setdefault(
            "treekle_class_loaded",
            ModuleType("treekle_class_loaded"),
        )
        sys.modules.pop("treekle_class_loaded")
        # A builtin callable: its path is the next read, after the change.
        return (int, ())


def test_a_nested_class_after_its_module_s_top_level_class_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(monkeypatch, "treekle_class_after")
    top = _slotted(module, "Top")
    inner = _slotted(module, "Outer.Inner")
    vars(module).update(
        Top=top,
        Outer=type("Outer", (), {"__module__": module.__name__, "Inner": inner}),
    )

    _assert_encodes_alike([top(x=1), inner(x=2)])


def test_a_class_with_no_module_entry_matches_python() -> None:
    # ``type`` takes ``__module__`` from the caller's globals; with no
    # ``__name__`` there, the class's dict holds none and its path is unread.
    code = compile("type('Bare', (), {})", "<moduleless>", "eval")
    kind = cast("object", FunctionType(code, {"__builtins__": __builtins__})())
    assert isinstance(kind, type)
    assert "__module__" not in vars(kind)

    _assert_encodes_alike(kind)


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
