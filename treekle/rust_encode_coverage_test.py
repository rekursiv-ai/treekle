"""Tests that the Rust encoder takes its less common paths as Python does.

Each case builds a value or replaces a ``serialize`` helper so the native
walk reaches a rarely taken branch, then compares ``to_plain`` with
``_python_to_plain``. The native call runs first where Python's walk would
fill a cache the native one then reads instead.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from types import FrameType, GenericAlias, ModuleType
from typing import TYPE_CHECKING, cast, override

import copyreg
import gc
import itertools
import sys
import weakref

import pytest

from treekle import deserialize, reflect, serialize


if TYPE_CHECKING:
    from collections.abc import Callable


def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result, or the type and text of what it raised."""
    try:
        return ("ok", call())
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either walker raises.
        return ("raise", type(error).__name__, str(error))


def _encoded(
    walker: Callable[..., object],
    value: object,
    options: dict[str, object],
) -> object:
    """Return what ``walker`` makes of ``value``, given options of any type."""
    return _outcome(lambda: walker(value, **options))


def _assert_encodes_alike(value: object, **options: object) -> None:
    actual = _encoded(serialize.to_plain, value, options)
    assert actual == _encoded(serialize._python_to_plain, value, options)


@dataclass(slots=True)  # house-ignore[dataclass] -- Built positionally by its tests.
class _Point:
    x: int


class _Bare:
    """A class with an instance ``__dict__`` and no slots; it encodes by reduce."""

    def __init__(self, **members: object) -> None:
        vars(self).update(members)


@dataclass  # house-ignore[dataclass] -- Its members are set per test.
class _Loose:
    """A dataclass with an instance ``__dict__``; it encodes as an object."""

    def __init__(self, **members: object) -> None:
        vars(self).update(members)


def test_a_huge_tree_does_not_keep_its_buffers_and_matches_python() -> None:
    # Past the buffer limit the walk drops its buffers rather than keep them.
    value = [[index] for index in range(70_000)]

    _assert_encodes_alike(value)
    _assert_encodes_alike([[1]])


class _Reducing:
    """A value whose reduce recipe is given, then retried by protocol."""

    def __init__(self, recipe: str | tuple[object, ...]) -> None:
        self.recipe = recipe

    @override
    def __reduce__(self) -> str | tuple[object, ...]:
        return self.recipe


class _Unencodable:
    """A leaf with no conversion: it raises ``TypeError``."""

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


class _Raising:
    """A value whose reduce works but whose protocol fallback raises."""

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_Raising, (_Unencodable(),))


@pytest.mark.parametrize(
    "value",
    [
        _Raising(),
        [_Raising()],
        _Reducing((_Point, (), None, None, None)),
        _Reducing((None, None)),
    ],
    ids=[
        "a reduce at the root",
        "a reduce in a list",
        "a recipe of Nones",
        "a recipe all None",
    ],
)
def test_a_reduce_whose_part_fails_recovers_as_python_does(value: object) -> None:
    _assert_encodes_alike(value)


class _BadMapping(Mapping[object, object]):
    """A mapping whose keys are not plain, read by iteration as Python reads it."""

    def __init__(self, items: dict[object, object]) -> None:
        self.items_ = items

    @override
    def __getitem__(self, key: object) -> object:
        return self.items_[key]

    @override
    def __iter__(self) -> Iterator[object]:
        return iter(self.items_)

    @override
    def __len__(self) -> int:
        return len(self.items_)


class _Unreducible(_BadMapping):
    """A mapping whose reduce is refused, so it encodes as a mapping."""

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("no reduce")


@pytest.mark.parametrize(
    "items",
    [{"a": 1}, {1: "int key"}, {"py/x": 1}],
    ids=["plain keys", "an int key", "a tag-like key"],
)
def test_a_mapping_that_is_not_a_dict_matches_python(
    items: dict[object, object],
) -> None:
    _assert_encodes_alike(_Unreducible(items))


def test_an_object_with_a_member_named_like_its_tag_matches_python() -> None:
    _assert_encodes_alike(_Loose(**{"py/object": 1, "b": [1]}))


def test_an_object_whose_dict_holds_a_non_str_key_matches_python() -> None:
    value = _Loose(a=1)
    members = cast("dict[object, object]", cast("object", vars(value)))
    members[1] = "int key"

    _assert_encodes_alike(value)


@dataclass  # house-ignore[dataclass] -- Declares its own slots, one of them ``__dict__``.
class _Mixed:
    __slots__ = ("__dict__", "b")


@dataclass  # house-ignore[dataclass] -- Needs an instance ``__dict__``.
class _Shadowed:
    @property
    def a(self) -> int:
        return 9

    @a.setter
    def a(self, value: int) -> None:
        del value


@dataclass  # house-ignore[dataclass] -- Its ``__dict__`` is its own property.
class _Proxied:
    @property
    @override
    def __dict__(self) -> dict[str, object]:  # pyright: ignore[reportIncompatibleVariableOverride] -- ``object`` declares ``__dict__`` a variable; a property is the case under test.  # pyrefly: ignore[bad-override] -- ``object`` declares ``__dict__`` a variable; a property is the case under test.
        return {"a": 1}


def test_a_slotted_object_with_a_dict_and_extra_keys_matches_python() -> None:
    value = _Mixed()
    value.b = 1  # pyrefly: ignore[missing-attribute] -- pyrefly reads no attribute from ``__slots__``; ``b`` is one.
    vars(value).update(z=2, a=3)

    _assert_encodes_alike(value, sort=True)
    _assert_encodes_alike(value, sort=False)


@pytest.mark.parametrize("sort", [True, False])
def test_objects_of_one_class_with_their_own_dict_orders_match_python(
    sort: bool,
) -> None:
    first = _Loose(b=1, a=2)
    second = _Loose(b=3, a=4)
    reordered = _Loose(a=5, b=6)
    pending = _Loose(b=reflect.PENDING, a=7)

    _assert_encodes_alike([first, second, reordered, first, pending, second], sort=sort)


def test_a_dict_member_shadowed_by_a_descriptor_matches_python() -> None:
    value = _Shadowed()
    vars(value)["a"] = 1

    _assert_encodes_alike(value)


def test_an_object_whose_dict_is_not_its_generic_dict_matches_python() -> None:
    _assert_encodes_alike(_Proxied())


class _Shape(Enum):
    ROUND = 1
    SQUARE = 2


class _Patched(Enum):
    ONE = 1


def test_an_enum_member_with_its_own_reduce_matches_python() -> None:
    member = _Patched.ONE

    def reduce_ex(protocol: object) -> tuple[object, ...]:
        del protocol
        return (_Patched, (1,))

    vars(member)["__reduce_ex__"] = reduce_ex

    _assert_encodes_alike([member, _Shape.ROUND])


def test_an_enum_member_without_a_value_attribute_matches_python() -> None:
    member = _Shape.SQUARE
    value = cast("object", vars(member).pop("_value_"))
    try:
        _assert_encodes_alike(member)
    finally:
        vars(member)["_value_"] = value


class _Real:
    """A number that is no float, read as one through ``__float__``."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


class _NoFloat:
    def __float__(self) -> float:
        raise ValueError("no float")


class _Float(float):
    __slots__ = ()


@pytest.mark.parametrize(
    "value",
    [
        [_Real(1.5), _Real(float("inf")), _NoFloat()],
        [_Float(1.5), _Float("inf"), _Float("nan")],
        len,
        [_Point(1), _Bare(a=1), _Point, len],
        [_Shape.ROUND, _Reducing((_Point, (1,)))],
    ],
    ids=[
        "numbers read through __float__",
        "float subclasses",
        "a builtin function",
        "records, plain objects, classes and functions",
        "enum members and reduce recipes",
    ],
)
def test_values_each_encoder_takes_match_python(value: object) -> None:
    _assert_encodes_alike(value)


def test_a_hooked_class_beside_an_unreduced_value_matches_python() -> None:
    def encode(real: _Real) -> float:
        return real.value

    hooks = {_Real: (encode, _Real)}

    _assert_encodes_alike([_Real(2.0), _Bare(a=1)], hooks=hooks)
    _assert_encodes_alike([_Real(2.0), _Bare(a=1)])


@pytest.mark.parametrize("dialect", ["json", "python"])
def test_a_dialect_that_is_a_str_subclass_matches_python(dialect: str) -> None:
    class Dialect(str):
        __slots__ = ()

    _assert_encodes_alike([float("nan")], dialect=Dialect(dialect))


def test_a_root_leaf_with_an_invalid_dialect_matches_python() -> None:
    _assert_encodes_alike(1, dialect="yaml")


def test_a_bare_name_reduce_names_a_module_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("treekle_encode_global")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def reduce(self: object) -> str:
        del self
        return "ONE"

    kind = type("Single", (), {"__module__": module.__name__, "__reduce__": reduce})
    one = kind()
    vars(module).update(Single=kind, ONE=one)

    _assert_encodes_alike(one)


def test_a_reduce_through_copyreg_constructor_matches_python() -> None:
    # Typeshed leaves out ``copyreg._reconstructor``.
    reconstructor = cast("object", vars(copyreg)["_reconstructor"])
    value = _Reducing((reconstructor, (_Bare, object, None)))

    _assert_encodes_alike(value)


class _Root:
    """A reduce at the root whose argument fails, so the root restarts."""

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_Root, ([_Unencodable()],))


class _RootSlotted:
    """A reduce at the root that restarts by its slots."""

    __slots__ = ("x",)

    def __init__(self) -> None:
        self.x = [1]

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_RootSlotted, ([_Unencodable()],))


class _KeyError:
    """A reduce whose part fails, then whose protocol fallback raises KeyError."""

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_KeyError, ([_Unencodable()],))

    @property
    @override
    def __dict__(self) -> dict[str, object]:  # pyright: ignore[reportIncompatibleVariableOverride] -- ``object`` declares ``__dict__`` a variable; a property is the case under test.  # pyrefly: ignore[bad-override] -- ``object`` declares ``__dict__`` a variable; a property is the case under test.
        raise KeyError("no dict")


def _deep(value: object, depth: int = 64) -> object:
    """Return ``value`` nested ``depth`` lists down, past the walk's in-place depth."""
    for _ in range(depth):
        value = list[object]([value])
    return value


@pytest.mark.parametrize(
    "value",
    [
        _Root(),
        _RootSlotted(),
        [_KeyError()],
        _deep(_Root()),
        _deep(_RootSlotted()),
        _deep([_KeyError()]),
        _deep(_Raising()),
        _Reducing((_Root, (_deep(_Unencodable()),))),
    ],
    ids=[
        "dict state",
        "slots",
        "a fallback that raises another error",
        "deep dict state",
        "deep slots",
        "deep fallback that raises another error",
        "deep reduce in a list",
        "a root reduce whose deep part fails",
    ],
)
def test_a_reduce_whose_part_fails_deep_down_restarts_as_python_does(
    value: object,
) -> None:
    _assert_encodes_alike(value)


class _Name(str):
    """A member name that is a str, but not an exact one."""

    __slots__ = ()


@pytest.mark.parametrize("sort", [True, False])
def test_an_object_with_a_str_subclass_member_name_matches_python(sort: bool) -> None:
    value = _Loose(a=1)
    vars(value)[_Name("b")] = [2]

    _assert_encodes_alike(value, sort=sort)


@pytest.mark.parametrize("value", [1, "text", True, None], ids=repr)
def test_a_root_leaf_is_itself_as_in_python(value: object) -> None:
    _assert_encodes_alike(value, mutable=False)


@dataclass(slots=True)  # house-ignore[dataclass] -- Built positionally by its test.
class _Pending:
    a: object
    b: object


def test_a_pending_slot_is_left_out_as_python_leaves_it_out() -> None:
    # Twice, so the second object takes the class's kept layout.
    values = [_Pending(reflect.PENDING, 1), _Pending(2, reflect.PENDING)]

    _assert_encodes_alike(values)
    _assert_encodes_alike(values)


def test_an_enum_member_whose_dict_is_not_a_dict_matches_python() -> None:
    class Odd(dict[str, object]):
        pass

    member = _Patched.ONE
    original = member.__dict__
    object.__setattr__(member, "__dict__", Odd(original))
    try:
        _assert_encodes_alike(member)
    finally:
        object.__setattr__(member, "__dict__", original)


def test_a_reduce_whose_recipe_is_all_none_matches_python() -> None:
    _assert_encodes_alike(_Reducing((None,)))


@dataclass  # house-ignore[dataclass] -- Takes no fields.
class _RootData:
    """A dataclass whose reduce holds a deep part that fails: it restarts as an object."""

    def __init__(self) -> None:
        self.a = [1]

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_RootData, (_deep(_Unencodable()),))


@dataclass  # house-ignore[dataclass] -- Takes no fields.
class _BadData:
    """A dataclass whose restart raises something other than ``TypeError``."""

    def __init__(self) -> None:
        self.a = 1

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_BadData, (_deep(_Unencodable()),))

    @override
    def __getattribute__(self, name: str) -> object:
        if name == "a":
            raise KeyError(name)
        return cast("object", object.__getattribute__(self, name))


@pytest.mark.parametrize(
    "value",
    [
        _RootData(),
        [_RootData()],
        _BadData(),
        [_BadData()],
        _deep(_RootData()),
        _deep([_BadData()]),
        _deep(_RootData(), 128),
        _deep([_BadData()], 128),
    ],
    ids=[
        "at the root",
        "in a list",
        "raising at the root",
        "raising in a list",
        "past the in-place depth",
        "raising past the in-place depth",
        "far past the in-place depth",
        "raising far past the in-place depth",
    ],
)
def test_a_dataclass_reduce_that_fails_deep_down_restarts_as_python_does(
    value: object,
) -> None:
    _assert_encodes_alike(value)


class _SlottedRestart:
    """A slotted value whose reduce argument fails far down; it restarts by its slots."""

    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a = [1]

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_SlottedRestart, (_deep(_Unencodable(), 130),))


class _SlottedRaising:
    """A slotted value whose restart reads a slot that raises ``KeyError``."""

    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a = 1

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_SlottedRaising, (_deep(_Unencodable(), 130),))

    @override
    def __getattribute__(self, name: str) -> object:
        if name == "a":
            raise KeyError(name)
        return cast("object", object.__getattribute__(self, name))


@pytest.mark.parametrize(
    "value",
    [
        _SlottedRestart(),
        _deep(_SlottedRestart(), 70),
        [_SlottedRaising()],
        _deep(_SlottedRaising(), 70),
    ],
    ids=["at the root", "deep", "raising", "raising deep"],
)
def test_a_slotted_reduce_that_fails_far_down_restarts_as_python_does(
    value: object,
) -> None:
    _assert_encodes_alike(value)


class _QueuedRestart:
    """A slotted value whose reduce has two parts; the first fails at once."""

    __slots__ = ("a",)

    def __init__(self) -> None:
        self.a = [1]

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        return (_QueuedRestart, ([_Unencodable()], [2]))


@pytest.mark.parametrize("depth", range(44, 54))
def test_a_reduce_at_the_in_place_depth_restarts_as_python_does(depth: int) -> None:
    # At the in-place depth a reduce's parts wait on the stack, so a part
    # that fails is popped, and the parts after it are dropped.
    _assert_encodes_alike(_deep(_QueuedRestart(), depth))


@dataclass  # house-ignore[dataclass] -- Its members are set per test.
class _Looked:
    """A dataclass with its own ``__getattribute__``, so its dict is read by ``getattr``."""

    def __init__(self, **members: object) -> None:
        vars(self).update(members)

    @override
    def __getattribute__(self, name: str) -> object:
        return cast("object", object.__getattribute__(self, name))


class _Members(dict[str, object]):
    """An instance ``__dict__`` that is not an exact dict."""


@dataclass  # house-ignore[dataclass] -- Its members are set per test.
class _OddDict:
    """A dataclass whose ``__dict__`` attribute is a dict subclass it returns."""

    def __init__(self) -> None:
        object.__setattr__(self, "__dict__", _Members(a=1))


@pytest.mark.parametrize("sort", [True, False])
def test_an_object_read_through_its_own_getattribute_matches_python(sort: bool) -> None:
    _assert_encodes_alike([_Looked(b=1, a=2), _OddDict()], sort=sort)


def _treekle_calls(call: Callable[[], object]) -> list[str]:
    """Return the treekle library functions ``call`` runs in Python, by name."""
    names: list[str] = []
    files = {module.__file__ for module in (serialize, reflect, deserialize)}

    def profile(frame: FrameType, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code.co_filename in files:
            names.append(frame.f_code.co_name)

    outer = sys.getprofile()
    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(outer)
    return names


def _plain_module() -> ModuleType:
    """Return a fresh module holding ``Plain``, a dataclass whose state is its instance dict."""
    module = ModuleType(f"{__name__}_plain_{len(sys.modules)}")
    sys.modules[module.__name__] = module
    kind = dataclass(
        type(
            "Plain",
            (),
            {
                "__annotations__": {"a": int, "b": list[int]},
                "__module__": module.__name__,
            },
        ),
    )
    vars(module)["Plain"] = kind
    return module


@pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)
def test_a_rebound_module_global_keeps_what_the_core_read_of_a_class() -> None:
    # A class's facts hold while the class is unchanged; a name rebound in
    # its module changes nothing the walk read of it.
    module = _plain_module()
    value = cast("Callable[..., object]", vars(module)["Plain"])(1, [2])
    expected = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == expected
    vars(module)["counter"] = 1

    assert _treekle_calls(lambda: serialize.to_plain(value)) == []
    assert serialize.to_plain(value) == expected


@pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)
def test_a_dataclass_with_an_instance_dict_is_read_natively_the_first_time() -> None:
    module = _plain_module()
    value = cast("Callable[..., object]", vars(module)["Plain"])(1, [2])

    calls = _treekle_calls(lambda: serialize.to_plain(value))

    assert calls == []
    assert serialize.to_plain(value) == serialize._python_to_plain(value)


def test_an_implied_record_with_an_extra_attribute_is_tagged_as_python_tags_it() -> (
    None
):
    # A record goes untagged only where every member is a field; an
    # attribute set on the instance beyond them keeps its tag.
    kind = cast("Callable[..., object]", vars(_plain_module())["Plain"])
    value = kind(1, [2])
    vars(value)["extra"] = 3

    expected = serialize._python_to_plain(value, target=kind)
    assert isinstance(expected, dict)
    assert "py/object" in expected
    assert serialize.to_plain(value, target=kind) == expected
    assert serialize.to_plain(value, target=kind) == expected


def test_a_hint_that_resolves_later_is_read_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unresolved hint may resolve once its module gains the name, so the
    # field types read before it does must not be kept for good.
    module = ModuleType(f"{__name__}_later")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    holder = dataclass(slots=True)(
        type(
            "Holder",
            (),
            {"__annotations__": {"part": "Later"}, "__module__": module.__name__},
        ),
    )
    vars(module)["Holder"] = holder
    value = cast("Callable[..., object]", holder)(_Point(1))
    before = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == before

    vars(module)["Later"] = _Point

    after = serialize._python_to_plain(value)
    assert after != before
    assert serialize.to_plain(value) == after


@pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)
def test_classes_past_the_core_s_limit_are_let_go_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The core keeps what it read of each class it met, in a table of bounded
    # size: past the bound each new class takes the oldest one's place, so no
    # call lets go of every class's facts at once. Each kept entry holds its
    # class's qualname, so the classes share one to count what is kept. The
    # module names each class as it is written, so its path resolves.
    module = ModuleType(f"{__name__}_many")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    qualname = "".join(("Ma", "ny"))
    kinds: list[type] = []
    kept: list[int] = []
    for _ in range(20_000):
        kind = type(
            "Many",
            (),
            {"__qualname__": qualname, "__module__": module.__name__},
        )
        kinds.append(kind)
        vars(module)["Many"] = kind
        serialize.to_plain(kind())
        kept.append(sys.getrefcount(qualname) - len(kinds))

    assert min(later - earlier for earlier, later in itertools.pairwise(kept)) >= -1
    assert kept[-1] > len(kinds) // 2
    vars(module)["Many"] = kinds[0]
    first = kinds[0]()
    assert serialize.to_plain(first) == serialize._python_to_plain(first)


@pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)
def test_declared_types_past_the_core_s_limit_are_let_go_one_at_a_time() -> None:
    # Each ``list[int]`` written anew is its own declared type, kept with what
    # the core decided for it until newer ones push it out.
    held: list[weakref.ref[GenericAlias]] = []
    for _ in range(20_000):
        declared = GenericAlias(list, (int,))
        held.append(weakref.ref(declared))
        assert serialize.to_plain([1], target=declared) == [1]
        del declared
    gc.collect()

    let_go = sum(ref() is None for ref in held)
    assert 0 < let_go < len(held) // 2


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
