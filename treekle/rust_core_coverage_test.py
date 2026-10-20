"""Tests that the Rust core's shared helpers act as the pure-Python walk does.

Each case reaches a native helper (freezing a tree, unpacking a mapping's
items, reading an option's truth) through ``to_plain`` or ``from_plain``,
then compares the result with the pure-Python function's.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override

import gc
import subprocess
import sys
import threading
import weakref

import pytest

from treekle import deserialize, serialize


if TYPE_CHECKING:
    from collections.abc import Callable

    from treekle.serialize import _Encoding


def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result, or the type and text of what it raised."""
    try:
        return ("ok", call())
    except Exception as error:  # noqa: BLE001 -- Parity compares whatever either walker raises.
        return ("raise", type(error).__name__, str(error))


class _Foreign:
    """A value a caller-installed encoder writes."""


def _cycle() -> list[object]:
    cycle: list[object] = []
    cycle.append(cycle)
    return cycle


@pytest.mark.parametrize(
    "written",
    [
        5.0,
        {1: "an int key"},
        _cycle(),
        {"held": object()},
        [object()],
    ],
    ids=[
        "a leaf",
        "an int key",
        "a cycle",
        "an object in a dict",
        "an object in a list",
    ],
)
def test_freezing_what_a_foreign_encoder_wrote_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    written: object,
) -> None:
    def encode(value: object, state: _Encoding) -> object:
        del value, state
        return written

    encoders = dict(serialize._ENCODERS)
    encoders[_Foreign] = encode
    monkeypatch.setattr(serialize, "_ENCODERS", MappingProxyType(encoders))
    value = _Foreign()

    expected = _outcome(lambda: serialize._python_to_plain(value, mutable=False))
    assert _outcome(lambda: serialize.to_plain(value, mutable=False)) == expected


def _encoded(walker: Callable[..., object], value: object, sort: object) -> object:
    """Return what ``walker`` makes of ``value``, given a ``sort`` of any type."""
    return _outcome(lambda: walker(value, sort=sort))


class _Truthy:
    """An option whose truth is asked of Python at each use."""

    def __init__(self, truth: bool) -> None:
        self.truth = truth

    def __bool__(self) -> bool:
        return self.truth


@pytest.mark.parametrize("truth", [True, False])
def test_a_sort_option_that_is_not_a_bool_matches_python(truth: bool) -> None:
    class Bare:
        def __init__(self) -> None:
            self.b = 1
            self.a = 2

    value = Bare()
    sort = _Truthy(truth)

    expected = _encoded(serialize._python_to_plain, value, sort)
    assert _encoded(serialize.to_plain, value, sort) == expected


class _Hooks(Mapping[type, tuple[object, object]]):
    """A hooks mapping whose ``items`` yields the rows it was given."""

    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    @override
    def __getitem__(self, key: type) -> tuple[object, object]:
        raise KeyError(key)

    @override
    def __iter__(self) -> Iterator[type]:
        return iter(())

    @override
    def __len__(self) -> int:
        return len(self.rows)

    @override
    def items(self) -> list[object]:  # ty: ignore[invalid-method-override] -- Rows are malformed on purpose, so they are no ItemsView.  # pyright: ignore[reportIncompatibleMethodOverride] -- Rows are malformed on purpose, so they are no ItemsView.  # pyrefly: ignore[bad-override] -- Rows are malformed on purpose, so they are no ItemsView.
        return self.rows


class _Hooked:
    """The class a hooked record names."""


def _pair() -> tuple[Callable[[object], object], Callable[[object], object]]:
    return (lambda _value: 0, lambda _raw: _Hooked())


@pytest.mark.parametrize(
    "row",
    [
        lambda: (_Hooked, _pair()),
        lambda: [_Hooked, _pair()],
        lambda: iter((_Hooked, _pair())),
        lambda: (_Hooked,),
        lambda: (_Hooked, _pair(), 1),
        lambda: [_Hooked, _pair(), 1, 2],
        lambda: iter((_Hooked,)),
        lambda: iter(()),
        lambda: iter((_Hooked, _pair(), 1)),
    ],
    ids=[
        "tuple",
        "list",
        "iterator",
        "short tuple",
        "long tuple",
        "long list",
        "short iterator",
        "empty iterator",
        "long iterator",
    ],
)
def test_hook_rows_unpack_as_python_unpacks_them(row: Callable[[], object]) -> None:
    data = {"py/hook": [f"{_Hooked.__module__}.{_Hooked.__qualname__}", 0]}

    def read(function: Callable[..., object]) -> object:
        # A fresh row per read: an iterator row is used up by the first.
        hooks = _Hooks([row()])
        return _outcome(lambda: type(function(data, object, hooks=hooks)).__name__)

    assert read(deserialize.from_plain) == read(deserialize._python_from_plain)


class _Record(dict[object, object]):
    """A mapping that is not an exact dict."""


@pytest.mark.parametrize(
    "data",
    [
        {"x": 1, 2: "an int key"},
        {"x": 1, "é": 2},
        _Record({"x": 1}),
    ],
    ids=["a key that is not a str", "a non-ASCII key", "a dict subclass"],
)
def test_a_record_with_unusual_keys_reads_as_python_reads_it(
    data: dict[object, object],
) -> None:
    @dataclass(kw_only=True, slots=True)
    class Record:
        x: int

    def read(function: Callable[..., object]) -> object:
        return _outcome(lambda: repr(function(data, Record)))

    assert read(deserialize.from_plain) == read(deserialize._python_from_plain)


def test_a_sort_option_that_is_not_a_bool_reaches_a_foreign_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def encode(value: object, state: _Encoding) -> object:
        del value
        return {"sorted": bool(state.sort)}  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.

    encoders = dict(serialize._ENCODERS)
    encoders[_Foreign] = encode
    monkeypatch.setattr(serialize, "_ENCODERS", MappingProxyType(encoders))
    sort = _Truthy(True)

    expected = _encoded(serialize._python_to_plain, _Foreign(), sort)
    assert _encoded(serialize.to_plain, _Foreign(), sort) == expected


@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="CPython checks the C stack from 3.14 on; before, the Python walk crashes.",
)
def test_a_set_nested_past_the_stack_raises_as_python_does() -> None:
    # Ordering a set's members encodes each one in a nested walk; on a
    # small thread stack both walks run out and raise ``RecursionError``.
    # The messages name different frames, so only the type is compared.
    value: object = 0
    for _ in range(30_000):
        value = frozenset[object]([(value,)])
    raised: list[str] = []

    def run(function: Callable[[object], object]) -> None:
        try:
            function(value)
        except RecursionError as error:
            raised.append(type(error).__name__)

    previous = threading.stack_size(512 * 1024)
    try:
        for function in (serialize.to_plain, serialize._python_to_plain):
            thread = threading.Thread(target=run, args=(function,))
            thread.start()
            thread.join()
    finally:
        threading.stack_size(previous)
    assert raised == ["RecursionError", "RecursionError"]


_NO_WATCHERS = """
import ctypes
from dataclasses import dataclass

callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.py_object
)
callback = callback_type(lambda *_: 0)
while True:
    try:
        ctypes.pythonapi.PyDict_AddWatcher(callback)
    except RuntimeError:
        break

from treekle import deserialize, serialize


@dataclass(slots=True)
class Point:
    x: int


value = {"points": [Point(1), Point(2)], "kind": Point}
tree = serialize.to_plain(value)
assert tree == serialize._python_to_plain(value), tree
read = deserialize.from_plain(tree, dict[str, object], allow_imports=True)
assert repr(read) == repr(deserialize._python_from_plain(tree, dict[str, object], allow_imports=True))
print("same")
"""


def test_with_no_dict_watcher_left_the_core_reads_every_time_and_matches_python() -> (
    None
):
    # CPython allows a few dict watchers per interpreter; with all taken
    # before the core loads, it keeps no fact across reads.
    completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
        [sys.executable, "-c", _NO_WATCHERS],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "same"


_FREE_WATCHERS = """
import ctypes
import sys
from dataclasses import dataclass

callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.py_object
)
callback = callback_type(lambda *_: 0)


def free() -> int:
    taken = []
    while True:
        try:
            taken.append(ctypes.pythonapi.PyDict_AddWatcher(callback))
        except RuntimeError:
            break
    for watcher in taken:
        ctypes.pythonapi.PyDict_ClearWatcher(watcher)
    return len(taken)


before = free()
from treekle import serialize


@dataclass(slots=True)
class Point:
    x: int


sys.modules[__name__].Point = Point
serialize.to_plain([Point(1), Point(2), Point])
print(before - free())
"""


@pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)
def test_the_core_takes_one_dict_watcher_for_all_it_watches() -> None:
    # An interpreter allows few dict watchers; the core's caches and its
    # ``sys.modules`` watch share one.
    completed = subprocess.run(  # noqa: S603 -- Runs this interpreter on the test's own script.
        [sys.executable, "-c", _FREE_WATCHERS],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1"


@dataclass(slots=True)  # house-ignore[dataclass] -- Built positionally by its test.
class _Point:
    x: int


def test_cache_keys_that_name_no_live_class_leave_encoding_as_python_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The core watches these caches and notes what each new key names: an
    # ``id`` too large for an address names nothing, nor does a dead weakref.
    serialize.to_plain(_Point(1))
    monkeypatch.setitem(serialize._IMPORT_PATHS, 2**70, None)

    class Gone:
        pass

    # A weakref hashes its referent once; hashed while alive, it stays a key
    # after the referent dies, and is added to the watched cache then.
    gone = Gone()
    dead = weakref.ref(gone)
    hash(dead)
    del gone
    gc.collect()
    # Typeshed leaves out ``WeakKeyDictionary.data``, the dict it wraps.
    data = cast("dict[object, object]", vars(serialize._ENCODER_CACHE)["data"])
    monkeypatch.setitem(data, dead, None)

    expected = _outcome(lambda: serialize._python_to_plain(_Point(2)))
    assert _outcome(lambda: serialize.to_plain(_Point(2))) == expected


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
