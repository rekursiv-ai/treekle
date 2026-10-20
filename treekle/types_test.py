"""Tests for ``treekle.types``."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import pickle

from treekle.types import FieldPath, Invalid, ReadError


def test_a_read_error_holds_its_partial_value_and_bad_parts() -> None:
    invalid = Invalid(raw="x", reason="not an int")
    bad: dict[FieldPath, Invalid] = {("a",): invalid}

    error = ReadError("1 part does not read", partial={"a": invalid}, bad=bad)

    assert str(error) == "1 part does not read"
    assert error.partial == {"a": invalid}
    assert error.bad == bad


def test_a_read_error_defaults_to_no_partial_value_and_no_bad_parts() -> None:
    error = ReadError("nothing reads")

    assert error.partial is None
    assert error.bad == {}


class _DetailedError(ReadError):
    """A subclass whose ``__init__`` takes a field of its own."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        partial: object = None,
        bad: MappingProxyType[FieldPath, Invalid] = MappingProxyType({}),
    ) -> None:
        super().__init__(message, partial=partial, bad=bad)
        self.code = code


def test_a_read_error_subclass_survives_pickling_as_itself() -> None:
    invalid = Invalid(raw="x", reason="not an int")
    error = _DetailedError(
        "1 part does not read",
        code=3,
        partial=[invalid],
        bad=MappingProxyType({(0,): invalid}),
    )

    restored = cast("object", pickle.loads(pickle.dumps(error)))

    assert isinstance(restored, _DetailedError)
    assert type(restored) is _DetailedError
    assert restored.args == ("1 part does not read",)
    assert restored.code == 3
    assert restored.partial == [invalid]
    assert restored.bad == {(0,): invalid}
    assert isinstance(restored.bad, MappingProxyType)


def test_a_read_error_survives_pickling_with_its_parts() -> None:
    # Process pools and pytest-xdist move a raised error between processes.
    invalid = Invalid(raw="x", reason="not an int")
    error = ReadError(
        "1 part does not read",
        partial={"a": invalid},
        bad=MappingProxyType({("a",): invalid}),
    )

    restored = cast("object", pickle.loads(pickle.dumps(error)))

    assert isinstance(restored, ReadError)
    assert type(restored) is ReadError
    assert str(restored) == "1 part does not read"
    assert restored.partial == {"a": invalid}
    assert restored.bad == {("a",): invalid}
    assert isinstance(restored.bad, MappingProxyType)


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
