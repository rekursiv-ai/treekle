"""Tests for ``treekle.types``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from treekle.types import Invalid, ReadError


if TYPE_CHECKING:
    from treekle.types import FieldPath


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


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
