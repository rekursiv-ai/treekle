"""Tests for ``treekle.parse``."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID

import datetime as dt
import math

import pytest

from treekle.parse import parse
from treekle.types import ReadError


class Mode(Enum):
    FAST = "fast"
    SLOW = 2


class TestParse:
    @pytest.mark.parametrize("text", ["123", "1.5", "run_a", "true", "null"])
    def test_a_str_target_keeps_the_text(self, text: str) -> None:
        assert parse(text, str) == text

    def test_a_quoted_json_string_reads_as_its_text(self) -> None:
        assert parse('"a b"', str) == "a b"

    @pytest.mark.parametrize(
        ("text", "target", "expected"),
        [
            ("7", int, 7),
            ("3e-4", float, 3e-4),
            ("2", float, 2.0),
            ("true", bool, True),
            ("[1, 2]", list[int], [1, 2]),
            ('{"a": 1}', dict[str, int], {"a": 1}),
            ("[1, 2]", tuple[int, int], (1, 2)),
            ("null", int | None, None),
            ("7", int | str, 7),
            ("run_a", int | str, "run_a"),
            ("123", str | None, "123"),
            ("fast", Mode, Mode.FAST),
            ("2", Mode, Mode.SLOW),
            ("a", Literal["a", "b"], "a"),
            ("/data/x", Path, Path("/data/x")),
            ('"/data/x"', Path, Path("/data/x")),
            ("00000000-0000-0000-0000-000000000001", UUID, UUID(int=1)),
            (
                "2026-01-02T03:04:05+00:00",
                dt.datetime,
                dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.UTC),
            ),
        ],
        ids=repr,
    )
    def test_text_reads_as_its_target(
        self,
        text: str,
        target: object,
        expected: object,
    ) -> None:
        value = parse(text, target)

        assert value == expected
        assert type(value) is type(expected)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("inf", math.inf),
            ("-inf", -math.inf),
            ("Infinity", math.inf),
            (" inf ", math.inf),
        ],
    )
    def test_non_finite_spellings_read_as_floats(
        self,
        text: str,
        expected: float,
    ) -> None:
        assert parse(text, float) == expected

    def test_nan_reads_as_a_float(self) -> None:
        assert math.isnan(parse("nan", float))

    def test_a_finite_number_is_not_a_spelling(self) -> None:
        value = parse("1e3", float | str)

        assert value == 1000.0
        assert type(value) is float

    @pytest.mark.parametrize(
        ("text", "target", "match"),
        [
            ("abc", int, "cannot read 'abc' as int"),
            ("[1]", int, r"cannot read \[1\] as int"),
            ("{}", int, "cannot read {} as int"),
            ("1.5", int, "cannot read 1.5 as int"),
            ("yes", bool, "cannot read 'yes' as bool"),
            ("inf", int, "cannot read inf as int"),
            ("[1, 2, 3]", tuple[int, int], "expected 2 items"),
            ("medium", Mode, "cannot read 'medium' as"),
        ],
        ids=repr,
    )
    def test_text_no_reading_fits_raises(
        self,
        text: str,
        target: object,
        match: str,
    ) -> None:
        with pytest.raises(ReadError, match=match):
            parse(text, target)


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
