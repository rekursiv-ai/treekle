"""Tests for ``treekle.wire``."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import json
import math

import pytest

from treekle.types import Hooks, ReadError
from treekle.wire import json_loads


class TestJsonLoads:
    def test_text_reads_as_a_mutable_tree(self) -> None:
        assert json_loads('{"a": [1, 2.5, null]}') == {"a": [1, 2.5, None]}
        assert json_loads(b"[true]") == [True]

    def test_non_finite_floats_read(self) -> None:
        loaded = json_loads("[NaN, Infinity]")
        assert isinstance(loaded, list)
        assert math.isnan(cast(float, loaded[0]))
        assert loaded[1] == math.inf

    def test_bad_text_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            json_loads("{")

    def test_a_target_reads_the_document_as_that_type(self) -> None:
        assert json_loads('{"x": [1, 2]}', dict[str, tuple[int, int]]) == {
            "x": (1, 2),
        }
        assert json_loads("null", None) is None

    def test_a_target_the_document_does_not_fit_raises(self) -> None:
        with pytest.raises(ReadError, match="cannot read 'a' as int"):
            json_loads('["a"]', list[int])

    def test_options_reach_from_plain(self) -> None:
        assert json_loads('"3"', int, strict=False) == 3
        with pytest.raises(ReadError, match="allow_imports"):
            json_loads('{"py/type": "pathlib.Path"}', object)
        assert (
            json_loads(
                '{"py/type": "pathlib.Path"}',
                object,
                allow_imports=True,
            )
            is Path
        )
        hooks: Hooks = {Path: (str, Path)}
        assert json_loads(
            '{"py/hook": ["pathlib.Path", "/a"]}',
            object,
            hooks=hooks,
        ) == Path("/a")


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
