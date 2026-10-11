"""Tests for ``treekle.wire``."""

from __future__ import annotations

from typing import cast

import json
import math

import pytest

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


if __name__ == "__main__":
    from treekle.lib.testing.main import test_main

    test_main(__file__)
