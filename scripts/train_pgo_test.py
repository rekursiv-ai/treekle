"""The PGO training run round-trips the configs it builds, and stops on one that does not."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, cast

import sys

import pytest

from scripts import train_pgo
from treekle import deserialize


class Layer(Protocol):
    """One layer of a training config."""

    child: object


@dataclass(frozen=True, kw_only=True, slots=True)
class Leaf:
    """A config as small as the training loop can round-trip."""

    width: int = 1


def test_a_config_nests_its_layers_in_a_module_of_their_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    config = train_pgo._config(7, depth_count=2)
    kind = type(config)
    assert is_dataclass(kind)
    assert (kind.__module__, kind.__name__) == ("treekle_pgo_7", "Layer1")
    assert cast("object", sys.modules["treekle_pgo_7"].Layer1) is kind
    child = cast("Layer", config).child
    assert type(child).__name__ == "Layer0"
    assert cast("Layer", child).child is None
    assert [field.name for field in fields(kind)][-1] == "child"


def test_main_round_trips_every_config(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[int] = []

    def config(index: int) -> Leaf:
        built.append(index)
        return Leaf(width=index)

    monkeypatch.setattr(train_pgo, "_config", config)
    assert train_pgo.main() == 0
    assert built == list(range(200))


def test_main_stops_on_a_config_that_does_not_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def config(index: int) -> Leaf:
        return Leaf(width=index)

    def differs(*_: object, **kwargs: object) -> Leaf:
        assert kwargs == {"allow_imports": True}
        return Leaf(width=-1)

    monkeypatch.setattr(train_pgo, "_config", config)
    monkeypatch.setattr(deserialize, "from_plain", differs)
    with pytest.raises(SystemExit, match="does not round-trip"):
        train_pgo.main()


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
