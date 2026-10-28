"""The PGO training run round-trips the configs it builds, and stops on one that does not."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, cast

import sys

import pytest

from scripts import train_pgo
from treekle import deserialize, serialize, wire


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


def test_a_config_writes_each_kind_the_profile_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each layer's ``options`` is a plain object, written as ``py/reduce``:
    # only the walk reads it, so training on it keeps the walk in the profile.
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    config = train_pgo._config(7, depth_count=2)

    tree = serialize.to_plain(config)

    def options(depth: int) -> object:
        arguments: list[object] = []
        return {
            "py/reduce": [
                {"py/type": "types.SimpleNamespace"},
                {"py/tuple": arguments},
                {"depth": depth},
            ],
        }

    assert tree == {
        "py/object": "treekle_pgo_7.Layer1",
        "child": {
            "child": None,
            "enabled": True,
            "extra": {"scale": 1.5},
            "name": "layer0",
            "options": options(0),
            "rate": 0.0,
            "shape": [0, 1],
            "sizes": [0, 0],
            "width": 0,
        },
        "enabled": False,
        "extra": {"scale": 1.5},
        "name": "layer1",
        "options": options(1),
        "rate": 0.5,
        "shape": [1, 2],
        "sizes": [1, 2],
        "width": 1,
    }
    assert deserialize.from_plain(tree, type(config), allow_imports=True) == config


def test_each_layer_is_a_frozen_slotted_keyword_dataclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    config = train_pgo._config(7, depth_count=3)
    layers = [config]
    while (child := cast("Layer", layers[-1]).child) is not None:
        layers.append(child)

    for layer in layers:
        kind = type(layer)
        assert is_dataclass(kind)
        params = cast("_Params", vars(kind)["__dataclass_params__"])
        assert (params.frozen, params.kw_only) == (True, True)
        assert "__slots__" in vars(kind)
        factories = {each.name: each.default_factory for each in fields(kind)}
        assert (factories["sizes"], factories["extra"]) == (list[int], dict[str, float])
    assert [cast("_Enabled", layer).enabled for layer in layers] == [True, False, True]
    last = type(layers[-1])
    assert is_dataclass(last)
    deepest = {each.name: each.type for each in fields(last)}
    assert deepest["child"] is object


class _Params(Protocol):
    frozen: bool
    kw_only: bool


class _Enabled(Protocol):
    enabled: bool


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


def test_main_round_trips_every_config_through_json_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The JSON writer and reader are trained as the tree walks are, so the
    # profile does not leave them cold.
    texts: list[str] = []

    def config(index: int) -> Leaf:
        return Leaf(width=index)

    def differs(text: str, *_: object, **kwargs: object) -> Leaf:
        assert kwargs == {"allow_imports": True}
        texts.append(text)
        return Leaf(width=-1)

    monkeypatch.setattr(train_pgo, "_config", config)
    monkeypatch.setattr(wire, "json_loads", differs)
    with pytest.raises(SystemExit, match="does not round-trip"):
        train_pgo.main()
    assert texts == [
        f'{{"py/object": "{Leaf.__module__}.Leaf", "width": 0}}',
    ]


if __name__ == "__main__":
    from treekle.lib.testing.runner import test_main

    test_main(__file__)
