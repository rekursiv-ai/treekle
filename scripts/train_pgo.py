# house-ignore[cli-shape] -- Runs as `python -P` against an installed instrumented wheel, outside any project.
"""Run treekle's common paths, so a profile-guided build learns which are hot.

Run as ``python -P scripts/train_pgo.py`` in an environment with an
instrumented core installed. Each round builds a fresh config: nested slotted
dataclasses holding plain leaves, lists, tuples, dicts and a function, as a
model config does. It is encoded and decoded once while its classes are new,
then again while they are known, so both cold and warm paths are taken.
"""

from __future__ import annotations

from dataclasses import field, make_dataclass
from types import ModuleType
from typing import cast

import sys

from treekle import deserialize, serialize


def main() -> int:
    """Encode and decode fresh configs, cold then warm.

    Returns:
      code: 0 once every config has round-tripped.

    """
    for index in range(200):
        config = _config(index)
        for _ in range(3):
            tree = serialize.to_plain(config)
            if deserialize.from_plain(tree, type(config), allow_imports=True) != config:
                raise SystemExit("a config does not round-trip")
    return 0


def _config(index: int, *, depth_count: int = 12) -> object:
    """Return a config whose classes live in a module of their own."""
    module = ModuleType(f"treekle_pgo_{index}")
    sys.modules[module.__name__] = module
    child: object = None
    parent: object = object
    for depth in range(depth_count):
        kind = make_dataclass(
            f"Layer{depth}",
            [
                ("width", int),
                ("rate", float),
                ("name", str),
                ("enabled", bool),
                ("shape", tuple[int, int]),
                ("sizes", list[int], field(default_factory=list[int])),
                ("extra", dict[str, float], field(default_factory=dict[str, float])),
                ("child", parent, None),
            ],
            module=module.__name__,
            slots=True,
            kw_only=True,
            frozen=True,
        )
        setattr(module, kind.__name__, kind)
        # Each layer declares its child's class, so the records it holds go
        # untagged, as a model config's nested configs do.
        parent = kind | None
        child = cast(
            "object",
            kind(
                width=depth,
                rate=0.5 * depth,
                name=f"layer{depth}",
                enabled=depth % 2 == 0,
                shape=(depth, depth + 1),
                sizes=[depth, depth * 2],
                extra={"scale": 1.5},
                child=child,
            ),
        )
    return child


if __name__ == "__main__":
    raise SystemExit(main())
