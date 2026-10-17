# house-ignore[cli-shape] -- Runs as `python -P` against an installed wheel, outside any project.
"""Check that the installed treekle runs the implementation its wheel should.

Run as ``python -P -W error scripts/check_wheel.py native|python`` in an
environment with the wheel installed. ``-P`` keeps the checkout off
``sys.path``; ``-W error`` turns the Rust core's fallback warning into a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util

import sys

from treekle import deserialize, serialize


@dataclass(
    frozen=True,
)  # house-ignore[dataclass] -- Built positionally, as the wheel's users would write it.
class _Point:
    x: int
    y: float


def main() -> int:
    """Check that treekle runs the implementation its wheel should.

    Returns:
      code: 0 when both functions are the expected implementation and a
        dataclass round-trips; any failure raises ``SystemExit`` with a message.

    """
    if len(sys.argv) != 2 or sys.argv[1] not in {"native", "python"}:
        raise SystemExit("usage: check_wheel.py native|python")
    native = sys.argv[1] == "native"
    core = util.find_spec("treekle._rust")
    if native != (core is not None):
        raise SystemExit(f"treekle._rust is {'missing' if native else 'present'}")
    if native:
        from treekle import _rust  # noqa: PLC0415 -- Only a native wheel has it.

        expected = (_rust.to_plain, _rust.from_plain)
    else:
        expected = (serialize._python_to_plain, deserialize._python_from_plain)  # noqa: SLF001 -- The Python walker has no public name.
    actual = (serialize.to_plain, deserialize.from_plain)
    if any(found is not wanted for found, wanted in zip(actual, expected, strict=True)):
        raise SystemExit(f"treekle does not run its {sys.argv[1]} implementation")
    shape = _Shape("square", (_Point(0, 0.0), _Point(1, 0.5)))
    if deserialize.from_plain(serialize.to_plain(shape), _Shape) != shape:
        raise SystemExit("a dataclass does not round-trip")
    print(f"{sys.argv[1]}: {core.origin if core else serialize.__file__}")
    return 0


@dataclass(
    frozen=True,
)  # house-ignore[dataclass] -- Built positionally, as the wheel's users would write it.
class _Shape:
    name: str
    corners: tuple[_Point, ...]


if __name__ == "__main__":
    raise SystemExit(main())
