#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Count the instructions treekle's Rust core spends on one workload at each ref.

Each ref's treekle package is checked out over the head's tree in a detached
worktree under ``/opt/scratch/runs``, so every core reads the same priml
configs, and built there as the release wheels are (PGO; ``--no-pgo`` skips
the profile). A head of ``WORKTREE`` is this checkout as it stands, uncommitted
edits included, built in place. Each case then runs in its own process under
``perf stat``, pinned to one CPU, counting user instructions only inside the
timed calls. Instruction counts barely move with machine load, so refs compare
on a busy machine where timings do not. The ratio is the head over the ref
just before it.

Every result is checked against Python's own walk (``_python_to_plain`` and
``_python_from_plain``) after the count, so the check warms nothing it counts.
The exit status is 1 if any result differs.

Examples:
  scripts/bench_counts.py
  scripts/bench_counts.py --base c4afb1c292 --head HEAD --cpu 3
  scripts/bench_counts.py --base c4afb1c292 --ref 1dae733a89 --head WORKTREE
  scripts/bench_counts.py --head WORKTREE --no-pgo --only "encode cold"

'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from scripts import bench_configs
from treekle import (
    deserialize,
    from_plain,
    json_dumps,
    json_loads,
    serialize,
    to_plain,
)


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent

_RUNS: Final = Path("/opt/scratch/runs/treekle-bench-counts")
_WORKTREE: Final = "WORKTREE"
"""The ``--head`` naming this checkout as it stands."""
# ``loop.`` in the monorepo, nothing where treekle is its own package.
_PREFIX: Final = serialize.__name__.removesuffix("treekle.serialize")


def main() -> int:
    """Count every case at both refs, or one case in this process.

    Returns:
      code: 0 when every result matches Python's walk, else 1.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    if flags.case is not None:
        calls, mismatched = _measure(_CASES[flags.case]())
        print(f"calls={calls} mismatched={mismatched}")
        return 0
    repo = Path(_git(_CWD, "rev-parse", "--show-toplevel"))
    pgo = not flags.no_pgo
    worktree = flags.head == _WORKTREE
    head = _git(repo, "rev-parse", "HEAD" if worktree else flags.head)
    refs = [_git(repo, "rev-parse", ref) for ref in (flags.base, *flags.ref)]
    if not worktree:
        refs.append(head)
    roots = [_prepared(repo, ref, head, pgo=pgo) for ref in refs]
    names = [ref[:10] for ref in refs]
    if worktree:
        _build(repo, _CWD.parent.relative_to(repo), pgo=pgo)
        roots.append(repo)
        names.append(_WORKTREE)
    rows: list[_Row] = []
    for case in flags.only or _CASES:
        counts = [_count(root, case, flags.cpu) for root in roots]
        rows.append(_Row(case=case, counts=counts))
    builds = "PGO builds" if pgo else "Builds without a profile"
    print(f"{builds}; user instructions on CPU {flags.cpu}; one process per case.")
    print(_render(rows, refs=names))
    return 1 if any(count.mismatched for row in rows for count in row.counts) else 0


class Flags(Protocol):
    """Parsed command-line flags."""

    base: str
    ref: list[str]
    head: str
    cpu: str
    no_pgo: bool
    only: list[str]
    case: str | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--base",
        default="c4afb1c292",
        help="The first ref counted.",
    )
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        help="A ref counted after the base, before the head; may repeat.",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help=f"The ref under test; {_WORKTREE} for this checkout as it stands.",
    )
    parser.add_argument("--cpu", default="2", help="The CPU every case is pinned to.")
    parser.add_argument(
        "--no-pgo",
        action="store_true",
        help="Build every ref without a profile: faster to build, slower to run.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=list(_CASES),
        help="Count only this case; may repeat.",
    )
    parser.add_argument(
        "--case",
        choices=list(_CASES),
        help="Count one case in this process, under a parent's perf (internal).",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Count:
    """One case at one ref."""

    instructions: int
    calls: int
    mismatched: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Row:
    """One case at every ref."""

    case: str
    counts: list[_Count]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Work:
    """The calls a case counts, and Python's answer for each."""

    calls: list[Callable[[], object]]
    references: list[Callable[[], object]]
    rounds: int
    warm: bool


def _git(where: Path, *args: str) -> str:
    """Return the stripped output of ``git -C where args``."""
    done = subprocess.run(  # noqa: S603 -- Fixed argv.
        ["git", "-C", str(where), *args],  # noqa: S607 -- git is on PATH.
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _prepared(repo: Path, ref: str, head: str, *, pgo: bool) -> Path:
    """Return a worktree of ``head`` holding ``ref``'s treekle, its core built."""
    package = _CWD.parent.relative_to(repo)
    root = _RUNS / f"{ref[:12]}-on-{head[:12]}{'' if pgo else '-no-pgo'}"
    if not root.exists():
        _git(repo, "worktree", "add", "--detach", str(root), head)
        shutil.rmtree(root / package)
        _git(root, "checkout", ref, "--", str(package))
        # The workload is the newer ref's, so both cores read the same one.
        workload = package / "scripts" / "bench_configs.py"
        _git(root, "checkout", head, "--", str(workload))
    # The build installs a profiling core before the final one, so only a
    # finished build counts as built.
    built = root / ".bench-counts-built"
    if not built.exists():
        _build(root, package, pgo=pgo)
        built.touch()
    return root


def _build(root: Path, package: Path, *, pgo: bool) -> None:
    """Build ``package``'s core in the checkout at ``root``; with a profile if ``pgo``."""
    subprocess.run(  # noqa: S603 -- Fixed argv.
        [
            sys.executable,
            "-P",
            str(root / package / "build_rust.py"),
            *(() if pgo else ("--no-pgo",)),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
    )


def _count(root: Path, case: str, cpu: str) -> _Count:
    """Run ``case`` against ``root``'s core under ``perf stat``; return its count."""
    with tempfile.TemporaryDirectory(prefix="bench-counts-") as scratch:
        control, ack, out = (Path(scratch) / name for name in ("ctl", "ack", "out"))
        os.mkfifo(control)
        os.mkfifo(ack)
        perf = [
            *("taskset", "-c", cpu, "perf", "stat", "-x,", "-e", "instructions:u"),
            *("-D", "-1", "--control", f"fifo:{control},{ack}", "-o", str(out)),
        ]
        done = subprocess.run(  # noqa: S603 -- Fixed argv.
            [*perf, sys.executable, "-P", str(_THIS), "--case", case],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "PERF_CTL": str(control),
                "PERF_ACK": str(ack),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        calls, mismatched = _read_child(done.stdout)
        return _Count(
            instructions=_read_perf(out.read_text()),
            calls=calls,
            mismatched=mismatched,
        )


def _read_perf(text: str) -> int:
    """Return the instruction count in ``perf stat -x,`` output."""
    # A hybrid CPU names the event per core type (``cpu_core/instructions/u``);
    # the pinned CPU's type is the one that counted.
    found = re.search(r"^(\d+),[^,]*,(?:\w+/)?instructions", text, re.MULTILINE)
    if found is None:
        raise ValueError(f"no instruction count in perf output: {text!r}")
    return int(found[1])


def _read_child(text: str) -> tuple[int, int]:
    """Return the calls and mismatches a ``--case`` run reports."""
    found = re.search(r"calls=(\d+) mismatched=(\d+)", text)
    if found is None:
        raise ValueError(f"no result in case output: {text!r}")
    return int(found[1]), int(found[2])


def _render(rows: list[_Row], *, refs: list[str]) -> str:
    """Return ``rows`` as a table: each ref's count, the ratio, the mismatches."""
    head = f"{'case':<26}{'calls':>7}" + "".join(f"{ref:>14}" for ref in refs)
    lines = [head + f"{'ratio':>8}{'mismatched':>12}"]
    for row in rows:
        counts = "".join(f"{count.instructions / 1e6:>13.1f}M" for count in row.counts)
        ratio = row.counts[-1].instructions / row.counts[-2].instructions
        mismatched = "/".join(str(count.mismatched) for count in row.counts)
        lines.append(
            f"{row.case:<26}{row.counts[-1].calls:>7}{counts}{ratio:>8.3f}{mismatched:>12}",
        )
    return "\n".join(lines)


def _measure(work: _Work) -> tuple[int, int]:
    """Count ``work``'s calls between perf's enable and disable; check them after."""
    if work.warm:
        for call in work.calls:
            call()
    gc.collect()
    gc.disable()
    _perf("enable")
    results = [call() for _ in range(work.rounds) for call in work.calls]
    _perf("disable")
    gc.enable()
    plain = serialize._python_to_plain  # noqa: SLF001 -- The Python walker has no public name.
    expected = [plain(reference()) for reference in work.references]
    mismatched = sum(
        plain(result) != expected[index % len(expected)]
        for index, result in enumerate(results)
    )
    return len(results), mismatched


def _perf(command: str) -> None:
    """Tell the parent's ``perf stat`` to start or stop counting, once it has."""
    control = os.environ.get("PERF_CTL")
    if control is None:
        return
    Path(control).write_text(f"{command}\n")
    # `perf` acts on a command when it reads it, not when it is written: without
    # its ack the count would start late and stop after the checks below.
    ack = os.open(os.environ["PERF_ACK"], os.O_RDONLY)
    try:
        taken = os.read(ack, 16).startswith(b"ack")
    finally:
        os.close(ack)
    if not taken:
        raise RuntimeError(f"perf did not take {command!r}")


def _configs(rounds: int) -> list[object]:
    """Every priml ``exp000`` config, mirrored ``rounds`` times into classes never seen."""
    configs, _ = bench_configs._configs(f"{_PREFIX}priml.baselines")  # noqa: SLF001 -- The bench's own workload, shared rather than copied.
    fresh = bench_configs._fresh  # noqa: SLF001 -- The bench's own workload, shared rather than copied.
    return [fresh(config) for _ in range(rounds) for _, config in configs]


def _encode(values: list[object], *, rounds: int, warm: bool) -> _Work:
    """Count ``to_plain`` of each value."""
    python = serialize._python_to_plain  # noqa: SLF001 -- The Python walker has no public name.
    return _Work(
        calls=[partial(to_plain, value) for value in values],
        references=[partial(python, value) for value in values],
        rounds=rounds,
        warm=warm,
    )


def _dumps(values: list[object], *, rounds: int, warm: bool) -> _Work:
    """Count ``json_dumps`` of each value."""
    return _Work(
        calls=[partial(json_dumps, value) for value in values],
        references=[partial(_python_dumps, value) for value in values],
        rounds=rounds,
        warm=warm,
    )


def _python_dumps(value: object) -> str:
    """Return ``value`` written by Python's walk, then the standard writer."""
    return json.dumps(serialize._python_to_plain(value))  # noqa: SLF001 -- The Python walker has no public name.


def _decode(
    values: list[object],
    *,
    rounds: int,
    warm: bool,
    target: object = None,
) -> _Work:
    """Count ``from_plain`` of each value's tree, as its class unless ``target``."""
    python = deserialize._python_from_plain  # noqa: SLF001 -- The Python walker has no public name.
    cases = [(to_plain(value), target or type(value)) for value in values]
    return _Work(
        calls=[
            partial(from_plain, tree, kind, allow_imports=True) for tree, kind in cases
        ],
        references=[
            partial(python, tree, kind, allow_imports=True) for tree, kind in cases
        ],
        rounds=rounds,
        warm=warm,
    )


def _loads(
    values: list[object],
    *,
    rounds: int,
    warm: bool,
    target: object = None,
) -> _Work:
    """Count ``json_loads`` of each value's text."""
    python = deserialize._python_from_plain  # noqa: SLF001 -- The Python walker has no public name.
    cases = [(json_dumps(value), target or type(value)) for value in values]
    return _Work(
        calls=[
            partial(json_loads, text, kind, allow_imports=True) for text, kind in cases
        ],
        references=[partial(_python_loads, python, text, kind) for text, kind in cases],
        rounds=rounds,
        warm=warm,
    )


def _read(trees: list[object], *, target: object, rounds: int, text: bool) -> _Work:
    """Count ``from_plain``, or ``json_loads`` when ``text``, of each tree as ``target``."""
    python = deserialize._python_from_plain  # noqa: SLF001 -- The Python walker has no public name.
    if text:
        texts = [json.dumps(tree) for tree in trees]
        return _Work(
            calls=[
                partial(json_loads, each, target, allow_imports=True) for each in texts
            ],
            references=[partial(_python_loads, python, each, target) for each in texts],
            rounds=rounds,
            warm=True,
        )
    return _Work(
        calls=[partial(from_plain, tree, target, allow_imports=True) for tree in trees],
        references=[
            partial(python, tree, target, allow_imports=True) for tree in trees
        ],
        rounds=rounds,
        warm=True,
    )


def _python_loads(read: Callable[..., object], text: str, kind: object) -> object:
    """Return ``text`` read by the standard parser, then by Python's walk."""
    return read(json.loads(text), kind, allow_imports=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Slotted:
    a: int
    b: float
    c: str


@dataclass(
    kw_only=True,
)  # house-ignore[dataclass] -- The dict-backed row: its fields live in __dict__.
class _DictBacked:
    a: int
    b: float
    c: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Base:
    a: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Derived(_Base):
    """A row whose class is not straight under ``object``: no plain layout."""

    b: float
    c: str


class _Plain:
    """A class that is no dataclass: written as ``py/object``, read by ``setattr``."""

    def __init__(self, *, a: int, b: float, c: str) -> None:
        self.a = a
        self.b = b
        self.c = c


@dataclass(frozen=True, slots=True, kw_only=True)
class _Defaulted:
    a: int
    b: float = 0.0
    c: str = ""
    tags: tuple[str, ...] = ()


def _rows(kind: Callable[..., object], *, count: int = 2_000) -> list[object]:
    """One list of ``count`` rows of ``kind``."""
    return [[kind(a=index, b=index * 0.5, c=f"r{index}") for index in range(count)]]


def _defaulted_rows(*, count: int = 2_000) -> list[object]:
    """One list of ``count`` ``_Defaulted`` rows' trees, every other one leaving out each field at its default, as ``to_plain`` writes it."""
    return [
        [
            {"a": index}
            if index % 2
            else {"a": index, "b": index * 0.5, "c": f"r{index}", "tags": ["t"]}
            for index in range(count)
        ],
    ]


# Cold cases read 5 rounds of fresh classes once each; warm ones repeat.
_CASES: dict[str, Callable[[], _Work]] = {
    "encode cold": lambda: _encode(_configs(5), rounds=1, warm=False),
    "encode warm": lambda: _encode(_configs(1), rounds=50, warm=True),
    "json_dumps cold": lambda: _dumps(_configs(5), rounds=1, warm=False),
    "json_dumps warm": lambda: _dumps(_configs(1), rounds=50, warm=True),
    "decode cold": lambda: _decode(_configs(5), rounds=1, warm=False),
    "decode warm": lambda: _decode(_configs(1), rounds=50, warm=True),
    "json_loads cold": lambda: _loads(_configs(5), rounds=1, warm=False),
    "json_loads warm": lambda: _loads(_configs(1), rounds=50, warm=True),
    "slotted rows decode": lambda: _decode(
        _rows(_Slotted),
        rounds=20,
        warm=True,
        target=list[_Slotted],
    ),
    "slotted rows json_loads": lambda: _loads(
        _rows(_Slotted),
        rounds=20,
        warm=True,
        target=list[_Slotted],
    ),
    "dict rows decode": lambda: _decode(
        _rows(_DictBacked),
        rounds=20,
        warm=True,
        target=list[_DictBacked],
    ),
    "dict rows encode": lambda: _encode(_rows(_DictBacked), rounds=20, warm=True),
    "derived rows encode": lambda: _encode(_rows(_Derived), rounds=20, warm=True),
    "derived rows decode": lambda: _decode(
        _rows(_Derived),
        rounds=20,
        warm=True,
        target=list[_Derived],
    ),
    "plain objects decode": lambda: _decode(
        _rows(_Plain),
        rounds=20,
        warm=True,
        target=list[_Plain],
    ),
    "defaulted rows decode": lambda: _read(
        _defaulted_rows(),
        target=list[_Defaulted],
        rounds=20,
        text=False,
    ),
    "defaulted rows json_loads": lambda: _read(
        _defaulted_rows(),
        target=list[_Defaulted],
        rounds=20,
        text=True,
    ),
}


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
