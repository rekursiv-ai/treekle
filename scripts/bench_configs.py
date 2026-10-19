# house-ignore[cli-shape] -- Runs as `python` in an environment that has priml, msgspec and jsonpickle.
"""Time msgspec, treekle, pickle and jsonpickle on every priml ``exp000`` config.

priml, msgspec and jsonpickle are not treekle dependencies, so run this in an
environment that has all three:

  python scripts/bench_configs.py
  python scripts/bench_configs.py --priml my.vendored.priml --columns msgspec pickle

Encode times each library on the config itself; msgspec's ``to_builtins`` takes
``enc_hook=repr``, its only way past the callables the configs hold. msgspec
cannot decode the configs at all: its unions may not mix a custom type with
anything but None. So decode times the closest analogue: each config rebuilt as
generated dataclasses holding only plain leaves (a callable, type or dtype
becomes its repr). ``msgspec`` decodes it into a tagged ``msgspec.Struct`` twin,
``msgspec-dataclass`` into the dataclasses themselves. A decode that raises or
does not equal its input prints as ❌, as does a total that includes one;
msgspec-dataclass raises wherever a union holds two dataclasses, which it cannot
tell apart.

Warm times repeat calls on classes each library has already seen, keeping the
fastest. Cold times each library's first call on classes built afresh for that
call, which is where per-class setup (type hints, field layouts, import paths)
is paid; since a config's own classes cannot be rebuilt, cold encode also runs
on the analogue.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import Protocol, cast, get_args, get_origin

import argparse
import dataclasses
import functools
import gc
import importlib
import operator
import pickle
import sys
import time
import timeit
import uuid

from treekle import deserialize, serialize


def main() -> int:
    """Time every library on every config and print the tables.

    Returns:
      code: 0 once the encode, decode and summary tables are printed.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    configs, skipped = _configs(f"{flags.priml}.baselines")
    for label in skipped:
        print(f"skipped {label}: exp000 raised")
    # Treekle caches the class each import path names, so every run needs a
    # module name of its own.
    module = ModuleType(f"treekle_bench_analogue_{uuid.uuid4().hex}")
    sys.modules[module.__name__] = module
    classes: dict[tuple[str, tuple[tuple[str, object], ...]], type] = {}
    codecs = [_CODECS[name]() for name in flags.columns]
    warm = {
        label: _row(
            config,
            _mirror(config, module=module, classes=classes),
            codecs=codecs,
            budget_sec=flags.budget_sec,
            repeat=flags.repeat,
        )
        for label, config in configs
    }
    cold = {
        label: _cold_row(config, codecs=codecs, repeat=flags.cold_repeat)
        for label, config in configs
    }
    encode = _cells(warm, names=flags.columns, index=0)
    decode = _cells(warm, names=flags.columns, index=1)
    print(_render(encode, title="Encode", names=flags.columns))
    print(_render(decode, title="Decode (analogue)", names=flags.columns))
    totals = {
        "warm encode": encode,
        "warm decode": decode,
        "cold encode": _cells(cold, names=flags.columns, index=0),
        "cold decode": _cells(cold, names=flags.columns, index=1),
    }
    print(_summary(totals, names=flags.columns))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    priml: str
    columns: list[str]
    budget_sec: float
    repeat: int
    cold_repeat: int


type _Times = dict[str, tuple[float | None, float | None]]
type _Prepared = tuple[Callable[[], object], Callable[[object], bool]]
type _Classes = dict[tuple[str, tuple[tuple[str, object], ...]], type]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Codec:
    """One library: how it encodes a config, and how it decodes an analogue.

    ``prepare`` takes an analogue and returns a zero-argument decode plus a
    check that the decode reproduced the analogue.
    """

    name: str
    encode: Callable[[object], object]
    prepare: Callable[[object], _Prepared]


class _Msgspec(Protocol):
    def to_builtins(
        self,
        obj: object,
        *,
        enc_hook: Callable[[object], object] | None = None,
    ) -> object: ...
    def convert(
        self,
        obj: object,
        type: object,
        *,
        from_attributes: bool = False,
    ) -> object: ...
    def defstruct(
        self,
        name: str,
        fields: Iterable[tuple[str, object]],
        *,
        tag: bool,
        tag_field: str,
        kw_only: bool,
    ) -> type: ...


class _FromPlain(Protocol):
    def __call__(
        self,
        data: object,
        target: object,
        *,
        allow_imports: bool,
    ) -> object: ...


class _Flattener(Protocol):
    def flatten(self, obj: object) -> object: ...


class _Restorer(Protocol):
    def restore(self, obj: object) -> object: ...


class _Jsonpickle(Protocol):
    Pickler: Callable[..., _Flattener]
    Unpickler: Callable[..., _Restorer]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--priml",
        default="priml",
        help="The importable name of the priml package.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=list(_CODECS),
        default=list(_CODECS),
        help="Libraries to time; ratio columns are relative to msgspec.",
    )
    parser.add_argument(
        "--budget-sec",
        type=float,
        default=0.002,
        help="Least time one timing repeat runs.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Repeats; the fastest counts.",
    )
    parser.add_argument(
        "--cold-repeat",
        type=int,
        default=3,
        help="Cold samples per config and library, each on fresh classes; the fastest counts.",
    )


def _configs(package: str) -> tuple[list[tuple[str, object]], list[str]]:
    """Build every ``exp000`` under ``package``; return them and those that raised."""
    module = importlib.import_module(package)
    baselines = Path(next(iter(cast("Iterable[str]", module.__path__))))
    configs: list[tuple[str, object]] = []
    skipped: list[str] = []
    for path in sorted(baselines.rglob("experiments*.py")):
        if path.name.endswith("_test.py") or "def exp000(" not in path.read_text():
            continue
        name = ".".join([package, *path.relative_to(baselines).with_suffix("").parts])
        label = str(path.relative_to(baselines).with_suffix("")).removesuffix(
            "/experiments",
        )
        try:
            module = importlib.import_module(name)
            configs.append((label, cast("Callable[[], object]", module.exp000)()))
        except Exception:  # noqa: BLE001 -- A baseline that cannot build is skipped.
            skipped.append(label)
    return configs, skipped


def _mirror(value: object, *, module: ModuleType, classes: _Classes) -> object:
    """Rebuild ``value`` from generated dataclasses in ``module`` and plain leaves."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _mirror(
                cast("object", getattr(value, field.name)),
                module=module,
                classes=classes,
            )
            for field in dataclasses.fields(value)
        }
        annotations = tuple((name, _annotation(item)) for name, item in fields.items())
        key = (type(value).__name__, annotations)
        if key not in classes:
            made = dataclasses.make_dataclass(
                f"{key[0].replace('.', '_')}_{len(classes)}",
                annotations,
                module=module.__name__,
                slots=True,
                kw_only=True,
            )
            setattr(module, made.__name__, made)
            classes[key] = made
        return cast("object", classes[key](**fields))
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        items = [_mirror(item, module=module, classes=classes) for item in sequence]
        return items if isinstance(sequence, list) else tuple(items)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {
            key if type(key) in {str, int} else repr(key): _mirror(
                item,
                module=module,
                classes=classes,
            )
            for key, item in mapping.items()
        }
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return repr(value)


def _annotation(value: object) -> object:
    """Return the narrowest type of a mirrored value that msgspec can decode."""
    if value is None:
        return type(None)
    if isinstance(value, list):
        return GenericAlias(list, (_union(cast("list[object]", value)),))
    if isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        return GenericAlias(tuple, tuple(map(_annotation, items)))
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return GenericAlias(dict, (_union(mapping), _union(mapping.values())))
    return type(value)


def _union(values: Iterable[object]) -> object:
    """Join the values' types; ``object`` where msgspec forbids the union."""
    members: list[object] = []
    for value in values:
        annotation = _annotation(value)
        if annotation not in members:
            members.append(annotation)
    origins = [get_origin(member) for member in members]
    arrays = sum(origin in {list, tuple} for origin in origins)
    mappings = origins.count(dict) + sum(
        isinstance(member, type) and dataclasses.is_dataclass(member)
        for member in members
    )
    if not members or arrays > 1 or (dict in origins and mappings > 1):
        return object
    return functools.reduce(_join, members)


def _join(left: object, right: object) -> object:
    """Return the union type ``left | right``."""
    return cast("Callable[[object, object], object]", operator.or_)(left, right)


def _row(
    config: object,
    analogue: object,
    *,
    codecs: list[_Codec],
    budget_sec: float,
    repeat: int,
) -> _Times:
    """Time each codec's encode of ``config`` and decode of ``analogue``."""
    times: _Times = {}
    for codec in codecs:
        decode, check = codec.prepare(analogue)
        encode = functools.partial(codec.encode, config)
        times[codec.name] = (
            _best_us(encode, budget_sec=budget_sec, repeat=repeat),
            _best_us(decode, budget_sec=budget_sec, repeat=repeat)
            if _decodes(decode, check)
            else None,
        )
    return times


def _cold_row(config: object, *, codecs: list[_Codec], repeat: int) -> _Times:
    """Time each codec's first encode and first decode on freshly built classes."""
    times: _Times = {}
    for codec in codecs:
        encodes: list[float] = []
        decodes: list[float | None] = []
        for _ in range(repeat):
            encode_us, _ = _once(functools.partial(codec.encode, _fresh(config)))
            encodes.append(encode_us)
            decode, check = codec.prepare(_fresh(config))
            decodes.append(_cold_decode_us(decode, check))
        failed = any(us is None for us in decodes)
        times[codec.name] = (
            min(encodes),
            None if failed else min(us for us in decodes if us is not None),
        )
    return times


def _fresh(config: object) -> object:
    """Mirror ``config`` into classes no library has seen, in a module of its own."""
    module = ModuleType(f"treekle_bench_cold_{uuid.uuid4().hex}")
    sys.modules[module.__name__] = module
    return _mirror(config, module=module, classes={})


def _cold_decode_us(
    decode: Callable[[], object],
    check: Callable[[object], bool],
) -> float | None:
    """Time one ``decode``; ``None`` if it raises or does not reproduce its input."""
    try:
        us, result = _once(decode)
        return us if check(result) else None
    except TypeError:
        return None


# A collection triggered by the call's own allocations would charge it for every
# object the run has built so far, so the collector waits until the call ends.
def _once(fn: Callable[[], object]) -> tuple[float, object]:
    """Return the microseconds one call to ``fn`` takes, and its result."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        result = fn()
        return (time.perf_counter() - start) * 1e6, result
    finally:
        if enabled:
            gc.enable()


def _decodes(decode: Callable[[], object], check: Callable[[object], bool]) -> bool:
    """Return whether ``decode`` runs and reproduces its input."""
    try:
        return check(decode())
    except TypeError:
        return False


def _best_us(fn: Callable[[], object], *, budget_sec: float, repeat: int) -> float:
    """Return the fastest per-call microseconds over ``repeat`` timed batches."""
    timer = timeit.Timer(fn)
    number = 1
    while (elapsed := timer.timeit(number)) < budget_sec:
        number *= 2
    batches = [elapsed, *timer.repeat(repeat=repeat - 1, number=number)]
    return min(batches) / number * 1e6


def _cells(
    rows: dict[str, _Times],
    *,
    names: list[str],
    index: int,
) -> dict[str, dict[str, float | None]]:
    """Return ``{config: {library: µs}}`` for one direction, plus a total row."""
    cells = {
        label: {n: times[n][index] for n in names} for label, times in rows.items()
    }
    cells["**total**"] = {
        n: None
        if any(row[n] is None for row in cells.values())
        else sum(row[n] or 0.0 for row in cells.values())
        for n in names
    }
    return cells


def _render(
    cells: dict[str, dict[str, float | None]],
    *,
    title: str,
    names: list[str],
) -> str:
    """Render one markdown table: microseconds, then each time vs msgspec's."""
    relative = [n for n in names if n != "msgspec"] if "msgspec" in names else []
    header = ["config", *names, *(f"{name} vs msgspec" for name in relative)]
    rows = [
        [
            label,
            *(_cell(row[n], "{:.1f}") for n in names),
            *(_cell(_ratio(row[n], row["msgspec"]), "{:.1f}") for n in relative),
        ]
        for label, row in cells.items()
    ]
    return f"**{title}** (µs per call)\n\n" + _table(header, rows)


def _summary(
    totals: dict[str, dict[str, dict[str, float | None]]],
    *,
    names: list[str],
) -> str:
    """Render the totals: one row per library, ms then each time vs msgspec's."""
    base = "msgspec" in names
    header = ["library"]
    for kind in totals:
        header += [f"{kind} (ms)", "×"] if base else [f"{kind} (ms)"]
    rows: list[list[str]] = []
    for name in sorted(names, key=list(_CODECS).index):
        row = [f"{name} (reference)" if base and name == "msgspec" else name]
        for cells in totals.values():
            total = cells["**total**"]
            us = total[name]
            row.append(_cell(None if us is None else us / 1000, "{:.2f}"))
            if base:
                multiple = _ratio(us, total["msgspec"])
                row.append("" if multiple is None else f"{multiple:.1f}")
        rows.append(row)
    note = "; × is the multiple of msgspec's" if base else ""
    title = f"**Summary** (total over every config{note})"
    return f"{title}\n\n" + _table(header, rows)


# The first column is left-aligned and the rest, numbers, right-aligned; each column is
# as wide as its widest cell.
def _table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table padded to read aligned as raw text."""
    widths = [max(map(len, column)) for column in zip(header, *rows, strict=True)]

    def line(cells: list[str]) -> str:
        padded = [
            cell.ljust(width) if index == 0 else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = (
        "|"
        + "|".join(
            "-" * (width + 2) if index == 0 else "-" * (width + 1) + ":"
            for index, width in enumerate(widths)
        )
        + "|"
    )
    return "\n".join([line(header), rule, *map(line, rows)]) + "\n"


def _ratio(value: float | None, base: float | None) -> float | None:
    """Return how many times ``base`` ``value`` takes."""
    if value is None or base is None or base == 0:
        return None
    return value / base


def _cell(value: float | None, spec: str) -> str:
    return "❌" if value is None else spec.format(value)


def _treekle_codec(
    name: str,
    *,
    to_plain: Callable[[object], object],
    from_plain: _FromPlain,
) -> _Codec:
    prepare = functools.partial(
        _prepare_treekle,
        to_plain=to_plain,
        from_plain=from_plain,
    )
    return _Codec(name=name, encode=to_plain, prepare=prepare)


def _prepare_treekle(
    analogue: object,
    *,
    to_plain: Callable[[object], object],
    from_plain: _FromPlain,
) -> _Prepared:
    tree, target = to_plain(analogue), type(analogue)
    return lambda: from_plain(tree, target, allow_imports=True), analogue.__eq__


def _pickle_codec() -> _Codec:
    return _Codec(name="pickle", encode=pickle.dumps, prepare=_prepare_pickle)


def _prepare_pickle(analogue: object) -> _Prepared:
    data = pickle.dumps(analogue)
    return lambda: cast("object", pickle.loads(data)), analogue.__eq__  # noqa: S301 -- Loads the bytes it pickled a line above.


def _jsonpickle_codec() -> _Codec:
    library = cast("_Jsonpickle", cast("object", importlib.import_module("jsonpickle")))
    return _Codec(
        name="jsonpickle",
        encode=lambda config: library.Pickler(keys=True).flatten(config),
        prepare=functools.partial(_prepare_jsonpickle, library=library),
    )


def _prepare_jsonpickle(analogue: object, *, library: _Jsonpickle) -> _Prepared:
    flat = library.Pickler(keys=True).flatten(analogue)
    return lambda: library.Unpickler(keys=True).restore(flat), analogue.__eq__


def _msgspec_codec() -> _Codec:
    library = cast("_Msgspec", cast("object", importlib.import_module("msgspec")))
    return _Codec(
        name="msgspec",
        encode=lambda config: library.to_builtins(config, enc_hook=repr),
        prepare=functools.partial(_prepare_msgspec, library=library, structs={}),
    )


def _msgspec_dataclass_codec() -> _Codec:
    library = cast("_Msgspec", cast("object", importlib.import_module("msgspec")))
    return _Codec(
        name="msgspec-dataclass",
        encode=lambda config: library.to_builtins(config, enc_hook=repr),
        prepare=functools.partial(_prepare_msgspec_dataclass, library=library),
    )


def _prepare_msgspec_dataclass(analogue: object, *, library: _Msgspec) -> _Prepared:
    decode = functools.partial(
        library.convert,
        library.to_builtins(analogue),
        type(analogue),
    )
    return decode, analogue.__eq__


def _prepare_msgspec(
    analogue: object,
    *,
    library: _Msgspec,
    structs: dict[object, type],
) -> _Prepared:
    target: object = _struct(type(analogue), library=library, structs=structs)
    built = _tagged(analogue)
    decode = functools.partial(library.convert, built, target)
    return decode, lambda struct: library.to_builtins(struct) == built


def _tagged(value: object) -> object:
    """Return ``value`` as builtins, each dataclass a dict tagged with its Struct name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _tagged(cast("object", getattr(value, field.name)))
            for field in dataclasses.fields(value)
        }
        return {"$type": type(value).__name__, **fields}
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        items = [_tagged(item) for item in sequence]
        return items if isinstance(sequence, list) else tuple(items)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {key: _tagged(item) for key, item in mapping.items()}
    return value


def _struct(
    annotation: object,
    *,
    library: _Msgspec,
    structs: dict[object, type],
) -> object:
    """Swap every generated dataclass in ``annotation`` for its tagged Struct twin."""
    recurse = functools.partial(_struct, library=library, structs=structs)
    if isinstance(annotation, UnionType):
        return functools.reduce(_join, map(recurse, get_args(annotation)))
    if isinstance(annotation, GenericAlias):
        arguments = tuple(map(recurse, get_args(annotation)))
        return GenericAlias(get_origin(annotation), arguments)
    if annotation in structs:
        return structs[annotation]
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        structs[annotation] = library.defstruct(
            annotation.__name__,
            [(f.name, recurse(f.type)) for f in dataclasses.fields(annotation)],
            tag=True,
            tag_field="$type",
            kw_only=True,
        )
        return structs[annotation]
    return cast("object", annotation)


# Listed in the order the summary tables print them.
_CODECS: dict[str, Callable[[], _Codec]] = {
    "msgspec-dataclass": _msgspec_dataclass_codec,
    "msgspec": _msgspec_codec,
    "treekle-rust": functools.partial(
        _treekle_codec,
        "treekle-rust",
        to_plain=serialize.to_plain,
        from_plain=deserialize.from_plain,
    ),
    "pickle": _pickle_codec,
    "treekle-python": functools.partial(
        _treekle_codec,
        "treekle-python",
        to_plain=serialize._python_to_plain,  # noqa: SLF001 -- The Python walker has no public name.
        from_plain=deserialize._python_from_plain,  # noqa: SLF001 -- The Python walker has no public name.
    ),
    "jsonpickle": _jsonpickle_codec,
}


if __name__ == "__main__":
    raise SystemExit(main())
