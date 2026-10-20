# Changelog

All notable treekle changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Neither wheel ships `treekle.lib.testing`, whose modules import `pytest`, a
  test-only dependency.
- Set members order by their encoding, then by `repr`; a set whose members
  `repr` with their address no longer writes differently from run to run.
  Stored sets of strings that need escaping may reorder.
- A strict `from_plain(text, float)` refuses `"inf"`, `"nan"`, and other
  float text, as it refuses all scalar text; `{"py/float": ...}` reads only
  the `repr` of a non-finite float.

### Fixed

- A private slot (`__x`) is written and read under its mangled name instead
  of being dropped, also on a class renamed after it was created.
- `ReadError` and its subclasses pickle as themselves, with `partial`, `bad`,
  and any attributes a subclass set.
- `PurePosixPath`, `PureWindowsPath`, and every other `PurePath` class read
  as that class.
- An encode hook returning the type it encodes raises `TypeError` instead of
  looping forever; a decode hook raising `TypeError` or `ValueError` is a bad
  part of `ReadError`.
- An import path cached for a class or function is rechecked, so a name
  rebound or reloaded since, on its module or on a class it is nested in, is
  not written or read stale.
- `json_dumps` with a huge `indent` raises `MemoryError` as `json.dumps` does,
  where the Rust core aborted the process.

## 0.2.0 - 2026-10-17

### Added

- A Rust core for `to_plain` and `from_plain`, shipped as wheels for Linux
  and macOS (x86_64, aarch64) and Windows (x64) on CPython 3.12-3.14. Other
  platforms install a pure-Python wheel with the same behavior; set
  `TREEKLE_PURE_PYTHON=1` to force it.
- `ABSENT` and `Absent` are exported from the package root.
- `to_plain(..., sort=False)` keeps attributes in declaration order.

### Changed

- Encoding, decoding, and rendering are iterative, so deeply nested values
  no longer raise `RecursionError`.

## 0.1.1 - 2026-10-11

### Added

- `json_loads(text, T)` reads JSON straight into a typed value, and is
  exported from the package root.

### Fixed

- Decoding checks every nested field and element against its annotation, and
  rejects a payload-named class that does not fit its field.
- Shared references are validated against the target type; dataclass state is
  restored without rerunning `__init__`; mutable reduced values and mapping
  proxies keep shared identity.
- Values with no importable module are rejected when written.
- Reduce recipes are canonical and set members encode in a stable order.

## 0.1.0 - 2026-10-10

### Added

- `to_plain` and `from_plain`: lossless Python values to and from plain trees,
  with classes, functions, shared references, and cycles; `py/ref`
  back-references by path; best-effort decoding with `ReadError.partial` and
  `ReadError.bad`.
- `parse`, `immutable`, and `mutable` helpers; `wire.json_loads`.
