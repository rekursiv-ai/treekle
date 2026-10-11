# Changelog

All notable treekle changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
