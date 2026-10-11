# Changelog

All notable treekle changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added

- `to_plain` and `from_plain`: lossless Python values to and from plain trees,
  with classes, functions, shared references, and cycles; `py/ref`
  back-references by path; best-effort decoding with `ReadError.partial` and
  `ReadError.bad`.
- `parse`, `immutable`, and `mutable` helpers; `wire.json_loads`.
