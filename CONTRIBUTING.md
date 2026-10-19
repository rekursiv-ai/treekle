# Contributing to treekle

Thanks for helping improve treekle.

## Why this file exists

treekle accepts public changes through a generated public repository while the source tree stays canonical. Contributors need to know how to validate changes locally and which branches are safe to edit.

## Development setup

Requires Python 3.12, uv, and a Rust toolchain (1.88 or newer) to build the
compiled core in `rust/`. `TREEKLE_PURE_PYTHON=1` runs the pure-Python
implementation instead. The native core does not build for free-threaded
Python; the release's universal wheel installs the Python implementation
without Rust.

```bash
uv sync --all-groups
uv run pytest
```

Release wheels are profile-guided (see `RELEASING.md`), so a local build runs
slower than a published wheel. Measure speed against a published wheel.

Before opening a pull request, run:

```bash
uv sync --all-groups
uv run ruff check --no-fix --no-cache .
uv run ruff format --check --no-cache .
uv run codespell .
uv run ty check
uv run basedpyright treekle
cargo fmt --manifest-path rust/Cargo.toml --check
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --manifest-path rust/Cargo.toml -- -D warnings
uv run pytest
TREEKLE_PURE_PYTHON=1 uv run pytest
uv run python -c "import treekle"
uv build
uv run scripts/build_pure_wheel.py dist
```

## Public contribution flow

The public repository is synchronized with the canonical source tree. Public
changes should be made on normal contributor branches. After validation, the
sync workflow imports accepted changes back to the source repository for review.

Do not edit generated `treekle/export/*` branches directly.

## Pull request expectations

- Keep changes focused.
- Include tests for behavior changes.
- Update README or docs when public behavior changes.
- Do not include secrets, private credentials, generated caches, or local environment files.
