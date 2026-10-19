# Releasing treekle to PyPI

Maintainer-only. Replace `X.Y.Z` with the new version throughout.

A published GitHub Release triggers `.github/workflows/publish-pypi.yml`,
which builds, validates, and uploads to PyPI via OIDC trusted publishing
(no API token). Follow the steps in order.

Each native wheel is profile-guided: the workflow builds an instrumented
wheel, runs `scripts/train_pgo.py` against it, and builds the wheels it ships
from the profile that run recorded. A local `uv build` skips that, so its
wheel is correct but not as fast as the release's.

Profiling needs `llvm-profdata`: `rustup component add llvm-tools`.

## Steps

1. **(First release only) Register the trusted publisher on PyPI.**
   Skip if already done. Web-console only, no CLI. Go to
   https://pypi.org/manage/project/treekle/settings/publishing/ and add
   a publisher with these exact values:

   | Field | Value |
   |---|---|
   | Owner | `rekursiv-ai` |
   | Repository name | `treekle` |
   | Workflow filename | `publish-pypi.yml` |
   | Environment name | `pypi` |

   Environment name **must** be `pypi` (matches `publish-pypi.yml:115`).
   Blank here = `invalid-publisher` at publish time.

2. **Bump the version** in `pyproject.toml` (source of truth). Must
   increase; PyPI rejects re-uploads.

3. **Refresh the lockfile and validate locally** (same checks CI runs):

   `uv lock` first: the lockfile pins this package's own version, so
   building without it publishes an artifact whose lock still names the
   previous one.

   ```bash
   uv lock
   rm -rf dist
   uv build --out-dir dist
   uv run --isolated --no-project --no-index --no-build --find-links dist --with treekle python -P -W error scripts/check_wheel.py native
   uv run --isolated --no-project --with "$(uv run scripts/build_pure_wheel.py dist)" python -P -W error scripts/check_wheel.py python
   uv run twine check --strict dist/*
   ```

4. **Commit and merge to `main`.** Confirm the published version:
   ```bash
   gh api repos/rekursiv-ai/treekle/contents/pyproject.toml --jq .content | base64 -d | grep '^version'
   ```

5. **Cut the release** (this triggers the PyPI publish):
   ```bash
   gh release create vX.Y.Z --repo rekursiv-ai/treekle --title "vX.Y.Z" --generate-notes
   ```

6. **Watch the workflow:**
   ```bash
   gh run watch --repo rekursiv-ai/treekle $(gh run list --repo rekursiv-ai/treekle --workflow publish-pypi.yml --event release --limit 1 --json databaseId --jq '.[0].databaseId')
   ```

7. **Confirm it's live:**
   ```bash
   curl -s https://pypi.org/pypi/treekle/json | jq -r '.info.version'   # should print X.Y.Z
   ```
   Browser: https://pypi.org/project/treekle/

## If the workflow fails

- **`invalid-publisher`** ("valid token, but no corresponding
  publisher") — the trusted publisher isn't registered or doesn't match.
  Do step 1, then re-run: `gh workflow run publish-pypi.yml --repo rekursiv-ai/treekle`.

- **Transient PyPI failure** — re-run without a new tag:
  `gh workflow run publish-pypi.yml --repo rekursiv-ai/treekle`.

- **Must ship now, publishing still broken** — upload the files the
  release run built (every build job must have passed), with the
  project's API token (`rekursiv-ai`-owned, e.g. `$PYPI_TOKEN_WORK`;
  a personal token can't publish `treekle`). Don't upload a local
  build: PyPI rejects its `linux_x86_64` wheel.
  ```bash
  run=$(gh run list --repo rekursiv-ai/treekle --workflow publish-pypi.yml --event release --limit 1 --json databaseId --jq '.[0].databaseId')
  rm -rf dist
  gh run download "$run" --repo rekursiv-ai/treekle --pattern 'dist-*' --dir dist
  uv run --with twine twine check --strict dist/*/*
  uv run --with twine twine upload dist/*/* -u __token__ -p "$PYPI_TOKEN_WORK"
  ```
  Permanent and public; PyPI never allows re-uploading a version.
