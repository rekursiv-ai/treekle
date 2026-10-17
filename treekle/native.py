"""Choose the Rust core's walker over the Python one, when a matching core exists.

``serialize`` and ``deserialize`` each pass their Python walker and the core's
setup function by name; the core installs itself against that module's globals.
"""

from __future__ import annotations

from typing import cast

import importlib
import os
import warnings


def native[F](python: F, *, name: str, setup: str, module: dict[str, object]) -> F:
    """Return the Rust core's ``name`` set up against ``module``, else ``python``.

    ``TREEKLE_PURE_PYTHON`` set to anything but ``0`` keeps ``python``, as does a
    missing core. A core that imports but fails to set up, as one built from
    older sources does, keeps ``python`` and warns.

    Args:
      python: The Python walker to keep.
      name: The core's walker, ``to_plain`` or ``from_plain``.
      setup: The core's setup function for ``module``.
      module: The calling module's globals.

    Returns:
      walker: The core's walker when it set up, else ``python``.

    """
    if os.environ.get("TREEKLE_PURE_PYTHON", "") not in {"", "0"}:
        return python
    package = cast("str", module["__package__"])
    try:
        core = importlib.import_module(f"{package}._rust")
    except ModuleNotFoundError:
        return python
    # A core built from other sources may lack this module's functions or fail
    # to set up against it; Python must still serve rather than fail import.
    try:
        walker = cast("F", getattr(core, name))
        getattr(core, setup)(module)
    except Exception as error:  # noqa: BLE001 -- Any failure of a stale core falls back to Python.
        warnings.warn(
            f"{package}'s Rust core does not match this {package} ({error}); "
            "using the Python implementation until the core is rebuilt",
            RuntimeWarning,
            stacklevel=3,
        )
        return python
    return walker
