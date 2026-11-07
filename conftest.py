"""Export-root pytest resource setup for treekle.

Binds the resource-marker hook and the per-user-directory isolation at the
repository root, the one conftest every exported test sits under. Without it the
vendored ``resources_test`` meets an unregistered ``real_user_dirs`` marker and
fails at collection. It also loads the Hypothesis budgets the monorepo's root
conftest loads, before any property test draws.
"""

from typing import Final

from treekle.lib.testing.hypothesis_profiles import configure_hypothesis
from treekle.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


# Honored only in the rootdir conftest: each invocation claims its own basetemp,
# so concurrent sessions never clean up one another's directories.
pytest_plugins: Final = ("treekle.lib.testing.basetemp",)

__all__ = ["isolate_user_dirs", "pytest_collection_modifyitems", "pytest_configure"]


configure_hypothesis()
