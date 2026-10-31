"""Pharmagen library builder.

Builds the offline artifacts the training pipeline loads: per-drug graphs and
the HGVS variant catalog. Use the orchestrator from code:

    >>> from src.data.library import LibraryBuilder, LibraryBuildConfig
    >>> cfg = LibraryBuildConfig.from_settings()
    >>> LibraryBuilder(cfg).run()

Or the CLI:

    $ python -m src.data.library --help

The exports are resolved lazily so importing a featurizer submodule (e.g.
``src.data.library.consequence``) doesn't pull in the whole builder stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.library.builder import LibraryBuilder
    from src.data.library.config import LibraryBuildConfig

__all__ = ["LibraryBuildConfig", "LibraryBuilder"]


def __getattr__(name: str) -> Any:
    if name == "LibraryBuilder":
        from src.data.library.builder import LibraryBuilder

        return LibraryBuilder
    if name == "LibraryBuildConfig":
        from src.data.library.config import LibraryBuildConfig

        return LibraryBuildConfig
    raise AttributeError(name)
