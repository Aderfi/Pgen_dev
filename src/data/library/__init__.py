"""Pharmagen library: Modephere drug graphs, ADMET profiles, the HGVS catalog.

Modephere builds and stores the drugs; dev_Pgen adds what is its own — the
ADMET-AI profile of each compound and the HGVS catalog the genotype tower reads.
:mod:`src.data.library.bridge` is the only place Modephere is configured, from
``Settings.modephere``.

    >>> from src.data.library import LibraryBuilder, LibraryBuildConfig
    >>> LibraryBuilder(LibraryBuildConfig.from_settings()).run()

Or the CLI: ``python -m src.data.library --help``.

Exports resolve lazily, so importing :mod:`src.data.library.admet` does not
pull in the builder stack.
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
