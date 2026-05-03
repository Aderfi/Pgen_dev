"""Pharmagen library builder.

Builds the offline graph library that the training pipeline lazy-loads from disk.
Use the orchestrator from code:

    >>> from src.data.library import LibraryBuilder, LibraryBuildConfig
    >>> cfg = LibraryBuildConfig.from_settings()
    >>> LibraryBuilder(cfg).run()

Or the CLI:

    $ python -m src.data.library --help
"""

from src.data.library.builder import LibraryBuilder
from src.data.library.config import LibraryBuildConfig

__all__ = ["LibraryBuilder", "LibraryBuildConfig"]
