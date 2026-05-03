"""Pure-Python file organization for the gene-graph library.

The previous implementation shelled out to bash / PowerShell scripts to
sort flat ``GENE_<variant>.pt`` files into per-gene subdirs. This module
does the same with ``pathlib.Path.rename`` — no shell, OS-portable, testable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path


logger = logging.getLogger(__name__)


_UGT1A_PATTERN = re.compile(r"^UGT1A([1-9]|10)$")


def organize_gene_files(graph_dir: Path) -> dict[str, int]:
    """Move flat ``GENE_<variant>.pt`` files into ``GENE/<...>.pt`` subdirs.

    UGT1A* sub-isoforms are merged into a single ``UGT1A`` directory to
    match the project's gene-family grouping convention.

    Returns a counter ``{gene_dir: files_moved}`` for logging.
    """
    if not graph_dir.exists():
        logger.warning("Gene graph dir not found: %s", graph_dir)
        return {}

    moved: dict[str, int] = {}

    for pt in graph_dir.glob("*.pt"):
        if not pt.is_file():
            continue
        prefix = pt.stem.split("_", 1)[0]
        target_gene = "UGT1A" if _UGT1A_PATTERN.match(prefix) else prefix
        if not target_gene:
            logger.debug("Skipping unparseable filename: %s", pt.name)
            continue

        target_dir = graph_dir / target_gene
        target_dir.mkdir(exist_ok=True)
        destination = target_dir / pt.name
        try:
            pt.rename(destination)
            moved[target_gene] = moved.get(target_gene, 0) + 1
        except OSError as e:
            logger.warning("Failed to move %s → %s: %s", pt, destination, e)

    if moved:
        total = sum(moved.values())
        logger.info("Organized %d files across %d gene dirs.", total, len(moved))
    return moved
