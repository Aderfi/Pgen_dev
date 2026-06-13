"""Resolve a training/inference genotype to its encoded subgraph.

Replaces the old ``GenoKeyBuilder`` (``geno_key = GEN_<star|rsID>``). A genotype
string is tokenised: star-allele tokens (``*4``) encode by label (a haplotype, or
a diplotype when two are given); rsID tokens are bridged to genomic HGVS via the
PharmVar ``rsID → HGVS`` index and encoded as an ad-hoc variant path. The library
key is HGVS throughout — the rsID is only a data-prep lookup. Genes absent from
the library resolve to ``None`` so the caller can substitute a placeholder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data.data import Data

    from src.data.library.geno_store import GenoLibrary

logger = logging.getLogger(__name__)


class GenotypeResolver:
    """Map ``(gene, genotype)`` to an encoded path/diplotype subgraph."""

    def __init__(self, library: GenoLibrary, rsid_to_hgvs: dict[str, str]) -> None:
        self.library = library
        self.rsid_to_hgvs = rsid_to_hgvs

    def resolve(self, gene: str, genotype: str) -> Data | None:
        """Encode a genotype, or return ``None`` when the gene is not catalogued.

        Star-allele tokens (``*4``) take the path-label route (one = haplotype,
        two = diplotype); rsID tokens are mapped through ``rsid_to_hgvs`` and
        encoded as an ad-hoc variant set. Unknown rsIDs are dropped.
        """
        if gene not in self.library:
            return None

        tokens = [t.strip() for t in str(genotype).split("|") if t.strip()]
        labels = [t for t in tokens if "*" in t]
        if labels:
            return self.library.encode(gene, labels)

        rsids = [t for t in tokens if t.lower().startswith("rs")]
        hgvs = [self.rsid_to_hgvs[r] for r in rsids if r in self.rsid_to_hgvs]
        return self.library.encode_variants(gene, hgvs)


__all__ = ["GenotypeResolver"]
