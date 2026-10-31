"""Genotype tower input: genotypes read as HGVS variant graphs.

* :mod:`.catalog`  — links rsIDs, star alleles and PharmGKB annotations to HGVS.
* :mod:`.features` — node features read off the HGVS notation.
* :mod:`.graph`    — builds the gene-hub + variant graph for one row.
"""

from src.data.genotype.catalog import (
    HGVS_CATALOG_FILE,
    HgvsCatalog,
    VariantInfo,
    build_hgvs_catalog,
    load_hgvs_catalog,
)
from src.data.genotype.features import HGVS_NODE_DIM
from src.data.genotype.graph import (
    GENO_EDGE_DIM,
    GenotypeGraphBuilder,
    build_gene_vocab,
    gene_vocab_size,
)

__all__ = [
    "GENO_EDGE_DIM",
    "HGVS_CATALOG_FILE",
    "HGVS_NODE_DIM",
    "GenotypeGraphBuilder",
    "HgvsCatalog",
    "VariantInfo",
    "build_gene_vocab",
    "build_hgvs_catalog",
    "gene_vocab_size",
    "load_hgvs_catalog",
]
