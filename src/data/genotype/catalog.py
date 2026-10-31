"""The HGVS variant catalog: how genotype labels link to HGVS variants.

HGVS is the key of the genotype tower. Everything a training row or an
inference request can name is translated to genomic HGVS (``NC_*:g.``) here:

* ``annotations`` — PharmGKB annotation id → the variants it actually studied
  (KG ``IMPLICATES`` edges: allele-specific, the most precise link);
* ``labels`` — allele label → its variants: rsIDs (every alternative at the
  site) and star alleles ``GENE*N`` (their *cis* variant set);
* ``variants`` — HGVS → annotation (gene, SO consequence terms, protein HGVS);
* ``functions`` — star-allele PGx function (no/decreased/normal/increased).

Sources, later ones winning on conflict: the pan-gene SNP table, PharmVar
haplotype definitions, and the knowledge-graph exports
(``python -m src.kg export`` — canonical NCBI SPDI/HGVS on GRCh38).
Built by ``python -m src.data.library``; saved as ``hgvs_catalog.pt``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import torch

from src.config import get_settings
from src.data.library.haplotype_function import HaplotypeFunctionProvider

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

HGVS_CATALOG_FILE = "hgvs_catalog.pt"
_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VariantInfo:
    """What the catalog knows about one HGVS variant."""

    gene: str | None = None
    so_terms: tuple[str, ...] = ()
    p_hgvs: str | None = None


@dataclass
class HgvsCatalog:
    """Label/annotation → HGVS links plus per-variant annotation."""

    variants: dict[str, VariantInfo] = field(default_factory=dict)
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict)
    annotations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    functions: HaplotypeFunctionProvider = field(
        default_factory=HaplotypeFunctionProvider.null
    )

    def info(self, hgvs: str) -> VariantInfo:
        return self.variants.get(hgvs, VariantInfo())

    def genes(self) -> list[str]:
        """Genes with at least one catalogued variant."""
        return sorted({i.gene for i in self.variants.values() if i.gene})

    def variants_of(self, gene: str) -> list[str]:
        return sorted(h for h, i in self.variants.items() if i.gene == gene)

    # ----- persistence ----------------------------------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": _SCHEMA_VERSION,
                "variants": self.variants,
                "labels": self.labels,
                "annotations": self.annotations,
                "functions": self.functions,
            },
            path,
        )
        logger.info(
            "HGVS catalog saved to %s (%d variants, %d labels, %d annotations)",
            path,
            len(self.variants),
            len(self.labels),
            len(self.annotations),
        )

    @classmethod
    def load(cls, path: Path) -> HgvsCatalog:
        # Local artifact written by this project (same trust boundary as checkpoints).
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            variants=blob["variants"],
            labels=blob["labels"],
            annotations=blob["annotations"],
            functions=blob["functions"],
        )


def load_hgvs_catalog(path: Path | None = None) -> HgvsCatalog:
    """Load ``paths.library/hgvs_catalog.pt`` (empty catalog, with a warning, if absent)."""
    path = path or get_settings().paths.library / HGVS_CATALOG_FILE
    if not path.exists():
        logger.warning(
            "HGVS catalog not found at %s; genotypes will only resolve from explicit "
            "HGVS. Build it with 'python -m src.data.library --skip-drugs'",
            path,
        )
        return HgvsCatalog()
    return HgvsCatalog.load(path)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _add_snp_table(catalog: HgvsCatalog, snp_table: Path) -> None:
    from src.data.library.ingest import snp_table as snp

    if not snp_table.exists():
        logger.warning("SNP table %s not found; skipped", snp_table)
        return
    for v in snp.iter_variants_from_snp_table(snp_table):
        catalog.variants[v.g_hgvs] = VariantInfo(v.gene, v.so_terms, v.p_hgvs)
    for rsid, hgvs in snp.rsid_hgvs_index(snp_table).items():
        catalog.labels[rsid] = (hgvs,)


def _add_pharmvar(catalog: HgvsCatalog, pharmvar_dir: Path) -> None:
    from src.data.library.ingest import pharmvar

    for h in pharmvar.iter_haplotypes(pharmvar_dir):
        catalog.labels[f"{h.gene}{h.label}"] = tuple(v.g_hgvs for v in h.variants)
        for v in h.variants:
            known = catalog.variants.get(v.g_hgvs, VariantInfo())
            catalog.variants[v.g_hgvs] = VariantInfo(
                gene=known.gene or v.gene,
                so_terms=known.so_terms or v.so_terms,
                p_hgvs=known.p_hgvs or v.p_hgvs,
            )
    for rsid, hgvs in pharmvar.rsid_hgvs_index(pharmvar_dir).items():
        catalog.labels.setdefault(rsid, (hgvs,))


def _add_kg(catalog: HgvsCatalog, kg_dir: Path) -> None:
    """KG exports: canonical HGVS per allele label and per annotation."""
    alleles = kg_dir / "allele_variants.parquet"
    if alleles.exists():
        links: dict[str, list[str]] = defaultdict(list)
        rows = pl.read_parquet(alleles).filter(pl.col("hgvs_g").is_not_null())
        for label, hgvs, kind in rows.select("label", "hgvs_g", "kind").iter_rows():
            if hgvs not in links[label]:
                links[label].append(hgvs)
            if kind == "star":
                gene = label.split("*", 1)[0]
                known = catalog.variants.get(hgvs, VariantInfo())
                catalog.variants[hgvs] = VariantInfo(
                    known.gene or gene, known.so_terms, known.p_hgvs
                )
        catalog.labels.update({k: tuple(v) for k, v in links.items()})
    else:
        logger.warning("KG export %s not found; run 'python -m src.kg export'", alleles)

    implicated = kg_dir / "annotation_variants.parquet"
    if implicated.exists():
        by_annotation: dict[str, list[str]] = defaultdict(list)
        rows = pl.read_parquet(implicated).filter(pl.col("hgvs_g").is_not_null())
        for annotation_id, hgvs in rows.select("annotation_id", "hgvs_g").iter_rows():
            by_annotation[str(annotation_id)].append(hgvs)
        catalog.annotations = {k: tuple(v) for k, v in by_annotation.items()}


def build_hgvs_catalog(
    *,
    snp_table: Path,
    pharmvar_dir: Path,
    kg_dir: Path,
    star_alleles_tsv: Path,
) -> HgvsCatalog:
    """Assemble the catalog from every available source (missing ones are skipped)."""
    catalog = HgvsCatalog()
    _add_snp_table(catalog, snp_table)
    _add_pharmvar(catalog, pharmvar_dir)
    _add_kg(catalog, kg_dir)
    if star_alleles_tsv.exists():
        catalog.functions = HaplotypeFunctionProvider.from_tsv(star_alleles_tsv)
    logger.info(
        "HGVS catalog built: %d variants (%d genes), %d labels, %d annotations",
        len(catalog.variants),
        len(catalog.genes()),
        len(catalog.labels),
        len(catalog.annotations),
    )
    return catalog


__all__ = [
    "HGVS_CATALOG_FILE",
    "HgvsCatalog",
    "VariantInfo",
    "build_hgvs_catalog",
    "load_hgvs_catalog",
]
