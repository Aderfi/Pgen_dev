"""The HGVS variant catalog: how genotype labels link to HGVS variants.

HGVS is the key of the genotype tower. Everything a training row or an
inference request can name is translated to genomic HGVS (``NC_*:g.``) here:

* ``annotations`` — PharmGKB annotation id → the variants it actually studied
  (``IMPLICATES`` links: allele-specific, the most precise link);
* ``labels`` — allele label → its variants: rsIDs (every alternative at the
  site) and star alleles ``GENE*N`` (their *cis* variant set);
* ``variants`` — HGVS → annotation (gene, SO consequence terms, protein HGVS);
* ``functions`` — star-allele PGx function (no/decreased/normal/increased).

Everything but ``functions`` comes from Modephere's pgx export
(``Settings.modephere.pgx_export``: ``variants``, ``haplotypes``,
``allele_variants`` and ``annotation_variants`` parquet; GRCh38 HGVS keyed
``hgvs_g38``). Later sources win: a PharmGKB label's own resolution
(``allele_variants``) over the PharmVar definition of the same name, over an
rsID's site alternatives. PharmVar haplotypes named by an rsID (DPYD's
``rs111858276``) are labelled ``<gene><name>`` (``DPYDrs111858276``), so they
never shadow the rsID of the site. ``functions`` is
dev_Pgen's ``data/dicts/star_alleles.tsv``. Built by ``python -m
src.data.library``; saved as ``hgvs_catalog.pt``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import torch

from src.config import get_settings
from src.data.genotype.haplotype_function import HaplotypeFunctionProvider

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

ALLELE_VARIANTS_FILE = "allele_variants.parquet"
ANNOTATION_VARIANTS_FILE = "annotation_variants.parquet"
VARIANTS_FILE = "variants.parquet"
HAPLOTYPES_FILE = "haplotypes.parquet"


def _read(pgx_export: Path, name: str) -> pl.DataFrame | None:
    path = pgx_export / name
    if not path.exists():
        logger.warning(
            "Modephere export %s not found; its links are missing from the catalog "
            "(run 'python -m src.data.library --refresh-pgx')",
            path,
        )
        return None
    return pl.read_parquet(path)


def _main_gene(genes: list[str], star_alleles: list[str]) -> str | None:
    """The gene a variant belongs to: its star allele's, else a named gene."""
    if star_alleles:
        return star_alleles[0].split("*", 1)[0]
    named = [g for g in genes if not g.startswith("LOC")]
    candidates = named or genes
    return candidates[0] if candidates else None


def _add_variants(catalog: HgvsCatalog, frame: pl.DataFrame) -> None:
    """``variants`` from the export, and every rsID as a label of its alleles."""
    by_rsid: dict[str, list[str]] = defaultdict(list)
    columns = ("hgvs_g38", "genes", "star_alleles", "so_terms", "p_hgvs", "rsids")
    for hgvs, genes, stars, so_terms, p_hgvs, rsids in frame.select(
        columns
    ).iter_rows():
        catalog.variants[hgvs] = VariantInfo(
            gene=_main_gene(genes, stars), so_terms=tuple(so_terms), p_hgvs=p_hgvs
        )
        for rsid in rsids:
            by_rsid[rsid].append(hgvs)
    for rsid, hgvs in by_rsid.items():
        catalog.labels[rsid] = tuple(sorted(hgvs))


def haplotype_label(name: str, gene: str | None) -> str:
    """``CYP2C19*2`` as is; an rsID-named haplotype gets its gene in front."""
    return name if "*" in name or not gene else f"{gene}{name}"


def _add_haplotypes(catalog: HgvsCatalog, frame: pl.DataFrame) -> None:
    """Every PharmVar / ClinPGx haplotype → its cis variants (structural: none)."""
    for name, gene, members in frame.select("name", "gene", "variants").iter_rows():
        catalog.labels[haplotype_label(name, gene)] = tuple(members)
        for hgvs in members:
            known = catalog.variants.get(hgvs, VariantInfo())
            if known.gene is None and gene:
                catalog.variants[hgvs] = VariantInfo(gene, known.so_terms, known.p_hgvs)


def _add_labels(catalog: HgvsCatalog, frame: pl.DataFrame) -> None:
    """Allele label → canonical variants (overrides an rsID's own entry)."""
    links: dict[str, list[str]] = defaultdict(list)
    rows = frame.filter(pl.col("hgvs_g38").is_not_null())
    for label, hgvs in rows.select("label", "hgvs_g38").iter_rows():
        if hgvs not in links[label]:
            links[label].append(hgvs)
    catalog.labels.update({k: tuple(v) for k, v in links.items()})


def _add_annotations(catalog: HgvsCatalog, frame: pl.DataFrame) -> None:
    by_annotation: dict[str, list[str]] = defaultdict(list)
    rows = frame.filter(pl.col("hgvs_g38").is_not_null())
    for annotation_id, hgvs in rows.select("annotation_id", "hgvs_g38").iter_rows():
        by_annotation[str(annotation_id)].append(hgvs)
    catalog.annotations = {k: tuple(v) for k, v in by_annotation.items()}


def build_hgvs_catalog(*, pgx_export: Path, star_alleles_tsv: Path) -> HgvsCatalog:
    """Assemble the catalog from Modephere's pgx export (missing tables are skipped)."""
    catalog = HgvsCatalog()
    if (variants := _read(pgx_export, VARIANTS_FILE)) is not None:
        _add_variants(catalog, variants)
    if (haplotypes := _read(pgx_export, HAPLOTYPES_FILE)) is not None:
        _add_haplotypes(catalog, haplotypes)
    if (labels := _read(pgx_export, ALLELE_VARIANTS_FILE)) is not None:
        _add_labels(catalog, labels)
    if (implicated := _read(pgx_export, ANNOTATION_VARIANTS_FILE)) is not None:
        _add_annotations(catalog, implicated)
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
    "haplotype_label",
    "load_hgvs_catalog",
]
