"""Orchestrate building the single-file genotype library (``geno_graphs.pt``).

Gathers variants (PharmVar haplotypes + the pan-gene SNP table), groups them by
gene, resolves each gene's genomic coordinates from the RefSeq annotation, builds
one variant-centric graph per gene and collates them into a :class:`GenoLibrary`.
Genes without annotation coordinates are skipped (logged).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from src.config import get_settings
from src.data.library.gene_graph import build_gene_graph
from src.data.library.geno_store import GenoLibrary
from src.data.library.haplotype_function import HaplotypeFunctionProvider

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant
    from src.genomics.annotation import GeneAnnotation

logger = logging.getLogger(__name__)


def build_geno_library(
    annotation: GeneAnnotation,
    *,
    variants: Iterable[IngestedVariant] = (),
    haplotypes: Iterable[IngestedHaplotype] = (),
    function_provider: HaplotypeFunctionProvider | None = None,
    rsid_to_hgvs: dict[str, str] | None = None,
) -> GenoLibrary:
    """Assemble per-gene variation graphs into a single :class:`GenoLibrary`.

    ``variants`` and ``haplotypes`` are grouped by gene; each gene present in the
    ``annotation`` becomes a graph (others are skipped). ``function_provider``
    supplies per-path PGx function (null when omitted). ``rsid_to_hgvs`` is the
    rsID → genomic-HGVS bridge stored with the library so rsID-based genotypes
    resolve to variant nodes at train/inference time.
    """
    variants_by_gene: dict[str, list[IngestedVariant]] = defaultdict(list)
    for variant in variants:
        if variant.gene:
            variants_by_gene[variant.gene].append(variant)
    haplotypes_by_gene: dict[str, list[IngestedHaplotype]] = defaultdict(list)
    for haplotype in haplotypes:
        haplotypes_by_gene[haplotype.gene].append(haplotype)

    functions = function_provider or HaplotypeFunctionProvider.null()
    graphs: dict[str, object] = {}
    skipped = 0
    for gene in sorted(set(variants_by_gene) | set(haplotypes_by_gene)):
        model = annotation.get(gene)
        if model is None:
            skipped += 1
            continue
        graphs[gene] = build_gene_graph(
            model,
            variants=variants_by_gene.get(gene, []),
            haplotypes=haplotypes_by_gene.get(gene, []),
            function_provider=functions,
        )
    logger.info(
        "build_geno_library: built %d gene graphs (%d genes skipped — no annotation).",
        len(graphs),
        skipped,
    )
    return GenoLibrary(graphs, rsid_to_hgvs=rsid_to_hgvs or {})  # type: ignore[arg-type]


def build_geno_library_from_settings(
    *,
    genes: Iterable[str] | None = None,
    save: bool = True,
) -> GenoLibrary:
    """Build the library from the project's standard inputs.

    ``genes`` restricts the build to an allow-list (fast partial / PGx builds);
    ``None`` builds every gene with PharmVar or SNP-table evidence. When ``save``
    is set, the library is written to ``paths.library / 'geno_graphs.pt'``.
    """
    # Local imports: heavy (polars/pysam-free here, but keeps the module light).
    from src.data.library.ingest import pharmvar, snp_table
    from src.genomics.annotation import GeneAnnotation

    paths = get_settings().paths
    allow = {str(g) for g in genes} if genes is not None else None

    pharmvar_dir = paths.data / "haplotype_variants"
    haplotypes = [
        h
        for h in pharmvar.iter_haplotypes(pharmvar_dir)
        if allow is None or h.gene in allow
    ]
    snp_path = paths.data / "snp_data_output.tsv"
    variants = (
        [
            v
            for v in snp_table.iter_variants_from_snp_table(snp_path)
            if allow is None or v.gene in allow
        ]
        if snp_path.exists()
        else []
    )

    # rsID → HGVS bridge: SNP table (broad, long tail) overlaid with PharmVar
    # (PGx genes). PharmVar wins on conflict (curated).
    rsid_to_hgvs = snp_table.rsid_hgvs_index(snp_path) if snp_path.exists() else {}
    rsid_to_hgvs.update(pharmvar.rsid_hgvs_index(pharmvar_dir))

    needed = {h.gene for h in haplotypes} | {v.gene for v in variants if v.gene}
    annotation = GeneAnnotation.from_gff(paths.ref_genome_gff, genes=needed)
    star_tsv = paths.data / "dicts" / "star_alleles.tsv"
    functions = (
        HaplotypeFunctionProvider.from_tsv(star_tsv)
        if star_tsv.exists()
        else HaplotypeFunctionProvider.null()
    )

    library = build_geno_library(
        annotation,
        variants=variants,
        haplotypes=haplotypes,
        function_provider=functions,
        rsid_to_hgvs=rsid_to_hgvs,
    )
    if save:
        library.save(_library_path(paths.library))
    return library


def _library_path(library_root: Path) -> Path:
    return library_root / "geno_graphs.pt"


__all__ = ["build_geno_library", "build_geno_library_from_settings"]
