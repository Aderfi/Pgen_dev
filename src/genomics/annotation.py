"""GRCh38 RefSeq gene-annotation reader (GFF3) for the genotype tower.

Why this exists
---------------
The HGVS genotype tower builds one variation graph per gene, anchored on the
gene's genomic reference sequence and its transcript/CDS structure. That requires
resolving a gene symbol to its RefSeq genomic accession (``NC_*``), span and
strand, plus its transcripts (``NM_*``), exons and CDS segments (for ``c.``/``p.``
mapping). This module reads the NCBI RefSeq GFF3 (``GCF_000001405.26``) into
typed :class:`GeneModel` objects and slices the reference sequence per gene.

Coordinates are **1-based inclusive** (GFF/HGVS convention); a 0-based slice is
only used internally against ``pyfaidx`` and never surfaced.

The GFF is large (~640 MB); parsing relies on the fact that RefSeq orders each
``gene`` immediately before its ``mRNA`` children, which precede their ``exon``/
``CDS`` lines. Passing ``genes=`` restricts parsing to a symbol allow-list and
skips everything else cheaply — the common path for a PGx build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pyfaidx import Fasta

logger = logging.getLogger(__name__)

# Feature types we retain. Genes carry the genomic span; mRNAs carry the
# transcript structure used for c./p. mapping.
_GENE_TYPES = frozenset({"gene", "pseudogene"})
_RNA_TYPES = frozenset({"mRNA"})


@dataclass(frozen=True, slots=True)
class Transcript:
    """A single RefSeq transcript of a gene (1-based inclusive coordinates)."""

    accession: str  # NM_* / NR_*
    protein_accession: str | None  # NP_* (None for non-coding transcripts)
    exons: tuple[tuple[int, int], ...]
    cds: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class GeneModel:
    """A gene's genomic location, strand and transcript structure."""

    symbol: str
    accession: str  # NC_* RefSeq genomic accession (the g. reference)
    start: int  # 1-based inclusive
    stop: int  # 1-based inclusive
    strand: str  # '+' or '-'
    biotype: str
    transcripts: tuple[Transcript, ...]

    @property
    def length(self) -> int:
        return self.stop - self.start + 1


def _parse_attrs(col9: str) -> dict[str, str]:
    """Parse a GFF3 column-9 attribute string into a flat dict."""
    attrs: dict[str, str] = {}
    for field in col9.rstrip(";").split(";"):
        key, sep, value = field.partition("=")
        if sep:
            attrs[key.strip()] = value
    return attrs


class _GffAccumulator:
    """Streams GFF feature rows into the gene/rna/exon/CDS tables.

    Relies on RefSeq ordering (gene → its mRNAs → their exons/CDS): a child is
    kept only when its parent was kept, so an allow-list on genes prunes whole
    subtrees cheaply.
    """

    def __init__(self, wanted: set[str] | None) -> None:
        self.wanted = wanted
        self.gene_rows: dict[str, dict[str, object]] = {}
        self.rna_to_gene: dict[str, str] = {}
        self.rna_meta: dict[str, dict[str, str | None]] = {}
        self.exons: dict[str, list[tuple[int, int]]] = {}
        self.cds: dict[str, list[tuple[int, int]]] = {}

    def add(
        self, accession: str, ftype: str, start: int, stop: int, strand: str, col9: str
    ) -> None:
        if ftype in _GENE_TYPES:
            self._add_gene(accession, start, stop, strand, col9)
        elif ftype in _RNA_TYPES:
            self._add_rna(col9)
        elif ftype == "exon":
            self._add_segment(self.exons, start, stop, col9)
        elif ftype == "CDS":
            self._add_cds(start, stop, col9)

    def _add_gene(
        self, accession: str, start: int, stop: int, strand: str, col9: str
    ) -> None:
        attrs = _parse_attrs(col9)
        symbol = attrs.get("Name") or attrs.get("gene")
        gid = attrs.get("ID")
        if (
            not symbol
            or not gid
            or (self.wanted is not None and symbol not in self.wanted)
        ):
            return
        self.gene_rows[gid] = {
            "symbol": symbol,
            "accession": accession,
            "start": start,
            "stop": stop,
            "strand": strand,
            "biotype": attrs.get("gene_biotype", "unknown"),
        }

    def _add_rna(self, col9: str) -> None:
        attrs = _parse_attrs(col9)
        rid, parent = attrs.get("ID"), attrs.get("Parent")
        if not rid or parent not in self.gene_rows:
            return
        self.rna_to_gene[rid] = parent
        self.rna_meta[rid] = {"accession": attrs.get("Name", rid), "protein": None}

    def _add_segment(
        self, table: dict[str, list[tuple[int, int]]], start: int, stop: int, col9: str
    ) -> None:
        rid = _parse_attrs(col9).get("Parent")
        if rid in self.rna_to_gene:
            table.setdefault(rid, []).append((start, stop))

    def _add_cds(self, start: int, stop: int, col9: str) -> None:
        attrs = _parse_attrs(col9)
        rid = attrs.get("Parent")
        if rid not in self.rna_to_gene:
            return
        self.cds.setdefault(rid, []).append((start, stop))
        if self.rna_meta[rid]["protein"] is None:
            self.rna_meta[rid]["protein"] = attrs.get("Name")


class GeneAnnotation:
    """Symbol → :class:`GeneModel` index built from a RefSeq GFF3."""

    def __init__(self, genes: dict[str, GeneModel]) -> None:
        self._genes = genes

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._genes

    def __len__(self) -> int:
        return len(self._genes)

    def get(self, symbol: str) -> GeneModel | None:
        return self._genes.get(symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._genes)

    @classmethod
    def from_gff(
        cls,
        gff_path: Path,
        *,
        genes: Iterable[str] | None = None,
    ) -> GeneAnnotation:
        """Build the index from a RefSeq GFF3.

        ``genes`` restricts parsing to an allow-list of symbols (the fast PGx
        path); ``None`` parses every gene. The first gene feature for a symbol
        wins — duplicates (paralogous copies on alt loci) are logged and skipped.
        """
        wanted = {str(g) for g in genes} if genes is not None else None
        acc = _GffAccumulator(wanted)

        with gff_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) != 9:
                    continue
                accession, _src, ftype, start, stop, _score, strand, _phase, col9 = cols
                acc.add(accession, ftype, int(start), int(stop), strand, col9)

        return cls(
            _assemble(acc.gene_rows, acc.rna_to_gene, acc.rna_meta, acc.exons, acc.cds)
        )


def _assemble(
    gene_rows: dict[str, dict[str, object]],
    rna_to_gene: dict[str, str],
    rna_meta: dict[str, dict[str, str | None]],
    exons: dict[str, list[tuple[int, int]]],
    cds: dict[str, list[tuple[int, int]]],
) -> dict[str, GeneModel]:
    """Join the streamed gene/rna/exon/CDS records into GeneModels by symbol."""
    rnas_by_gene: dict[str, list[str]] = {}
    for rid, gid in rna_to_gene.items():
        rnas_by_gene.setdefault(gid, []).append(rid)

    genes: dict[str, GeneModel] = {}
    for gid, row in gene_rows.items():
        transcripts = tuple(
            Transcript(
                accession=str(rna_meta[rid]["accession"]),
                protein_accession=rna_meta[rid]["protein"],
                exons=tuple(sorted(exons.get(rid, []))),
                cds=tuple(sorted(cds.get(rid, []))),
            )
            for rid in rnas_by_gene.get(gid, [])
        )
        symbol = str(row["symbol"])
        model = GeneModel(
            symbol=symbol,
            accession=str(row["accession"]),
            start=int(row["start"]),  # type: ignore[call-overload]
            stop=int(row["stop"]),  # type: ignore[call-overload]
            strand=str(row["strand"]),
            biotype=str(row["biotype"]),
            transcripts=transcripts,
        )
        if symbol in genes:
            logger.warning(
                "Annotation: duplicate gene symbol %s (accession %s) — keeping first.",
                symbol,
                model.accession,
            )
            continue
        genes[symbol] = model
    logger.info("Annotation: assembled %d gene models.", len(genes))
    return genes


def gene_sequence(gene: GeneModel, fasta: Fasta) -> str:
    """Return the plus-strand genomic reference sequence spanning ``gene``.

    The slice is on the gene's ``NC_*`` accession in genomic (g.) coordinates, so
    positions map directly to HGVS ``g.`` — strand is *not* applied here (HGVS g.
    is always plus-strand); callers needing the coding orientation use
    :attr:`GeneModel.strand` downstream.
    """
    record = fasta[gene.accession]
    return str(record[gene.start - 1 : gene.stop].seq).upper()


__all__ = [
    "GeneAnnotation",
    "GeneModel",
    "Transcript",
    "gene_sequence",
]
