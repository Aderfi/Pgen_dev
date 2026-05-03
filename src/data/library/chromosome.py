"""Chromosome label ↔ RefSeq accession mapping for GRCh38.

Used by the variant validator to resolve user-supplied chromosome labels
('chr1', '1', 'X', 'NC_000001.11', etc.) to the canonical RefSeq accession
that appears in the Ensembl FASTA. Lookups go through
``src.domain.variant.normalize_chromosome`` first to canonicalize input.
"""

from __future__ import annotations

from src.domain.variant import normalize_chromosome


# GRCh38 RefSeq accessions for human chromosomes 1-22, X, Y, MT.
CHROM_TO_REFSEQ: dict[str, str] = {
    "1":  "NC_000001.11",
    "2":  "NC_000002.12",
    "3":  "NC_000003.12",
    "4":  "NC_000004.12",
    "5":  "NC_000005.10",
    "6":  "NC_000006.12",
    "7":  "NC_000007.14",
    "8":  "NC_000008.11",
    "9":  "NC_000009.12",
    "10": "NC_000010.11",
    "11": "NC_000011.10",
    "12": "NC_000012.12",
    "13": "NC_000013.11",
    "14": "NC_000014.9",
    "15": "NC_000015.10",
    "16": "NC_000016.10",
    "17": "NC_000017.11",
    "18": "NC_000018.10",
    "19": "NC_000019.10",
    "20": "NC_000020.11",
    "21": "NC_000021.9",
    "22": "NC_000022.11",
    "X":  "NC_000023.11",
    "Y":  "NC_000024.10",
    "MT": "NC_012920.1",
}


def to_refseq(chrom: str) -> str:
    """Translate any chromosome label to its RefSeq accession.

    Falls back to the input value if no mapping exists (alt contigs, scaffolds).
    """
    canonical = normalize_chromosome(chrom)
    return CHROM_TO_REFSEQ.get(canonical, canonical)


def matches_fasta(chrom: str, fasta_keys: set[str]) -> str | None:
    """Pick whichever form of the chromosome label appears in ``fasta_keys``.

    Returns the matching key, or None if neither the bare form nor the
    RefSeq accession is present in the FASTA index.
    """
    canonical = normalize_chromosome(chrom)
    if canonical in fasta_keys:
        return canonical
    accession = CHROM_TO_REFSEQ.get(canonical)
    if accession and accession in fasta_keys:
        return accession
    return None
