"""Star-allele catalog — loaded from data/dicts/star_alleles.tsv.

Replaces the hardcoded ``STAR_ALLELE_MAP`` that was buried in src/interface/io.py.
The TSV is the source of truth; updates ship as data changes, not code changes.

TSV schema (tab-separated, header required):
    gene\tallele\trsids\tfunction\tnotes

- ``gene``     — HGNC symbol
- ``allele``   — star-allele suffix (e.g. "4", "1F", "9A"); the leading '*' is implied
- ``rsids``    — pipe-separated dbSNP IDs (haplotypes have multiple)
- ``function`` — one of AlleleFunction values (snake_case)
- ``notes``    — free text (optional)
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.settings import PROJECT_ROOT
from src.domain.gene import AlleleFunction, Gene, StarAllele

_DEFAULT_TSV = PROJECT_ROOT / "data" / "dicts" / "star_alleles.tsv"


class StarAlleleRecord(BaseModel):
    """A single row from the star-allele catalog."""

    model_config = ConfigDict(frozen=True)

    star_allele: StarAllele
    rsids: tuple[str, ...] = Field(default_factory=tuple)
    notes: str = ""

    @field_validator("rsids", mode="before")
    @classmethod
    def _normalize_rsids(cls, v: object) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, str):
            parts = [p.strip() for p in v.split("|") if p.strip()]
            return tuple(parts)
        if isinstance(v, Iterable):
            return tuple(str(p).strip() for p in v if str(p).strip())
        msg = f"rsids must be str or iterable, got {type(v).__name__}"
        raise TypeError(msg)


class StarAlleleMap:
    """Bidirectional index over the star-allele catalog.

    Lookup paths:
      - by full label  ``map["CYP2D6*4"]`` → StarAlleleRecord
      - by rsID        ``map.alleles_for_rsid("rs3892097")`` → list[StarAllele]
      - by gene        ``map.alleles_for_gene("CYP2D6")`` → list[StarAllele]
    """

    def __init__(self, records: list[StarAlleleRecord]):
        self._records: dict[str, StarAlleleRecord] = {}
        self._by_rsid: dict[str, list[StarAllele]] = {}
        self._by_gene: dict[str, list[StarAllele]] = {}

        for record in records:
            label = record.star_allele.label
            if label in self._records:
                msg = f"duplicate star allele in catalog: {label}"
                raise ValueError(msg)
            self._records[label] = record
            self._by_gene.setdefault(record.star_allele.gene.symbol, []).append(record.star_allele)
            for rsid in record.rsids:
                self._by_rsid.setdefault(rsid, []).append(record.star_allele)

    def __contains__(self, label: str) -> bool:
        return label in self._records

    def __getitem__(self, label: str) -> StarAlleleRecord:
        return self._records[label]

    def __len__(self) -> int:
        return len(self._records)

    def get(self, label: str, default: StarAlleleRecord | None = None) -> StarAlleleRecord | None:
        return self._records.get(label, default)

    def alleles_for_rsid(self, rsid: str) -> list[StarAllele]:
        return list(self._by_rsid.get(rsid, ()))

    def alleles_for_gene(self, gene_symbol: str) -> list[StarAllele]:
        return list(self._by_gene.get(gene_symbol.upper(), ()))

    @property
    def labels(self) -> list[str]:
        return list(self._records.keys())

    @property
    def rsid_to_labels(self) -> dict[str, list[str]]:
        """Inverse map (legacy compat for code that consumes the old
        ``RSID_TO_STAR_ALLELES`` dict)."""
        return {rsid: [a.label for a in alleles] for rsid, alleles in self._by_rsid.items()}


def _row_to_record(row: dict[str, str | None]) -> StarAlleleRecord:
    def _cell(key: str) -> str:
        # csv.DictReader fills short rows with None; treat None and missing the same.
        value = row.get(key)
        return value.strip() if isinstance(value, str) else ""

    gene_symbol = _cell("gene")
    allele = _cell("allele")
    gene = Gene(symbol=gene_symbol)

    function_label = _cell("function") or AlleleFunction.UNKNOWN.value
    try:
        function = AlleleFunction(function_label)
    except ValueError as e:
        msg = f"unknown function label {function_label!r} for {gene_symbol}*{allele}"
        raise ValueError(msg) from e

    star = StarAllele(gene=gene, allele=allele, function=function)
    return StarAlleleRecord(
        star_allele=star,
        rsids=_cell("rsids"),
        notes=_cell("notes"),
    )


def load_star_alleles(tsv_path: Path | None = None) -> StarAlleleMap:
    """Read the catalog TSV and return a fully-indexed StarAlleleMap.

    Raises FileNotFoundError if ``tsv_path`` doesn't exist (useful so callers
    fail fast rather than silently using an empty map).
    """
    path = tsv_path or _DEFAULT_TSV
    if not path.exists():
        msg = f"star allele catalog not found at {path}"
        raise FileNotFoundError(msg)

    records: list[StarAlleleRecord] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "gene" not in reader.fieldnames:
            msg = f"star allele TSV missing header at {path}"
            raise ValueError(msg)
        for row in reader:
            if not row.get("gene", "").strip() or not row.get("allele", "").strip():
                continue  # skip blank rows
            records.append(_row_to_record(row))

    return StarAlleleMap(records)


@lru_cache(maxsize=1)
def get_default_map() -> StarAlleleMap:
    """Cached singleton over the project's default catalog."""
    return load_star_alleles()
