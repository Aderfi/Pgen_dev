"""Row identity and the entity keys each split scenario groups on.

Every split scenario partitions rows by an *entity*: a drug, a variant, a
gene, an article or a (drugs, variants) pair. A row can name several
entities of one kind (``"11|22"``), so each row maps to a set of keys.

* **Variant** = its resolved HGVS (through :class:`HgvsCatalog`), so one change
  written as an rsID in one row and as HGVS in another is one entity. Tokens
  the catalog cannot resolve are kept raw. ``Gene_Deletion`` is a variant
  (``<GENE>:gene_deletion``); a reference-only genotype is ``<GENE>:reference``.
* **Drug** = the structural identifier the drug tower reads (the configured
  drug column); rows without one fall back to ``name:<normalized name>``.
"""

from __future__ import annotations

import hashlib
import logging
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

from src.core import DataError
from src.data.genotype import GenotypeGraphBuilder
from src.data.genotype.graph import split_values

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.data.genotype import HgvsCatalog

logger = logging.getLogger(__name__)

ROW_ID = "row_id"
ANNOTATION_ID_COLUMN = "variant_annotation_id"
GENE_DELETION_TOKEN = "Gene_Deletion"
REFERENCE_TOKEN = "REF_SEQ"
_HASH_CHUNK = 1 << 20


class Scenario(StrEnum):
    """How the rows are grouped before being dealt into folds."""

    RANDOM = "random"
    PAIR = "pair"
    PMID = "pmid"
    DRUG_COLD = "drug_cold"
    VARIANT_COLD = "variant_cold"
    GENE_COLD = "gene_cold"


def add_row_id(df: pl.DataFrame) -> pl.DataFrame:
    """Append a stable ``row_id``: the annotation id, else a hash of the row."""
    if (
        ANNOTATION_ID_COLUMN in df.columns
        and df[ANNOTATION_ID_COLUMN].null_count() == 0
    ):
        ids = df[ANNOTATION_ID_COLUMN].cast(pl.String)
    else:
        ids = pl.Series(
            [
                hashlib.sha1("\x1f".join(map(str, row)).encode()).hexdigest()
                for row in df.iter_rows()
            ],
            dtype=pl.String,
        )
    if ids.n_unique() != len(ids):
        msg = f"row ids are not unique ({len(ids) - ids.n_unique()} duplicates)"
        raise DataError(msg)
    return df.with_columns(ids.alias(ROW_ID))


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(row: Mapping[str, object], col: str) -> object:
    value = row.get(col)
    return None if value is None or str(value).strip() == "" else value


class EntityResolver:
    """Maps a row to the entity keys of each scenario."""

    def __init__(
        self, catalog: HgvsCatalog, drug_col: str, drug_name_col: str = "drugs"
    ) -> None:
        self.catalog = catalog
        self.drug_col = drug_col
        self.drug_name_col = drug_name_col
        self._linker = GenotypeGraphBuilder(catalog, {})

    def drugs(self, row: Mapping[str, object]) -> frozenset[str]:
        ids = split_values(_cell(row, self.drug_col))
        if ids:
            return frozenset(ids)
        names = split_values(_cell(row, self.drug_name_col))
        return frozenset(f"name:{n.strip().lower()}" for n in names)

    def variants(self, row: Mapping[str, object]) -> frozenset[str]:
        genes = split_values(_cell(row, "gene"))
        tokens = split_values(_cell(row, "genotype"))
        linked, _ = self._linker.link(
            genes,
            _cell(row, "genotype"),
            _cell(row, "alleles"),
            _cell(row, ANNOTATION_ID_COLUMN),
        )
        keys = {item.hgvs for item in linked}
        for token in tokens:
            if token == GENE_DELETION_TOKEN:
                keys.update(f"{g}:gene_deletion" for g in genes)
            elif token == REFERENCE_TOKEN:
                continue  # the reference allele is not a variant
            elif "*" in token:
                label = (
                    token
                    if not token.startswith("*") or not genes
                    else f"{genes[0]}{token}"
                )
                if not self.catalog.labels.get(label):
                    keys.add(label)  # unresolved star allele: keep the label
            elif token not in keys and not self.catalog.labels.get(token):
                keys.add(token)
        if not keys:
            keys = {f"{g}:reference" for g in genes} or {"reference"}
        return frozenset(keys)

    def genes(self, row: Mapping[str, object]) -> frozenset[str]:
        return frozenset(split_values(_cell(row, "gene")))

    def pmid(self, row: Mapping[str, object]) -> frozenset[str]:
        value = _cell(row, "pmid")
        return frozenset({str(value)}) if value is not None else frozenset()

    def pair(self, row: Mapping[str, object]) -> frozenset[str]:
        drugs = ",".join(sorted(self.drugs(row)))
        variants = ",".join(sorted(self.variants(row)))
        return frozenset({f"{drugs}||{variants}"})

    def keys(self, df: pl.DataFrame, scenario: Scenario) -> list[frozenset[str]]:
        """Entity keys of every row of ``df`` (which must carry ``row_id``)."""
        if scenario is Scenario.RANDOM:
            return [frozenset({rid}) for rid in df[ROW_ID].to_list()]
        fn = {
            Scenario.PAIR: self.pair,
            Scenario.PMID: self.pmid,
            Scenario.DRUG_COLD: self.drugs,
            Scenario.VARIANT_COLD: self.variants,
            Scenario.GENE_COLD: self.genes,
        }[scenario]
        keys = []
        for row in df.iter_rows(named=True):
            k = fn(row)
            keys.append(k if k else frozenset({f"row:{row[ROW_ID]}"}))
        return keys


__all__ = [
    "ROW_ID",
    "EntityResolver",
    "Scenario",
    "add_row_id",
    "file_sha256",
]
