"""The knowledge graph as the training dataset.

Each ``:Annotation`` with at least one drug that has a ChEMBL ID, one gene and
one allele label becomes one training row, in the shape of the training TSV
(multi-value fields ``|``-joined). Drugs are identified by their **ChEMBL
parent ID** (``drugs_chembl``), taken — with the drug names, in the same order —
from Modephere's ``annotation_rows`` export
(:func:`src.data.library.bridge.annotation_drugs`), the same identity the drug
library is built and indexed by. A drug without a ChEMBL ID is left out of its
row; a row left without any drug is dropped (and counted in the log).

The genotype is stated in HGVS whenever the KG knows the exact variant the
annotation studied (``IMPLICATES``); otherwise the allele labels (rsIDs, star
alleles) are kept and linked to HGVS by the genotype tower's catalog.

Two ways to use it:

* live — ``load_training_rows()`` queries Neo4j (``python main.py train
  --neo4j``);
* snapshot — ``python -m src.kg export`` writes :func:`training_snapshot_path`
  (parquet), trainable offline with ``--input <that path>``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from src.config import get_settings
from src.kg.graph import KnowledgeGraph

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.config import KGSettings
    from src.domain.schemas.annotation import AnnotationSource

logger = logging.getLogger(__name__)

MULTI_VALUE_SEP = "|"
#: ``direction_of_effect`` value for annotations that state no direction.
UNDETERMINED_DIRECTION = "__UNDETERMINED__"
_SNAPSHOT_FILE = "kg/training_rows.parquet"


def training_snapshot_path() -> Path:
    """Where ``python -m src.kg export`` writes the training-row snapshot."""
    return get_settings().paths.library / _SNAPSHOT_FILE


def _joined(col: str) -> pl.Expr:
    return pl.col(col).list.join(MULTI_VALUE_SEP)


def with_drug_identity(rows: pl.DataFrame, drugs: pl.DataFrame) -> pl.DataFrame:
    """Replace each row's drugs with Modephere's (name + ChEMBL ID, aligned).

    ``drugs`` is :func:`src.data.library.bridge.annotation_drugs` output
    (``annotation_id, position, name, chembl_id``).
    """
    identity = (
        drugs.drop_nulls("chembl_id")
        .sort("annotation_id", "position")
        .group_by("annotation_id", maintain_order=True)
        .agg(
            pl.col("name").alias("drug_names"),
            pl.col("chembl_id").alias("drug_chembl_ids"),
        )
    )
    out = rows.drop("drug_names", strict=False).join(
        identity,
        left_on=pl.col("annotation_id").cast(pl.String),
        right_on="annotation_id",
        how="inner",
    )
    if (dropped := rows.height - out.height) > 0:
        logger.info(
            "Training rows: %d of %d dropped, no drug with a ChEMBL ID",
            dropped,
            rows.height,
        )
    return out


def to_training_frame(
    rows: pl.DataFrame, drugs: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Map raw ``KnowledgeGraph.training_rows`` output to the training-TSV schema.

    ``drugs`` (default: Modephere's export, see :func:`with_drug_identity`)
    gives each row its drugs' names and ChEMBL IDs.
    """
    if drugs is None:
        from src.data.library.bridge import annotation_drugs

        drugs = annotation_drugs()
    rows = with_drug_identity(rows, drugs)
    return rows.select(
        _joined("drug_chembl_ids").alias("drugs_chembl"),
        _joined("drug_names").alias("drugs"),
        pl.when(pl.col("implicated_hgvs").list.len() > 0)
        .then(_joined("implicated_hgvs"))
        .otherwise(_joined("genotypes"))
        .alias("genotype"),
        _joined("genes").alias("gene"),
        pl.col("alleles"),
        _joined("phenotype_categories").alias("phenotype_category"),
        pl.col("direction")
        .fill_null(UNDETERMINED_DIRECTION)
        .alias("direction_of_effect"),
        pl.col("comparison_alleles").alias("comparison_allele(s)_or_genotype(s)"),
        pl.col("significance"),
        pl.when(pl.col("associated"))
        .then(pl.lit("Associated with"))
        .when(pl.col("associated").not_())
        .then(pl.lit("Not associated with"))
        .otherwise(None)
        .alias("is/is_not_associated"),
        pl.col("annotation_id")
        .cast(pl.Int64, strict=False)
        .alias("variant_annotation_id"),
        pl.col("pmid").cast(pl.Int64, strict=False),
        pl.col("source"),
    )


def load_training_rows(
    sources: Iterable[AnnotationSource] | None = None,
    settings: KGSettings | None = None,
) -> pl.DataFrame:
    """Query Neo4j for the training rows (training-TSV schema)."""
    wanted = [s.value for s in sources] if sources is not None else None
    with KnowledgeGraph.connect(settings) as kg:
        raw = kg.training_rows(wanted)
    frame = to_training_frame(raw)
    logger.info("Loaded %d training rows from Neo4j (sources=%s)", len(frame), wanted)
    return frame


__all__ = [
    "MULTI_VALUE_SEP",
    "UNDETERMINED_DIRECTION",
    "load_training_rows",
    "to_training_frame",
    "training_snapshot_path",
    "with_drug_identity",
]
