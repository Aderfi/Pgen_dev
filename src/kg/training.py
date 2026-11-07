"""The knowledge graph as the training dataset.

Each ``:Annotation`` with at least one drug entity, one gene and one allele
label becomes one training row, in the shape of the training TSV (multi-value
fields ``|``-joined). ``drugs_chembl`` lists the row's **drug entities**, taken
— with the drug names, in the same order — from Modephere's ``annotation_rows``
export (:func:`src.data.library.bridge.annotation_drugs`):

* a drug with a ChEMBL parent ID is that ID (``CHEMBL25``);
* a combination ("sulfamethoxazole / trimethoprim", FOLFOX) is one ID per
  component;
* a drug class ("opioids") is one ``ATC:<codes>`` key (``ATC:N02A``), pooled
  over its member drugs by the drug tower; ``is_class_row`` marks rows that
  have one (training-only, see ``src/evaluation/class_rows.py``).

Anything else (metabolites, non-drugs) is left out of its row, counted per
reason in the log; a row left without any entity is dropped.

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


_EMPTY_COMPONENTS = pl.List(pl.Struct({"name": pl.String, "chembl_id": pl.String}))


def _drug_entities(drugs: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """One row per entity: ``annotation_id, position, part, name, key``.

    Also returns how many drugs were left out, per reason.
    """
    from src.data.library.bridge import class_entity_key

    if "atc_codes" not in drugs.columns:
        drugs = drugs.with_columns(pl.lit([], pl.List(pl.String)).alias("atc_codes"))
    if "components" not in drugs.columns:
        drugs = drugs.with_columns(pl.lit([], _EMPTY_COMPONENTS).alias("components"))
    rows: list[tuple[str, int, int, str, str]] = []
    left_out: dict[str, int] = {}
    for r in drugs.sort("annotation_id", "position").iter_rows(named=True):
        aid, pos, name = str(r["annotation_id"]), r["position"], r["name"]
        if r["chembl_id"]:
            rows.append((aid, pos, 0, name, r["chembl_id"]))
        elif parts := [c for c in r["components"] or [] if c["chembl_id"]]:
            rows += [
                (aid, pos, i, c["name"], c["chembl_id"]) for i, c in enumerate(parts)
            ]
        elif r["atc_codes"]:
            rows.append((aid, pos, 0, name, class_entity_key(r["atc_codes"])))
        else:
            reason = (
                "combination without resolved components"
                if r["components"]
                else "no ChEMBL ID, ATC class or components"
            )
            left_out[reason] = left_out.get(reason, 0) + 1
    schema = {
        "annotation_id": pl.String,
        "position": pl.Int64,
        "part": pl.Int64,
        "name": pl.String,
        "key": pl.String,
    }
    return pl.DataFrame(rows, schema=schema, orient="row"), left_out


def with_drug_identity(rows: pl.DataFrame, drugs: pl.DataFrame) -> pl.DataFrame:
    """Replace each row's drugs with its drug entities (keys + names, aligned).

    ``drugs`` is :func:`src.data.library.bridge.annotation_drugs` output
    (``annotation_id, position, name, chembl_id[, atc_codes, components]``).
    Adds ``drug_chembl_ids`` (entity keys) and ``drug_names``.
    """
    entities, left_out = _drug_entities(drugs)
    if left_out:
        logger.info(
            "Training rows: drugs left out of their rows: %s",
            ", ".join(f"{n} {reason}" for reason, n in sorted(left_out.items())),
        )
    identity = (
        entities.sort("annotation_id", "position", "part")
        .group_by("annotation_id", maintain_order=True)
        .agg(
            pl.col("name").alias("drug_names"),
            pl.col("key").alias("drug_chembl_ids"),
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
            "Training rows: %d of %d dropped, no drug entity (drug, component or "
            "class)",
            dropped,
            rows.height,
        )
    return out


def to_training_frame(
    rows: pl.DataFrame, drugs: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Map raw ``KnowledgeGraph.training_rows`` output to the training-TSV schema.

    ``drugs`` (default: Modephere's export, see :func:`with_drug_identity`)
    gives each row its drug entities and their names.
    """
    if drugs is None:
        from src.data.library.bridge import annotation_drugs

        drugs = annotation_drugs()
    rows = with_drug_identity(rows, drugs)
    from src.data.library.bridge import CLASS_PREFIX

    return rows.select(
        _joined("drug_chembl_ids").alias("drugs_chembl"),
        _joined("drug_names").alias("drugs"),
        pl.col("drug_chembl_ids")
        .list.eval(pl.element().str.starts_with(CLASS_PREFIX))
        .list.any()
        .alias("is_class_row"),
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
