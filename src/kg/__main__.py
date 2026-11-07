"""CLI for the knowledge graph.

Usage::

    python -m src.kg fetch            # network: dbSNP, NCBI SPDI, ClinPGx tables -> raw cache
    python -m src.kg ingest [--reset] # offline: raw cache + local files -> Neo4j, then audit
    python -m src.kg export           # Neo4j -> parquet tables under data/library/kg/
    python -m src.kg audit
    python -m src.kg stats

Only ``fetch`` touches the network. ``ingest`` builds the whole graph in
memory first; if any upstream response is missing it exits 1 without
connecting to Neo4j. Connection settings come from ``Settings.kg``
(``PHARMAGEN_KG__*``); NCBI identity from ``Settings.ncbi``
(``PHARMAGEN_NCBI__*``).
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from openpyxl.utils.exceptions import InvalidFileException
from pydantic import BaseModel, Field

from src.config import configure_logging_level, get_settings
from src.core import BioinformaticsError, PharmagenException
from src.domain.schemas.annotation import AnnotationSource, PGxAnnotation
from src.kg.build import (
    allele_table_dataset,
    build_kg,
    local_file_dataset,
    pharmgkb_dataset,
)
from src.kg.fetch import download_allele_tables, plan_fetch, read_table_index, run_fetch
from src.kg.graph import KnowledgeGraph
from src.kg.ncbi import NCBIClient
from src.kg.provenance import Dataset, IngestRun
from src.kg.raw_cache import CacheMissError
from src.kg.report import write_report
from src.kg.sources.allele_tables import load_allele_table
from src.kg.sources.definitions import HaplotypeDefinition
from src.kg.sources.pharmgkb import (
    ALIGNED_TSV_DATASET,
    NAME_CACHE_DATASET,
    iter_annotations,
    load_drug_cid_provenance,
)
from src.kg.sources.pharmvar import (
    grch38_haplotype_files,
    load_pharmvar,
    pharmvar_dataset,
)
from src.kg.training import to_training_frame, training_snapshot_path

logger = logging.getLogger("src.kg")

_SETTINGS = get_settings()
ANNOTATIONS_DIR = _SETTINGS.paths.project_root / "train_data" / "var_anno"
PHARMVAR_DIR = _SETTINGS.paths.data / "haplotype_variants"
NAME_CACHE = _SETTINGS.paths.data / "dicts" / "drug_name_cache.json"
ALIGNED_TSV = _SETTINGS.paths.data / "processed" / "train_data.tsv"
RAW_DIR = _SETTINGS.paths.library / "kg" / "raw"
REPORT_DIR = _SETTINGS.paths.reports / "kg"


class Inputs(BaseModel):
    annotations: list[PGxAnnotation]
    pharmvar: list[HaplotypeDefinition]
    tables: list[HaplotypeDefinition]
    datasets: list[Dataset]
    drug_cids: dict[str, tuple[int, str]]
    annotation_datasets: dict[str, str] = Field(default_factory=dict)
    failed_tables: list[str] = Field(default_factory=list)


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


def load_inputs() -> Inputs:
    """Every local input, each paired with its pinned :class:`Dataset`."""
    datasets: list[Dataset] = []
    annotations: list[PGxAnnotation] = []
    annotation_datasets: dict[str, str] = {}
    for source in AnnotationSource:
        tsv = ANNOTATIONS_DIR / f"{source.value}.tsv"
        annotations.extend(iter_annotations(tsv, source))
        ds = pharmgkb_dataset(tsv, ANNOTATIONS_DIR)
        datasets.append(ds)
        annotation_datasets[source.value] = ds.id

    pharmvar: list[HaplotypeDefinition] = []
    for tsv in grch38_haplotype_files(PHARMVAR_DIR):
        ds = pharmvar_dataset(tsv, _mtime(tsv))
        datasets.append(ds)
        pharmvar.extend(load_pharmvar(tsv, ds.id))

    tables: list[HaplotypeDefinition] = []
    failed_tables: list[str] = []
    tables_dir = RAW_DIR / "allele_tables"
    for gene, entry in read_table_index(tables_dir).items():
        if entry.status != "ok" or entry.path is None:
            continue
        xlsx = tables_dir / entry.path
        ds = allele_table_dataset(xlsx, gene, entry.url, entry.retrieved_at)
        try:
            tables.extend(load_allele_table(xlsx, ds.id))
        except (
            BioinformaticsError,
            zipfile.BadZipFile,
            InvalidFileException,
            KeyError,
        ) as e:
            logger.error("Unreadable ClinPGx table %s: %s", gene, e)
            failed_tables.append(gene)
            continue
        datasets.append(ds)

    local_ids: dict[str, str] = {}
    for logical_id, path, name in (
        (NAME_CACHE_DATASET, NAME_CACHE, "Pharmagen drug name -> CID cache"),
        (
            ALIGNED_TSV_DATASET,
            ALIGNED_TSV,
            "Pharmagen training data drug/CID alignment",
        ),
    ):
        if path.exists():
            ds = local_file_dataset(logical_id, path, name)
            datasets.append(ds)
            local_ids[logical_id] = ds.id
    drug_cids = {
        drug: (cid, local_ids[logical])
        for drug, (cid, logical) in load_drug_cid_provenance(
            NAME_CACHE, ALIGNED_TSV
        ).items()
    }
    return Inputs(
        annotations=annotations,
        pharmvar=pharmvar,
        tables=tables,
        datasets=datasets,
        drug_cids=drug_cids,
        annotation_datasets=annotation_datasets,
        failed_tables=failed_tables,
    )


REFRESH_STATUSES = frozenset({"not_found", "error"})


def _client(*, offline: bool, refresh: bool = False) -> NCBIClient:
    return NCBIClient(
        get_settings().ncbi,
        RAW_DIR,
        offline=offline,
        refresh=REFRESH_STATUSES if refresh else frozenset(),
    )


def _fetch(refresh: bool) -> int:
    inputs = load_inputs()
    pharmvar_genes = {d.gene for d in inputs.pharmvar}
    plan = plan_fetch(inputs.annotations, inputs.pharmvar, pharmvar_genes)
    download_allele_tables(
        plan.allele_table_genes, RAW_DIR / "allele_tables", refresh=refresh
    )
    inputs = load_inputs()  # pick up the freshly downloaded tables
    run_fetch(
        plan, inputs.pharmvar + inputs.tables, _client(offline=False, refresh=refresh)
    )
    logger.info("Fetch complete; raw cache at %s", RAW_DIR)
    return 0


def _table_gaps(inputs: Inputs) -> list[str]:
    """Genes whose ClinPGx table the labels need but the cache cannot supply."""
    needed = plan_fetch(
        inputs.annotations, inputs.pharmvar, {d.gene for d in inputs.pharmvar}
    ).allele_table_genes
    index = read_table_index(RAW_DIR / "allele_tables")
    return sorted(
        g
        for g in needed
        if g not in index or index[g].status == "error" or g in inputs.failed_tables
    )


def _ingest(reset: bool) -> int:
    inputs = load_inputs()
    gaps = _table_gaps(inputs)
    if gaps:
        logger.error(
            "Ingest aborted before writing anything: ClinPGx allele tables missing or "
            "unreadable for %s; run 'python -m src.kg fetch'",
            gaps,
        )
        return 1
    run = IngestRun.start({"reset": str(reset), "raw_dir": str(RAW_DIR)})
    try:
        build = build_kg(
            annotations=inputs.annotations,
            definitions=inputs.pharmvar + inputs.tables,
            client=_client(offline=True),
            drug_cids=inputs.drug_cids,
            datasets=inputs.datasets,
            run=run,
            annotation_datasets=inputs.annotation_datasets,
        )
    except CacheMissError as e:
        logger.error("Ingest aborted before writing anything: %s", e)
        return 1
    build.run.finished_at = datetime.now(UTC)
    with KnowledgeGraph.connect() as kg:
        existing = kg.existing_runs()
        if existing and not reset:
            # Mixing runs would let stale elements from an older run survive.
            logger.error(
                "Graph already holds run(s) %s; rerun with --reset to rebuild it",
                existing,
            )
            return 1
        if reset:
            kg.reset()
        kg.ensure_schema()
        kg.write_build(build)
        audit = kg.audit()
    write_report(build, REPORT_DIR)
    return _report_audit(audit)


def _report_audit(audit: dict[str, int]) -> int:
    for name, count in audit.items():
        logger.info("audit %s: %d", name, count)
    if any(audit.values()):
        logger.error("Provenance audit failed: %s", audit)
        return 1
    logger.info("Provenance audit passed")
    return 0


def _export() -> int:
    snapshot = training_snapshot_path()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with KnowledgeGraph.connect() as kg:
        # Allele / annotation -> HGVS links come from Modephere's pgx export now
        # (the HGVS catalog); only the training rows are dev_Pgen's own.
        tables = {snapshot.name: to_training_frame(kg.training_rows())}
    for name, df in tables.items():
        df.write_parquet(snapshot.parent / name)
        logger.info("Exported %d rows to %s", len(df), snapshot.parent / name)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.kg",
        description="Build, audit and export the pharmacogenomic knowledge graph (Neo4j).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser(
        "fetch", help="Download dbSNP / NCBI / ClinPGx data into the raw cache."
    )
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="Re-request cached not_found / error responses (history kept, latest wins).",
    )
    ingest = sub.add_parser(
        "ingest", help="Build the graph offline and write it to Neo4j."
    )
    ingest.add_argument(
        "--reset", action="store_true", help="Delete the whole graph first."
    )
    sub.add_parser(
        "export", help="Write the training-row snapshot and standardization tables."
    )
    sub.add_parser("audit", help="Check that every element carries provenance.")
    sub.add_parser("stats", help="Node counts per label.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    configure_logging_level()
    logging.getLogger("src.kg").setLevel(logging.INFO)
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            return _fetch(args.refresh)
        if args.command == "ingest":
            return _ingest(args.reset)
        if args.command == "export":
            return _export()
        with KnowledgeGraph.connect() as kg:
            if args.command == "audit":
                return _report_audit(kg.audit())
            for label, n in kg.stats().items():
                logger.info("%s: %d", label, n)
    except PharmagenException as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
