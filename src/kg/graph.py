"""Neo4j access for the pharmacogenomic knowledge graph (model v2).

Graph model::

    (:Annotation)-[:CONCERNS_ALLELE]->(:Allele {label, kind, status, reason})
    (:Allele)-[:REFERS_TO]->(:RefSNP {rsid}) | (:Haplotype {name, gene, is_reference})
    (:RefSNP)-[:MERGED_INTO]->(:RefSNP)
    (:RefSNP)-[:HAS_ALLELE]->(:Variant)                 alternatives at one site
    (:Haplotype)-[:HAS_VARIANT {phase: "cis"}]->(:Variant)
    (:Haplotype)-[:SUBALLELE_OF]->(:Haplotype)
    (:Annotation)-[:IMPLICATES]->(:Variant)             allele-specific
    (:Variant {spdi, hgvs_g, seq_id, pos, ref, alt, assembly, vrs_id})-[:IN_GENE]->(:Gene)
    (:Annotation)-[:CONCERNS_GENE]->(:Gene), (:Annotation)-[:FOR_DRUG]->(:Drug)
    (:IngestRun)-[:USED]->(:Dataset)                    W3C PROV-O activity/entity

Every node and relationship carries ``sources`` (Dataset ids) and ``run_id``;
derived relationships also carry ``method`` and ``evidence``. Re-ingesting
unions ``sources`` instead of overwriting them. Labels and relationship
types are interpolated into Cypher only after an allow-list check.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Self

import polars as pl

from src.config import KGSettings, get_settings
from src.core import ResourceError

if TYPE_CHECKING:
    from types import TracebackType

    from neo4j import Driver

    from src.kg.build import KGBuild

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 1000

#: Label → its unique key property.
NODE_KEYS: dict[str, str] = {
    "Annotation": "id",
    "Allele": "label",
    "RefSNP": "rsid",
    "Haplotype": "name",
    "Variant": "spdi",
    "Gene": "symbol",
    "Drug": "name",
    "Dataset": "id",
    "IngestRun": "id",
}
SCHEMA_LABELS = frozenset(NODE_KEYS)
SCHEMA_REL_TYPES = frozenset(
    {
        "CONCERNS_ALLELE",
        "REFERS_TO",
        "MERGED_INTO",
        "HAS_ALLELE",
        "HAS_VARIANT",
        "SUBALLELE_OF",
        "IMPLICATES",
        "IN_GENE",
        "CONCERNS_GENE",
        "FOR_DRUG",
        "USED",
    }
)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    "DROP CONSTRAINT variant_id IF EXISTS",  # v1 keyed Variant on its raw label
    *(
        f"CREATE CONSTRAINT {label.lower()}_{key} IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        for label, key in NODE_KEYS.items()
    ),
    "CREATE INDEX drug_cid IF NOT EXISTS FOR (d:Drug) ON (d.cid)",
)

_UNION_SOURCES = (
    "CASE WHEN {v}.sources IS NULL THEN r.sources "
    "ELSE {v}.sources + [s IN r.sources WHERE NOT s IN {v}.sources] END"
)

#: One row per annotation in the shape of the training TSV (multi-value fields
#: as lists; see ``src.kg.training``). Drug name/CID lists stay aligned because
#: both are collected from the same ordered drug rows. ``implicated_hgvs`` is the
#: allele-specific variant(s) the annotation studied (canonical genomic HGVS).
TRAINING_ROWS = """
MATCH (a:Annotation)-[:FOR_DRUG]->(d:Drug)
WHERE d.cid IS NOT NULL AND ($sources IS NULL OR a.source IN $sources)
WITH a, d ORDER BY d.name
WITH a, collect(toString(d.cid)) AS drug_cids, collect(d.name) AS drug_names
MATCH (a)-[:CONCERNS_GENE]->(g:Gene)
WITH a, drug_cids, drug_names, g ORDER BY g.symbol
WITH a, drug_cids, drug_names, collect(g.symbol) AS genes
MATCH (a)-[:CONCERNS_ALLELE]->(al:Allele)
WITH a, drug_cids, drug_names, genes, al ORDER BY al.label
WITH a, drug_cids, drug_names, genes, collect(al.label) AS genotypes
OPTIONAL MATCH (a)-[:IMPLICATES]->(v:Variant)
WITH a, drug_cids, drug_names, genes, genotypes, v ORDER BY v.hgvs_g
WITH a, drug_cids, drug_names, genes, genotypes,
     collect(DISTINCT v.hgvs_g) AS implicated_hgvs
RETURN a.id AS annotation_id,
       a.source AS source,
       a.pmid AS pmid,
       drug_cids,
       drug_names,
       genes,
       genotypes,
       implicated_hgvs,
       a.alleles AS alleles,
       a.comparison_alleles AS comparison_alleles,
       coalesce(a.phenotype_categories, []) AS phenotype_categories,
       a.direction AS direction,
       a.significance AS significance,
       a.associated AS associated
ORDER BY annotation_id
"""

EXPORT_ALLELE_VARIANTS = """
MATCH (al:Allele)
OPTIONAL MATCH (al)-[:REFERS_TO]->(t)
OPTIONAL MATCH (t)-[:MERGED_INTO]->(cur:RefSNP)
WITH al, t, coalesce(cur, t) AS site
OPTIONAL MATCH (site)-[hv:HAS_ALLELE|HAS_VARIANT]->(v:Variant)
RETURN al.label AS label, al.kind AS kind, al.status AS status, al.reason AS reason,
       v.spdi AS spdi, v.hgvs_g AS hgvs_g,
       CASE WHEN t:RefSNP THEN 'rsid' WHEN t:Haplotype THEN 'haplotype' END AS path,
       CASE WHEN hv IS NULL THEN null
            WHEN type(hv) = 'HAS_VARIANT' THEN 'cis_member'
            ELSE 'site_alternative' END AS relation,
       coalesce(hv.sources, al.sources) AS sources,
       coalesce(hv.run_id, al.run_id) AS run_id
ORDER BY label, spdi
"""

EXPORT_ANNOTATION_VARIANTS = """
MATCH (a:Annotation)-[r:IMPLICATES]->(v:Variant)
RETURN a.id AS annotation_id, v.spdi AS spdi, v.hgvs_g AS hgvs_g,
       r.method AS method, r.evidence AS evidence, r.sources AS sources, r.run_id AS run_id
ORDER BY annotation_id, spdi
"""

AUDIT = """
CALL () { MATCH (n) WHERE n.sources IS NULL OR size(n.sources) = 0 OR n.run_id IS NULL
          RETURN count(n) AS nodes_missing }
CALL () { MATCH ()-[r]->() WHERE r.sources IS NULL OR size(r.sources) = 0 OR r.run_id IS NULL
          RETURN count(r) AS rels_missing }
CALL () { CALL () { MATCH (n) UNWIND n.sources AS s RETURN s
                    UNION
                    MATCH ()-[r]->() UNWIND r.sources AS s RETURN s }
          WITH DISTINCT s WHERE NOT EXISTS { MATCH (:Dataset {id: s}) }
          RETURN count(s) AS unknown_sources }
RETURN nodes_missing, rels_missing, unknown_sources
"""

STATS = """
MATCH (n) WITH labels(n)[0] AS label, count(*) AS n RETURN label, n ORDER BY label
"""

RESET = "MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS"

TRAINING_ROW_SCHEMA = {
    "annotation_id": pl.String,
    "source": pl.String,
    "pmid": pl.String,
    "drug_cids": pl.List(pl.String),
    "drug_names": pl.List(pl.String),
    "genes": pl.List(pl.String),
    "genotypes": pl.List(pl.String),
    "implicated_hgvs": pl.List(pl.String),
    "alleles": pl.String,
    "comparison_alleles": pl.String,
    "phenotype_categories": pl.List(pl.String),
    "direction": pl.String,
    "significance": pl.String,
    "associated": pl.Boolean,
}
_ALLELE_VARIANT_SCHEMA = {
    "label": pl.String,
    "kind": pl.String,
    "status": pl.String,
    "reason": pl.String,
    "spdi": pl.String,
    "hgvs_g": pl.String,
    "path": pl.String,
    "relation": pl.String,
    "sources": pl.List(pl.String),
    "run_id": pl.String,
}
_ANNOTATION_VARIANT_SCHEMA = {
    "annotation_id": pl.String,
    "spdi": pl.String,
    "hgvs_g": pl.String,
    "method": pl.String,
    "evidence": pl.String,
    "sources": pl.List(pl.String),
    "run_id": pl.String,
}


def _check_label(label: str) -> str:
    if label not in SCHEMA_LABELS:
        msg = f"unknown node label {label!r}"
        raise ValueError(msg)
    return label


def _check_rel_type(rel_type: str) -> str:
    if rel_type not in SCHEMA_REL_TYPES:
        msg = f"unknown relationship type {rel_type!r}"
        raise ValueError(msg)
    return rel_type


def _node_query(label: str) -> str:
    key = NODE_KEYS[label]
    return (
        f"UNWIND $rows AS r MERGE (n:{label} {{{key}: r.key}}) "
        f"SET n += r.props, n.run_id = r.run_id, n.sources = {_UNION_SOURCES.format(v='n')}"
    )


def _rel_query(rel_type: str, start: str, end: str) -> str:
    return (
        f"UNWIND $rows AS r "
        f"MATCH (a:{start} {{{NODE_KEYS[start]}: r.start}}) "
        f"MATCH (b:{end} {{{NODE_KEYS[end]}: r.end}}) "
        f"MERGE (a)-[x:{rel_type}]->(b) "
        f"SET x += r.props, x.run_id = r.run_id, x.method = r.method, "
        f"x.evidence = r.evidence, x.sources = {_UNION_SOURCES.format(v='x')} "
        f"RETURN count(x) AS written"
    )


def _cypher_props(props: dict[str, object]) -> dict[str, object]:
    """Neo4j properties: datetimes as ISO strings, dicts as 'k=v' strings."""
    out: dict[str, object] = {}
    for k, v in props.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()  # type: ignore[union-attr]
        elif isinstance(v, dict):
            out[k] = [f"{a}={b}" for a, b in sorted(v.items())]
        else:
            out[k] = v
    return out


class KnowledgeGraph:
    """Thin wrapper over a Neo4j driver. Use as a context manager."""

    def __init__(self, driver: Driver, database: str) -> None:
        self._driver = driver
        self._database = database

    @classmethod
    def connect(cls, settings: KGSettings | None = None) -> KnowledgeGraph:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable

        s = settings or get_settings().kg
        driver = GraphDatabase.driver(
            s.uri, auth=(s.user, s.password.get_secret_value())
        )
        try:
            driver.verify_connectivity()
        except (ServiceUnavailable, AuthError) as e:
            driver.close()
            msg = f"Cannot reach Neo4j at {s.uri} as {s.user!r}: {e}"
            raise ResourceError(msg) from e
        logger.info("Connected to Neo4j at %s (database=%s)", s.uri, s.database)
        return cls(driver, s.database)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _run(self, query: str, **params: object):
        return self._driver.execute_query(query, params, database_=self._database)

    # ----- schema / lifecycle --------------------------------------------- #

    def ensure_schema(self) -> None:
        for statement in SCHEMA_STATEMENTS:
            self._run(statement)
        logger.info("KG schema ensured (%d statements)", len(SCHEMA_STATEMENTS))

    def reset(self) -> None:
        """Delete every node and relationship (batched auto-commit transactions)."""
        with self._driver.session(database=self._database) as session:
            session.run(RESET).consume()
        logger.info("KG reset: all nodes deleted")

    # ----- writing --------------------------------------------------------- #

    def write_build(
        self, build: KGBuild, *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """Write datasets, the run, then all nodes and relationships of ``build``."""
        nodes: dict[str, list[dict]] = defaultdict(list)
        rels: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for n in build.nodes:
            label = _check_label(n.label)
            nodes[label].append(
                {
                    "key": n.key[NODE_KEYS[label]],
                    "props": _cypher_props(n.props),
                    **n.stamp.props(),
                }
            )
        for r in build.rels:
            key = (
                _check_rel_type(r.type),
                _check_label(r.start_label),
                _check_label(r.end_label),
            )
            rels[key].append(
                {
                    "start": r.start_key[NODE_KEYS[r.start_label]],
                    "end": r.end_key[NODE_KEYS[r.end_label]],
                    "props": _cypher_props(r.props),
                    "method": r.stamp.method,
                    "evidence": r.stamp.evidence,
                    "sources": list(r.stamp.sources),
                    "run_id": r.stamp.run_id,
                }
            )

        run = build.run
        dataset_ids = [d.id for d in build.datasets]
        self._write_nodes(
            "Dataset",
            [
                {
                    "key": d.id,
                    "props": _cypher_props(d.model_dump(exclude={"id"})),
                    "sources": [d.id],
                    "run_id": run.id,
                }
                for d in build.datasets
            ],
            batch_size,
        )
        self._write_nodes(
            "IngestRun",
            [
                {
                    "key": run.id,
                    "props": _cypher_props(run.model_dump(exclude={"id"})),
                    "sources": dataset_ids,
                    "run_id": run.id,
                }
            ],
            batch_size,
        )
        self._write_rels(
            "USED",
            "IngestRun",
            "Dataset",
            [
                {
                    "start": run.id,
                    "end": d,
                    "props": {},
                    "method": None,
                    "evidence": None,
                    "sources": [d],
                    "run_id": run.id,
                }
                for d in dataset_ids
            ],
            batch_size,
        )
        for label, rows in nodes.items():
            self._write_nodes(label, rows, batch_size)
        for (rel_type, start, end), rows in rels.items():
            self._write_rels(rel_type, start, end, rows, batch_size)
        logger.info(
            "Wrote run %s: %d datasets, %d nodes, %d relationships",
            run.id,
            len(build.datasets),
            len(build.nodes),
            len(build.rels),
        )

    def _write_nodes(self, label: str, rows: list[dict], batch_size: int) -> None:
        query = _node_query(label)
        for start in range(0, len(rows), batch_size):
            self._run(query, rows=rows[start : start + batch_size])

    def _write_rels(
        self, rel_type: str, start: str, end: str, rows: list[dict], batch_size: int
    ) -> None:
        query = _rel_query(rel_type, start, end)
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            records = self._run(query, rows=batch).records
            written = records[0]["written"] if records else 0
            if written != len(batch):
                msg = (
                    f"{rel_type} ({start}->{end}): wrote {written} of {len(batch)} "
                    "relationships — an endpoint node is missing"
                )
                raise RuntimeError(msg)

    # ----- reading --------------------------------------------------------- #

    def audit(self) -> dict[str, int]:
        records = self._run(AUDIT).records
        return dict(records[0].data()) if records else {}

    def existing_runs(self) -> list[str]:
        """Ids of the IngestRuns already in the graph."""
        records = self._run("MATCH (r:IngestRun) RETURN r.id AS id ORDER BY id").records
        return [r["id"] for r in records]

    def stats(self) -> dict[str, int]:
        return {r["label"]: r["n"] for r in self._run(STATS).records}

    def training_rows(self, sources: list[str] | None = None) -> pl.DataFrame:
        """Annotation-level training rows (raw lists; see ``src.kg.training``).

        ``sources`` restricts to PharmGKB files (``AnnotationSource`` values).
        """
        records = self._run(TRAINING_ROWS, sources=sources).records
        return pl.DataFrame([r.data() for r in records], schema=TRAINING_ROW_SCHEMA)

    def export_allele_variants(self) -> pl.DataFrame:
        """Allele label → canonical variants (the standardization table)."""
        records = self._run(EXPORT_ALLELE_VARIANTS).records
        return pl.DataFrame([r.data() for r in records], schema=_ALLELE_VARIANT_SCHEMA)

    def export_annotation_variants(self) -> pl.DataFrame:
        records = self._run(EXPORT_ANNOTATION_VARIANTS).records
        return pl.DataFrame(
            [r.data() for r in records], schema=_ANNOTATION_VARIANT_SCHEMA
        )


__all__ = [
    "EXPORT_ALLELE_VARIANTS",
    "EXPORT_ANNOTATION_VARIANTS",
    "NODE_KEYS",
    "SCHEMA_LABELS",
    "SCHEMA_REL_TYPES",
    "SCHEMA_STATEMENTS",
    "TRAINING_ROWS",
    "TRAINING_ROW_SCHEMA",
    "KnowledgeGraph",
]
