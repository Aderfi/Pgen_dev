"""Tests for the Neo4j v2 writer, against a fake driver (no database needed)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from src.kg.build import KGBuild, NodeRow, RelRow
from src.kg.graph import (
    SCHEMA_LABELS,
    SCHEMA_REL_TYPES,
    SCHEMA_STATEMENTS,
    TRAINING_ROW_SCHEMA,
    TRAINING_ROWS,
    KnowledgeGraph,
)
from src.kg.provenance import Dataset, IngestRun, Stamp
from src.kg.training import UNDETERMINED_DIRECTION, to_training_frame


class _Record:
    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        return self._data

    def __getitem__(self, key: str) -> object:
        return self._data[key]


class _FakeSession:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver

    def run(self, query: str):
        self.driver.session_queries.append(query)
        return SimpleNamespace(consume=lambda: None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(
        self, records: list[dict] | None = None, short_rel: str | None = None
    ) -> None:
        self.calls: list[tuple[str, dict, str]] = []
        self.session_queries: list[str] = []
        self._records = records
        self.short_rel = short_rel
        self.closed = False

    def execute_query(self, query, params, database_):
        self.calls.append((query, params, database_))
        if self._records is not None:
            return SimpleNamespace(records=[_Record(r) for r in self._records])
        if "rows" in params and "MERGE (a)-" in query:
            short = self.short_rel is not None and f":{self.short_rel}]" in query
            n = len(params["rows"]) - (1 if short else 0)
            return SimpleNamespace(records=[_Record({"written": n})])
        return SimpleNamespace(records=[])

    def session(self, database: str):  # noqa: ARG002
        return _FakeSession(self)

    def close(self) -> None:
        self.closed = True


NOW = datetime(2026, 10, 18, tzinfo=UTC)


def _stamp(*sources: str) -> Stamp:
    return Stamp(sources=sources or ("pharmgkb:var_drug_ann",), run_id="run1")


def small_build(nodes=None, rels=None) -> KGBuild:
    ds = Dataset(
        id="pharmgkb:var_drug_ann",
        name="n",
        version="v",
        uri="u",
        retrieved_at=NOW,
        sha256="0" * 64,
    )
    run = IngestRun(id="run1", started_at=NOW, code_version="abc", dirty=False)
    return KGBuild(
        datasets=[ds],
        run=run,
        nodes=nodes
        or [
            NodeRow(
                label="Annotation",
                key={"id": "a1"},
                props={"pmid": "1"},
                stamp=_stamp(),
            ),
            NodeRow(label="Gene", key={"symbol": "CYP2D6"}, props={}, stamp=_stamp()),
        ],
        rels=rels
        or [
            RelRow(
                type="CONCERNS_GENE",
                start_label="Annotation",
                start_key={"id": "a1"},
                end_label="Gene",
                end_key={"symbol": "CYP2D6"},
                stamp=_stamp(),
            )
        ],
        resolution=[],
        disagreements=[],
        inference_counts={},
    )


def test_write_build_writes_provenance_nodes_first():
    driver = _FakeDriver()
    KnowledgeGraph(driver, "neo4j").write_build(small_build())
    queries = [q for q, _, _ in driver.calls]
    assert "MERGE (n:Dataset" in queries[0]
    assert any("MERGE (n:IngestRun" in q for q in queries)
    assert any(":USED" in q for q in queries)


def test_every_write_sets_sources_and_run_id():
    driver = _FakeDriver()
    KnowledgeGraph(driver, "neo4j").write_build(small_build())
    for q, params, _ in driver.calls:
        if "rows" in params:
            assert "sources" in q and "run_id" in q
            for row in params["rows"]:
                assert row["sources"] and row["run_id"]


def test_sources_are_unioned_not_overwritten():
    driver = _FakeDriver()
    KnowledgeGraph(driver, "neo4j").write_build(small_build())
    node_query = next(q for q, _, _ in driver.calls if "MERGE (n:Annotation" in q)
    assert "NOT s IN n.sources" in node_query


def test_unknown_label_rejected_before_any_write():
    bad = small_build(
        nodes=[NodeRow(label="Evil) DETACH DELETE (n", key={"id": "x"}, stamp=_stamp())]
    )
    driver = _FakeDriver()
    with pytest.raises(ValueError, match="label"):
        KnowledgeGraph(driver, "neo4j").write_build(bad)
    assert driver.calls == []


def test_unknown_rel_type_rejected():
    bad = small_build(
        rels=[
            RelRow(
                type="X]->() DELETE",
                start_label="Gene",
                start_key={"symbol": "A"},
                end_label="Gene",
                end_key={"symbol": "B"},
                stamp=_stamp(),
            )
        ]
    )
    with pytest.raises(ValueError, match="relationship type"):
        KnowledgeGraph(_FakeDriver(), "neo4j").write_build(bad)


def test_relationship_with_missing_endpoint_is_an_error():
    with pytest.raises(RuntimeError, match="CONCERNS_GENE"):
        KnowledgeGraph(_FakeDriver(short_rel="CONCERNS_GENE"), "neo4j").write_build(
            small_build()
        )


def test_schema_covers_every_label():
    joined = "\n".join(SCHEMA_STATEMENTS)
    for label in SCHEMA_LABELS:
        assert f"(n:{label})" in joined
    assert "DROP CONSTRAINT variant_id IF EXISTS" in joined


def test_rel_types_allow_list():
    assert {
        "HAS_VARIANT",
        "HAS_ALLELE",
        "IMPLICATES",
        "REFERS_TO",
        "USED",
    } <= SCHEMA_REL_TYPES


def test_reset_uses_auto_commit_batches():
    driver = _FakeDriver()
    KnowledgeGraph(driver, "neo4j").reset()
    assert driver.session_queries and "IN TRANSACTIONS" in driver.session_queries[0]


def test_audit_counts():
    driver = _FakeDriver(
        records=[{"nodes_missing": 0, "rels_missing": 2, "unknown_sources": 1}]
    )
    assert KnowledgeGraph(driver, "neo4j").audit() == {
        "nodes_missing": 0,
        "rels_missing": 2,
        "unknown_sources": 1,
    }


def _training_record(**overrides):
    record = {
        "annotation_id": "1449",
        "source": "var_drug_ann",
        "pmid": "123",
        "drug_cids": ["2907", "5460033"],
        "drug_names": ["cyclophosphamide", "cisplatin"],
        "genes": ["XRCC1"],
        "genotypes": ["rs25487"],
        "implicated_hgvs": [],
        "alleles": "CC",
        "comparison_alleles": "CT + TT",
        "phenotype_categories": ["Efficacy", "Toxicity"],
        "direction": None,
        "significance": "yes",
        "associated": True,
    }
    return record | overrides


def test_training_rows_query_and_schema():
    driver = _FakeDriver([_training_record()])
    df = KnowledgeGraph(driver, "neo4j").training_rows(["var_drug_ann"])
    query, params, _db = driver.calls[0]
    assert query == TRAINING_ROWS
    assert params["sources"] == ["var_drug_ann"]
    assert df.schema == pl.Schema(TRAINING_ROW_SCHEMA)
    assert df.row(0, named=True) == _training_record()


def test_training_frame_matches_tsv_conventions():
    raw = pl.DataFrame(
        [_training_record(), _training_record(annotation_id="7", associated=False)],
        schema=TRAINING_ROW_SCHEMA,
    )
    row = to_training_frame(raw).row(0, named=True)
    assert row["drugs_cid"] == "2907|5460033"
    assert row["drugs"] == "cyclophosphamide|cisplatin"
    assert row["gene"] == "XRCC1"
    assert row["genotype"] == "rs25487"
    assert row["phenotype_category"] == "Efficacy|Toxicity"
    assert row["direction_of_effect"] == UNDETERMINED_DIRECTION
    assert row["is/is_not_associated"] == "Associated with"
    assert row["variant_annotation_id"] == 1449
    second = to_training_frame(raw).row(1, named=True)
    assert second["is/is_not_associated"] == "Not associated with"


def test_training_frame_prefers_implicated_hgvs():
    hgvs = "NC_000019.10:g.43551574T>C"
    raw = pl.DataFrame(
        [_training_record(implicated_hgvs=[hgvs])], schema=TRAINING_ROW_SCHEMA
    )
    assert to_training_frame(raw).row(0, named=True)["genotype"] == hgvs


def test_export_allele_variants_schema():
    record = {
        "label": "CYP2C19*2",
        "kind": "star",
        "status": "resolved",
        "reason": None,
        "spdi": "NC_000010.11:94781858:G:A",
        "hgvs_g": "NC_000010.11:g.94781859G>A",
        "path": "haplotype",
        "relation": "cis_member",
        "sources": ["pharmvar:x"],
        "run_id": "r",
    }
    df = KnowledgeGraph(_FakeDriver([record]), "neo4j").export_allele_variants()
    assert df.row(0, named=True) == record


def test_context_manager_closes_driver():
    driver = _FakeDriver()
    with KnowledgeGraph(driver, "neo4j"):
        pass
    assert driver.closed


def test_audit_checks_relationship_sources_too():
    from src.kg.graph import AUDIT

    assert "MATCH ()-[r]->() UNWIND r.sources" in AUDIT


def test_existing_runs():
    driver = _FakeDriver(records=[{"id": "old"}])
    assert KnowledgeGraph(driver, "neo4j").existing_runs() == ["old"]
