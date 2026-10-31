"""Round-trip against a live Neo4j: build (from recorded fixtures) -> write -> audit -> export.

DESTRUCTIVE: the test resets the whole database. It only runs when
``PHARMAGEN_KG_TEST_ALLOW_RESET=1`` is set and a database is reachable with
``Settings.kg`` — point it at a disposable instance, e.g.
``docker compose -f docker-compose.kg.yml up -d``, then rebuild the real
graph afterwards with ``python -m src.kg ingest --reset``.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

from src.core import ResourceError
from src.kg.graph import KnowledgeGraph

from ..unit.kg.test_build import _build

pytestmark = pytest.mark.integration


@pytest.fixture
def kg():
    if os.environ.get("PHARMAGEN_KG_TEST_ALLOW_RESET") != "1":
        pytest.skip("set PHARMAGEN_KG_TEST_ALLOW_RESET=1 to run (resets the database)")
    try:
        graph = KnowledgeGraph.connect()
    except ResourceError as e:
        pytest.skip(f"Neo4j not reachable: {e}")
    graph.reset()
    yield graph
    graph.close()


def test_v2_round_trip(kg, tmp_path):
    build = _build(tmp_path)
    kg.ensure_schema()
    kg.write_build(build)
    kg.write_build(build)  # idempotent: MERGE + source union

    assert kg.audit() == {"nodes_missing": 0, "rels_missing": 0, "unknown_sources": 0}

    n = kg._run(
        "MATCH (:Haplotype {name: 'CYP2C19*2'})-[:HAS_VARIANT]->(v) RETURN count(v) AS n"
    ).records[0]["n"]
    assert n == 3
    sources = kg._run(
        "MATCH (v:Variant {spdi: 'NC_000010.11:94781858:G:A'}) RETURN v.sources AS s"
    ).records[0]["s"]
    assert len(sources) == len(set(sources))  # re-ingest did not duplicate sources

    av = kg.export_allele_variants()
    star = av.filter(pl.col("label") == "CYP2C19*2")
    assert "NC_000010.11:94781858:G:A" in star["spdi"].to_list()
    assert set(star["relation"].to_list()) == {"cis_member"}
    rsid = av.filter(pl.col("label") == "rs3892097")
    assert set(rsid["relation"].to_list()) == {"site_alternative"}
    assert len(rsid) == 3

    implicated = kg.export_annotation_variants()
    assert implicated.filter(pl.col("annotation_id") == "A_RS3892097_CT")[
        "spdi"
    ].to_list() == ["NC_000022.11:42128944:C:T"]
