"""Tests for the resolution report and the KG command line."""

from __future__ import annotations

import logging

import polars as pl

import src.kg.__main__ as cli
from src.kg.build import DisagreementRow, ResolutionRow
from src.kg.report import write_report

from .test_graph import small_build


def _build_with_rows():
    b = small_build()
    return b.model_copy(
        update={
            "resolution": [
                ResolutionRow(
                    label="CYP2C19*2",
                    kind="star",
                    status="resolved",
                    reason=None,
                    n_annotations=3,
                    n_variants=3,
                    path="haplotype",
                ),
                ResolutionRow(
                    label="HLA-B*58:01",
                    kind="hla",
                    status="unresolved",
                    reason="structural_allele",
                    n_annotations=2,
                    n_variants=0,
                    path=None,
                ),
            ],
            "disagreements": [
                DisagreementRow(
                    issue="allele_not_in_dbsnp",
                    haplotype="X*2",
                    rsid="rs1",
                    pharmvar_spdi="NC_000010.11:1:G:A",
                    dbsnp_spdis=("NC_000010.11:1:G:T",),
                ),
            ],
            "inference_counts": {"linked": 8, "allele_strand_mismatch": 2},
        }
    )


def test_report_files_and_columns(tmp_path):
    path = write_report(_build_with_rows(), tmp_path)
    df = pl.read_csv(path, separator="\t")
    assert set(df.columns) >= {
        "label",
        "kind",
        "status",
        "reason",
        "n_annotations",
        "n_variants",
        "path",
    }
    assert len(df) == 2
    dis = pl.read_csv(tmp_path / "disagreements.tsv", separator="\t")
    assert dis["issue"].to_list() == ["allele_not_in_dbsnp"]
    assert dis["dbsnp_spdis"].to_list() == ["NC_000010.11:1:G:T"]


def test_report_logs_strand_mismatch_rate(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="src.kg.report"):
        write_report(_build_with_rows(), tmp_path)
    assert "strand mismatches 2/10 (0.2000)" in caplog.text
    assert "resolved mentions 3/5" in caplog.text


def test_ingest_cache_miss_writes_nothing(tmp_path, monkeypatch):
    connected: list[bool] = []
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(
        cli.KnowledgeGraph,
        "connect",
        classmethod(lambda cls, *a: connected.append(True)),
    )
    monkeypatch.setattr(
        cli,
        "load_inputs",
        lambda: cli.Inputs(
            annotations=[
                cli.PGxAnnotation(id="a", source="var_drug_ann", variants=["rs4244285"])
            ],
            pharmvar=[],
            tables=[],
            datasets=[],
            drug_cids={},
        ),
    )
    assert cli.main(["ingest"]) == 1
    assert connected == []


class _FakeKG:
    def __init__(self, runs):
        self.runs = runs
        self.written = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def existing_runs(self):
        return self.runs

    def reset(self):
        self.runs = []

    def ensure_schema(self):
        pass

    def write_build(self, build):
        self.written.append(build)

    def audit(self):
        return {"nodes_missing": 0, "rels_missing": 0, "unknown_sources": 0}


def _empty_inputs(labels=()):
    return cli.Inputs(
        annotations=[
            cli.PGxAnnotation(id="a", source="var_drug_ann", variants=list(labels))
        ],
        pharmvar=[],
        tables=[],
        datasets=[],
        drug_cids={},
    )


def test_ingest_without_reset_refuses_to_mix_runs(tmp_path, monkeypatch):
    fake = _FakeKG(["old-run"])
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cli, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(cli, "load_inputs", _empty_inputs)
    monkeypatch.setattr(cli, "build_kg", lambda **kw: small_build())
    monkeypatch.setattr(
        cli.KnowledgeGraph, "connect", classmethod(lambda cls, *a: fake)
    )
    assert cli.main(["ingest"]) == 1
    assert fake.written == []
    assert cli.main(["ingest", "--reset"]) == 0
    assert len(fake.written) == 1


def test_ingest_fails_when_a_needed_allele_table_is_missing(tmp_path, monkeypatch):
    connected: list[bool] = []
    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cli, "load_inputs", lambda: _empty_inputs(["TPMT*3A"]))
    monkeypatch.setattr(cli, "build_kg", lambda **kw: small_build())
    monkeypatch.setattr(
        cli.KnowledgeGraph,
        "connect",
        classmethod(lambda cls, *a: connected.append(True)),
    )
    assert cli.main(["ingest"]) == 1
    assert connected == []
