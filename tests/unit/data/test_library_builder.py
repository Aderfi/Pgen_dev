"""The library build: which ChEMBL IDs are built, and stale-toolkit rebuilds."""

from __future__ import annotations

import logging

import polars as pl
from modephere.library import BuildSummary as DrugSummary
from modephere.pgx.export import ANNOTATION_ROW_SCHEMA

from src.config import ModephereSettings
from src.data.library import bridge
from src.data.library import builder as builder_module
from src.data.library.config import LibraryBuildConfig
from tests.conftest import make_modephere_library


def _drug(name, chembl_id=None, atc_codes=(), components=()):
    return {
        "name": name,
        "cid": None,
        "chembl_id": chembl_id,
        "inchikey": None,
        "atc_codes": list(atc_codes),
        "components": [{"name": n, "chembl_id": c} for n, c in components],
    }


def _export(s: ModephereSettings, *annotations: list[dict]) -> None:
    s.pgx_export.mkdir(parents=True, exist_ok=True)
    rows = [
        dict.fromkeys(ANNOTATION_ROW_SCHEMA)
        | {
            "annotation_id": str(i),
            "drugs": drugs,
            "genes": [],
            "genotypes": [],
            "implicated_hgvs": [],
            "phenotype_categories": [],
        }
        for i, drugs in enumerate(annotations)
    ]
    pl.DataFrame(rows, schema=ANNOTATION_ROW_SCHEMA).write_parquet(
        s.pgx_export / "annotation_rows.parquet"
    )
    pl.DataFrame(
        {
            "atc_code": ["N02A", "N02A", "C09AA"],
            "chembl_id": ["CHEMBL70", "CHEMBL1201", "CHEMBL1237"],
        }
    ).write_parquet(s.pgx_export / "atc_members.parquet")


def _settings(tmp_path) -> ModephereSettings:
    return ModephereSettings(
        library=tmp_path / "library",
        pgx_export=tmp_path / "pgx",
        chembl=tmp_path,
        logs=tmp_path / "logs",
    )


def test_catalog_ids_include_components_and_class_members(tmp_path):
    s = _settings(tmp_path)
    _export(
        s,
        [_drug("aspirin", "CHEMBL25"), _drug("opioids", atc_codes=["N02A"])],
        [
            _drug(
                "sulfamethoxazole / trimethoprim",
                components=[("sulfamethoxazole", "CHEMBL443"), ("trimethoprim", None)],
            )
        ],
        [_drug("radiotherapy")],
    )
    assert bridge.annotation_chembl_ids(s) == [
        "CHEMBL1201",
        "CHEMBL25",
        "CHEMBL443",
        "CHEMBL70",
    ]


class _FakeBuilder:
    runs: list[tuple[tuple[str, ...], bool]] = []

    def __init__(self, config) -> None:  # noqa: ANN001
        self.config = config
        self.failures: list = []
        self.degraded: list = []

    def run(self) -> DrugSummary:
        ids = tuple(self.config.chembl_ids)
        _FakeBuilder.runs.append((ids, self.config.force))
        return DrugSummary(
            queries=len(ids), built=len(ids), skipped=0, duplicates=0, failed=0
        )


def test_stale_toolkit_compounds_are_forced(tmp_path, monkeypatch, caplog):
    s = _settings(tmp_path)
    store, _ = make_modephere_library(
        s.library, {"CHEMBL25": "CC(=O)Oc1ccccc1C(=O)O", "CHEMBL545": "CCO"}
    )
    compounds = store.read_table("compounds").with_columns(
        pl.when(pl.col("chembl_id") == "CHEMBL25")
        .then(pl.lit("2025.09.6"))
        .otherwise(pl.col("toolkit_version"))
        .alias("toolkit_version")
    )
    compounds.write_parquet(store.table_path("compounds"))
    _export(
        s,
        [_drug("aspirin", "CHEMBL25"), _drug("ethanol", "CHEMBL545")],
        [_drug("opioids", atc_codes=["N02A"])],
    )
    monkeypatch.setattr(bridge, "latest_staging", lambda root: tmp_path / "chembl_37")
    monkeypatch.setattr(builder_module, "ModephereLibraryBuilder", _FakeBuilder)
    _FakeBuilder.runs = []
    cfg = LibraryBuildConfig(
        modephere=s,
        library_root=tmp_path,
        star_alleles_tsv=tmp_path / "star.tsv",
        skip_admet=True,
        skip_genes=True,
    )
    with caplog.at_level(logging.INFO):
        summary = builder_module.LibraryBuilder(cfg).run()
    forced = [ids for ids, force in _FakeBuilder.runs if force]
    normal = [ids for ids, force in _FakeBuilder.runs if not force]
    assert forced == [("CHEMBL25",)]
    assert normal == [("CHEMBL1201", "CHEMBL545", "CHEMBL70")]
    assert summary.drugs is not None and summary.drugs.queries == 4
    assert summary.stale_rebuilt == 1
    # neither N02A member is in the (fake-built) library
    assert "class ATC:N02A: 0 of 2 members built" in caplog.text
