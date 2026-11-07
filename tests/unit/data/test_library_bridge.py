"""The Modephere bridge: aliases, ATC members and the heavy-atom cap."""

from __future__ import annotations

import polars as pl
import pytest

from src.config import ModephereSettings
from src.data.library import bridge
from tests.conftest import make_modephere_library


def _settings(tmp_path, **kw) -> ModephereSettings:
    return ModephereSettings(
        library=tmp_path / "library", pgx_export=tmp_path / "pgx", **kw
    )


def test_chembl_index_merges_aliases(tmp_path):
    store, keys = make_modephere_library(
        tmp_path / "library", {"CHEMBL1": "CC(=O)Oc1ccccc1C(=O)O"}
    )
    k1 = keys["CHEMBL1"]
    store.upsert_aliases(
        [("CHEMBL2", k1, "chembl_id:CHEMBL2"), ("CHEMBL1", "K9", "chembl_id:CHEMBL1")]
    )
    assert bridge.chembl_index(store) == {"CHEMBL1": k1, "CHEMBL2": k1}


def test_class_entity_key_sorts_and_dedupes():
    assert bridge.class_entity_key(["H02A", "A01AC", "H02A"]) == "ATC:A01AC,H02A"


def test_class_members_union():
    members = {
        "N02A": frozenset({"CHEMBL70"}),
        "H02A": frozenset({"CHEMBL131"}),
        "A01AC": frozenset({"CHEMBL131", "CHEMBL384467"}),
    }
    assert bridge.class_members("ATC:A01AC,H02A", members) == frozenset(
        {"CHEMBL131", "CHEMBL384467"}
    )
    assert bridge.class_members("ATC:X99", members) == frozenset()
    with pytest.raises(ValueError, match="not a class entity"):
        bridge.class_members("CHEMBL70", members)


def test_atc_members_reads_the_export(tmp_path):
    s = _settings(tmp_path)
    s.pgx_export.mkdir(parents=True)
    pl.DataFrame(
        {
            "atc_code": ["N02A", "N02A", "N02AA"],
            "chembl_id": ["CHEMBL70", "CHEMBL1201", "CHEMBL70"],
        }
    ).write_parquet(s.pgx_export / "atc_members.parquet")
    assert bridge.atc_members(s) == {
        "N02A": frozenset({"CHEMBL70", "CHEMBL1201"}),
        "N02AA": frozenset({"CHEMBL70"}),
    }


def test_build_config_passes_heavy_atom_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "latest_staging", lambda root: tmp_path / "chembl_37")
    s = _settings(tmp_path, chembl=tmp_path, max_heavy_atoms=300)
    cfg = bridge.library_build_config(chembl_ids=["CHEMBL25"], settings=s)
    assert cfg.pipeline.standardize.max_heavy_atoms == 300
    assert ModephereSettings().max_heavy_atoms == 300


def test_annotation_drugs_carry_classes_and_components(tmp_path):
    from modephere.pgx.export import ANNOTATION_ROW_SCHEMA

    s = _settings(tmp_path)
    s.pgx_export.mkdir(parents=True)
    drug = {
        "name": "sulfamethoxazole / trimethoprim",
        "cid": None,
        "chembl_id": None,
        "inchikey": None,
        "atc_codes": [],
        "components": [
            {"name": "sulfamethoxazole", "chembl_id": "CHEMBL443"},
            {"name": "trimethoprim", "chembl_id": "CHEMBL22"},
        ],
    }
    row = dict.fromkeys(ANNOTATION_ROW_SCHEMA) | {
        "annotation_id": "1",
        "drugs": [drug],
        "genes": [],
        "genotypes": [],
        "implicated_hgvs": [],
        "phenotype_categories": [],
    }
    pl.DataFrame([row], schema=ANNOTATION_ROW_SCHEMA).write_parquet(
        s.pgx_export / "annotation_rows.parquet"
    )
    frame = bridge.annotation_drugs(s)
    assert frame.columns == [
        "annotation_id",
        "position",
        "name",
        "chembl_id",
        "atc_codes",
        "components",
    ]
    assert [c["chembl_id"] for c in frame["components"][0]] == ["CHEMBL443", "CHEMBL22"]


def test_annotation_drugs_reads_a_0_1_2_export(tmp_path):
    # final review #2: exports written before Modephere 0.1.3 lack the drug
    # struct's atc_codes / components
    s = _settings(tmp_path)
    s.pgx_export.mkdir(parents=True)
    old_drug = pl.Struct(
        {
            "name": pl.String,
            "cid": pl.Int64,
            "chembl_id": pl.String,
            "inchikey": pl.String,
        }
    )
    pl.DataFrame(
        {
            "annotation_id": ["1"],
            "drugs": [
                [
                    {
                        "name": "aspirin",
                        "cid": 2244,
                        "chembl_id": "CHEMBL25",
                        "inchikey": None,
                    }
                ]
            ],
        },
        schema={"annotation_id": pl.String, "drugs": pl.List(old_drug)},
    ).write_parquet(s.pgx_export / "annotation_rows.parquet")
    frame = bridge.annotation_drugs(s)
    assert frame["chembl_id"].to_list() == ["CHEMBL25"]
    assert frame["atc_codes"].to_list() == [[]]
    assert frame["components"].to_list() == [[]]
    assert bridge.annotation_chembl_ids(s) == ["CHEMBL25"]
