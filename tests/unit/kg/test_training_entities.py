"""Training rows read each annotation's drugs as entities (drug, component, class)."""

from __future__ import annotations

import polars as pl

from src.kg.graph import TRAINING_ROW_SCHEMA
from src.kg.training import to_training_frame
from tests.unit.kg.test_graph import _training_record

_DRUG_SCHEMA = {
    "annotation_id": pl.String,
    "position": pl.Int64,
    "name": pl.String,
    "chembl_id": pl.String,
    "atc_codes": pl.List(pl.String),
    "components": pl.List(pl.Struct({"name": pl.String, "chembl_id": pl.String})),
}


def _frame(*drugs: tuple) -> pl.DataFrame:
    ids = sorted({d[0] for d in drugs})
    raw = pl.DataFrame(
        [_training_record(annotation_id=i) for i in ids], schema=TRAINING_ROW_SCHEMA
    )
    table = pl.DataFrame(list(drugs), schema=_DRUG_SCHEMA, orient="row")
    return to_training_frame(raw, table)


def _component(name: str, chembl_id: str | None) -> dict:
    return {"name": name, "chembl_id": chembl_id}


def test_combination_becomes_component_entities():
    frame = _frame(
        (
            "1",
            0,
            "sulfamethoxazole / trimethoprim",
            None,
            [],
            [
                _component("sulfamethoxazole", "CHEMBL443"),
                _component("trimethoprim", "CHEMBL22"),
            ],
        )
    )
    row = frame.row(0, named=True)
    assert row["drugs_chembl"] == "CHEMBL443|CHEMBL22"
    assert row["drugs"] == "sulfamethoxazole|trimethoprim"
    assert row["is_class_row"] is False


def test_class_becomes_atc_entity():
    row = _frame(("1", 0, "opioids", None, ["N02A"], [])).row(0, named=True)
    assert row["drugs_chembl"] == "ATC:N02A"
    assert row["drugs"] == "opioids"
    assert row["is_class_row"] is True


def test_multi_code_class_is_one_entity():
    row = _frame(("1", 0, "corticosteroids", None, ["H02A", "A01AC"], [])).row(
        0, named=True
    )
    assert row["drugs_chembl"] == "ATC:A01AC,H02A"


def test_mixed_row_keeps_order():
    row = _frame(
        ("1", 0, "morphine", "CHEMBL70", [], []),
        ("1", 1, "opioids", None, ["N02A"], []),
    ).row(0, named=True)
    assert row["drugs_chembl"] == "CHEMBL70|ATC:N02A"
    assert row["drugs"] == "morphine|opioids"
    assert row["is_class_row"] is True


def test_chembl_id_wins_over_codes_and_unresolved_components_drop(caplog):
    frame = _frame(
        ("1", 0, "lithium", "CHEMBL1201333", ["N05AN"], []),
        ("2", 0, "x / y", None, [], [_component("x", None), _component("y", None)]),
        ("3", 0, "radiotherapy", None, [], []),
    )
    assert frame["drugs_chembl"].to_list() == ["CHEMBL1201333"]
    assert "2 of 3 dropped" in caplog.text
