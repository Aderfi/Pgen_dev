"""Class rows (drug classes pooled over ATC members) train only, without leaking."""

from __future__ import annotations

import polars as pl
import pytest

from src.data.genotype import HgvsCatalog
from src.evaluation.class_rows import class_row_mask, leaking_class_rows
from src.evaluation.entities import ROW_ID, EntityResolver, Scenario, add_row_id
from src.evaluation.splits import fold_column, make_split
from src.model.engine.base import stratified_split

MEMBERS = {"N02A": frozenset({"CHEMBL70"})}


def _row(drugs, gene, genotype, pmid, *, is_class=False, label="Efficacy"):
    return {
        "drugs_chembl": drugs,
        "drugs": drugs.lower(),
        "gene": gene,
        "genotype": genotype,
        "alleles": None,
        "pmid": pmid,
        "phenotype_category": label,
        "is_class_row": is_class,
    }


@pytest.fixture
def frame() -> pl.DataFrame:
    rows = [
        _row(
            f"CHEMBL{i % 12}",
            f"G{i % 7}",
            f"rs{i % 9}",
            str(i % 15),
            label="Efficacy" if i % 3 else "Toxicity",
        )
        for i in range(60)
    ]
    rows += [
        _row("CHEMBL70", "G0", "rs200", "98"),  # a member drug of N02A
        _row("CHEMBL0|CHEMBL1", "G1", "rs1", "97"),  # multi-drug, not a class
        # class rows
        _row("ATC:N02A", "G2", "rs300", "90", is_class=True),  # unrelated
        _row("ATC:N02A", "G3", "rs301", "3", is_class=True),  # shares pmid 3
        _row("ATC:N02A", "G0", "rs200", "91", is_class=True),  # variant of CHEMBL70
        _row("CHEMBL5|ATC:N02A", "G4", "rs302", "92", is_class=True),  # mixed
    ]
    frame = pl.DataFrame(rows)
    return add_row_id(frame.with_columns(pl.col("pmid").cast(pl.String)))


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver(HgvsCatalog(), drug_col="drugs_chembl")


def _split(frame, resolver, scenario):
    return make_split(
        frame,
        scenario,
        resolver,
        k=5,
        seed=1,
        data_sha256="d",
        catalog_sha256="c",
        members=MEMBERS,
    )


def _parts(assignment, frame, row: int) -> list[str]:
    rid = frame[ROW_ID][row]
    picked = assignment.filter(pl.col(ROW_ID) == rid)
    return [picked[fold_column(f)][0] for f in range(5)]


def _holdout(assignment, frame, fold: int) -> pl.DataFrame:
    col = fold_column(fold)
    ids = assignment.filter(pl.col(col).is_in(["val", "test"]))[ROW_ID]
    return frame.filter(pl.col(ROW_ID).is_in(ids.implode()))


@pytest.mark.parametrize("scenario", list(Scenario))
def test_class_rows_never_in_val_or_test(frame, resolver, scenario):
    assignment, manifest = _split(frame, resolver, scenario)
    for row in range(62, 66):
        assert set(_parts(assignment, frame, row)) <= {"train", "drop"}
    assert manifest.class_rows == 4
    assert len(manifest.class_rows_dropped_per_fold) == 5


def test_class_row_mask(frame):
    assert class_row_mask(frame).sum() == 4
    plain = frame.filter(~pl.col("is_class_row")).drop("is_class_row")
    assert class_row_mask(plain).sum() == 0


def test_class_row_with_unrelated_drug_kept(frame, resolver):
    assignment, _ = _split(frame, resolver, Scenario.DRUG_COLD)
    assert _parts(assignment, frame, 62) == ["train"] * 5


def test_class_row_sharing_pmid_is_dropped(frame, resolver):
    assignment, _ = _split(frame, resolver, Scenario.DRUG_COLD)
    parts = _parts(assignment, frame, 63)
    for f in range(5):
        shared = "3" in _holdout(assignment, frame, f)["pmid"].to_list()
        assert parts[f] == ("drop" if shared else "train")
    assert "drop" in parts


def test_class_row_sharing_variant_with_member_drug_is_dropped(frame, resolver):
    assignment, _ = _split(frame, resolver, Scenario.DRUG_COLD)
    member_parts = _parts(assignment, frame, 60)  # the CHEMBL70 + rs200 row
    parts = _parts(assignment, frame, 64)
    for f in range(5):
        held = member_parts[f] in ("val", "test")
        assert parts[f] == ("drop" if held else "train")
    assert "drop" in parts and "train" in parts


def test_mixed_multi_drug_row_is_dropped_in_drug_cold(frame, resolver):
    assignment, _ = _split(frame, resolver, Scenario.DRUG_COLD)
    # a multi-drug row is held out (test) only with all its drugs, else dropped
    multi = _parts(assignment, frame, 61)
    for f in range(5):
        drugs = set(_holdout(assignment, frame, f)["drugs_chembl"].to_list())
        if multi[f] == "test":
            continue
        if {"CHEMBL0", "CHEMBL1"} & drugs:
            assert multi[f] == "drop"
    # a class row naming a held-out drug directly is dropped in that fold
    mixed = _parts(assignment, frame, 65)
    for f in range(5):
        held = "CHEMBL5" in _holdout(assignment, frame, f)["drugs_chembl"].to_list()
        assert mixed[f] == ("drop" if held else "train")


def test_leaking_class_rows_against_a_holdout(frame, resolver):
    train = frame.filter(pl.col("is_class_row"))
    holdout = frame.filter(pl.col("drugs_chembl") == "CHEMBL70")  # rs200, pmid 98
    leak = leaking_class_rows(train, holdout, resolver, MEMBERS)
    assert leak.to_list() == [False, False, True, False]
    assert not leaking_class_rows(train, holdout, resolver, {}).any()


def test_plain_split_keeps_class_rows_in_train(frame, resolver):
    train, val = stratified_split(
        frame.drop(ROW_ID), 0.25, seed=0, members=MEMBERS, resolver=resolver
    )
    assert not val["is_class_row"].any()
    kept = train.filter(pl.col("is_class_row"))
    assert 1 <= kept.height <= 4
    # whatever was kept does not leak against val
    assert not leaking_class_rows(kept, val, resolver, MEMBERS).any()


def test_drug_cold_removes_heldout_members_from_train_pools(pharm_library):
    from src.data.cache import GraphCache
    from src.model.engine.base import fold_drug_caches

    store, keys = pharm_library
    cache = GraphCache(
        keys, store=store, class_members={"X": frozenset({"CHEMBL25", "CHEMBL545"})}
    )
    held = pl.DataFrame({"drugs_chembl": ["CHEMBL25|CHEMBL671"]})
    train_cache, val_cache = fold_drug_caches(
        cache, Scenario.DRUG_COLD, [held], drug_col="drugs_chembl"
    )
    assert train_cache.entity_members("ATC:X") == ["CHEMBL545"]
    assert val_cache.entity_members("ATC:X") == ["CHEMBL25", "CHEMBL545"]
    same, _ = fold_drug_caches(
        cache, Scenario.GENE_COLD, [held], drug_col="drugs_chembl"
    )
    assert same.entity_members("ATC:X") == ["CHEMBL25", "CHEMBL545"]


def test_class_row_mask_from_keys_without_column(frame):
    # final review #4: a source without is_class_row still has its ATC keys
    no_column = frame.drop("is_class_row")
    assert class_row_mask(no_column).sum() == 4


def test_validation_reads_the_lowest_energy_conformer(pharm_library):
    # final review #9: val (early stopping, thresholds, temperature) is
    # deterministic; training keeps sampling
    from src.data.cache import GraphCache
    from src.model.engine.base import fold_drug_caches

    store, keys = pharm_library
    cache = GraphCache(keys, store=store)
    held = pl.DataFrame({"drugs_chembl": ["CHEMBL25"]})
    for scenario in (Scenario.DRUG_COLD, Scenario.GENE_COLD, None):
        train_cache, val_cache = fold_drug_caches(
            cache, scenario, [held], drug_col="drugs_chembl"
        )
        assert train_cache.conformer is None
        assert val_cache.conformer == 0
