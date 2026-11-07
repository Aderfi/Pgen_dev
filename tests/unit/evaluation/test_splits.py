"""Grouped k-fold assignment, persistence and hash verification."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.core import EvaluationError
from src.data.genotype import HgvsCatalog
from src.evaluation import splits as splits_module
from src.evaluation.entities import ROW_ID, EntityResolver, Scenario, add_row_id
from src.evaluation.splits import (
    assign_folds,
    fold_frames,
    load_split,
    make_split,
    save_split,
    verify_split,
)


def _single_keys(n_groups: int, rows_per_group: int) -> list[frozenset[str]]:
    return [
        frozenset({f"g{g}"}) for g in range(n_groups) for _ in range(rows_per_group)
    ]


def test_each_group_tested_once() -> None:
    keys = _single_keys(50, 4)
    a = assign_folds(keys, k=5, seed=1, val_fraction=0.15)
    assert a.shape == (200, 5)
    assert ((a == "test").sum(axis=1) == 1).all()
    for f in range(5):
        test_keys = {k for k, v in zip(keys, a[:, f], strict=True) if v == "test"}
        other = {k for k, v in zip(keys, a[:, f], strict=True) if v in ("train", "val")}
        assert not test_keys & other


def test_mixed_row_dropped_only_where_mixed() -> None:
    keys = [*_single_keys(40, 3), frozenset({"g0", "g1"})]
    a = assign_folds(keys, k=5, seed=3, val_fraction=0.15)
    f0 = int(np.flatnonzero(a[0] == "test")[0])
    f1 = int(np.flatnonzero(a[3] == "test")[0])
    mixed = a[-1]
    if f0 == f1:
        assert mixed[f0] == "test"
    else:
        assert mixed[f0] == "drop"
        assert mixed[f1] == "drop"
        assert "test" not in mixed


def test_val_fraction_approx() -> None:
    a = assign_folds(_single_keys(100, 3), k=5, seed=0, val_fraction=0.15)
    for f in range(5):
        non_test = (a[:, f] != "test").sum()
        val = (a[:, f] == "val").sum()
        assert abs(val / non_test - 0.15) < 0.05


def test_random_stratified_folds() -> None:
    labels = ["A"] * 700 + ["B"] * 300
    keys = [frozenset({str(i)}) for i in range(1000)]
    a = assign_folds(keys, k=5, seed=0, val_fraction=0.15, stratify=labels)
    lab = np.array(labels)
    for f in range(5):
        test = lab[a[:, f] == "test"]
        assert abs((test == "B").mean() - 0.3) < 0.1
    assert ((a == "test").sum(axis=1) == 1).all()


def test_deterministic() -> None:
    keys = _single_keys(30, 2)
    a = assign_folds(keys, k=5, seed=7, val_fraction=0.15)
    assert (a == assign_folds(keys, k=5, seed=7, val_fraction=0.15)).all()
    assert not (a == assign_folds(keys, k=5, seed=8, val_fraction=0.15)).all()


@pytest.fixture
def frame() -> pl.DataFrame:
    n = 60
    return add_row_id(
        pl.DataFrame(
            {
                "variant_annotation_id": list(range(n)),
                "drugs_chembl": [str(i % 12) for i in range(n)],
                "drugs": [f"d{i % 12}" for i in range(n)],
                "gene": [f"G{i % 7}" for i in range(n)],
                "genotype": [f"rs{i % 9}" for i in range(n)],
                "alleles": [None] * n,
                "pmid": [str(i % 15) for i in range(n)],
                "phenotype_category": [
                    "Efficacy" if i % 3 else "Toxicity" for i in range(n)
                ],
            }
        )
    )


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver(HgvsCatalog(), drug_col="drugs_chembl")


def test_save_load_roundtrip(tmp_path, monkeypatch, frame, resolver) -> None:
    monkeypatch.setattr(splits_module, "split_dir", lambda s: tmp_path / str(s))
    assignment, manifest = make_split(
        frame,
        Scenario.DRUG_COLD,
        resolver,
        k=3,
        seed=1,
        data_sha256="d",
        catalog_sha256="c",
    )
    save_split(assignment, manifest)
    loaded, loaded_manifest = load_split(Scenario.DRUG_COLD)
    assert loaded.equals(assignment)
    assert loaded_manifest.split_sha256 != ""
    assert loaded_manifest.model_dump(exclude={"split_sha256"}) == manifest.model_dump(
        exclude={"split_sha256"}
    )
    assert splits_module.file_sha256(tmp_path / "drug_cold" / "assignment.parquet") == (
        loaded_manifest.split_sha256
    )


def test_manifest_counts(frame, resolver) -> None:
    assignment, manifest = make_split(
        frame,
        Scenario.GENE_COLD,
        resolver,
        k=3,
        seed=1,
        data_sha256="d",
        catalog_sha256="c",
    )
    assert manifest.n_rows == 60
    assert manifest.k == 3
    for f, counts in enumerate(manifest.rows_per_fold):
        assert sum(counts.values()) == 60
        assert counts["test"] == (assignment[f"fold_{f}"] == "test").sum()


def test_verify_split_hash_mismatch(frame, resolver) -> None:
    _, manifest = make_split(
        frame,
        Scenario.PMID,
        resolver,
        k=3,
        seed=1,
        data_sha256="aaa",
        catalog_sha256="c",
    )
    verify_split(manifest, data_sha256="aaa", catalog_sha256="c")
    with pytest.raises(EvaluationError, match="data") as exc:
        verify_split(manifest, data_sha256="bbb", catalog_sha256="c")
    assert "aaa" in str(exc.value)
    assert "bbb" in str(exc.value)
    with pytest.raises(EvaluationError, match="catalog"):
        verify_split(manifest, data_sha256="aaa", catalog_sha256="zzz")


def test_fold_frames_partition(frame, resolver) -> None:
    assignment, _ = make_split(
        frame,
        Scenario.DRUG_COLD,
        resolver,
        k=3,
        seed=1,
        data_sha256="d",
        catalog_sha256="c",
    )
    train, val, test = fold_frames(frame, assignment, 0)
    ids = [set(x[ROW_ID].to_list()) for x in (train, val, test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    kept = set(assignment.filter(pl.col("fold_0") != "drop")[ROW_ID].to_list())
    assert ids[0] | ids[1] | ids[2] == kept
    assert not set(train["drugs_chembl"]) & set(test["drugs_chembl"])
