"""Novelty flags, Tanimoto similarity and train/test overlap."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.data.genotype import HgvsCatalog
from src.evaluation.diagnostics import max_tanimoto, novelty_flags, overlap_report
from src.evaluation.entities import EntityResolver, Scenario, add_row_id


@pytest.fixture
def resolver() -> EntityResolver:
    return EntityResolver(HgvsCatalog(), drug_col="drugs_chembl")


def _frame(rows: list[tuple[str, str, str, str]], start: int = 0) -> pl.DataFrame:
    return add_row_id(
        pl.DataFrame(
            {
                "variant_annotation_id": list(range(start, start + len(rows))),
                "drugs_chembl": [r[0] for r in rows],
                "drugs": ["x"] * len(rows),
                "genotype": [r[1] for r in rows],
                "gene": [r[2] for r in rows],
                "pmid": [r[3] for r in rows],
                "alleles": [None] * len(rows),
            }
        )
    )


def test_flags(resolver) -> None:
    train = _frame([("1", "rs1", "G1", "10")])
    test = _frame([("1", "rs2", "G1", "11"), ("2", "rs1", "G2", "10")], start=5)
    flags = novelty_flags(resolver, train, test)
    first, second = flags.row(0, named=True), flags.row(1, named=True)
    assert first["seen_drug"] and not first["seen_variant"] and not first["seen_pair"]
    assert first["seen_gene"] and not first["seen_pmid"]
    assert not second["seen_drug"] and second["seen_variant"] and second["seen_pmid"]
    assert flags["row_id"].to_list() == test["row_id"].to_list()


def test_tanimoto_identical_is_one_and_missing_is_none() -> None:
    fps = {
        "a": np.array([1, 1, 0, 0], dtype=bool),
        "b": np.array([1, 0, 0, 0], dtype=bool),
        "c": np.array([1, 1, 0, 0], dtype=bool),
    }
    out = max_tanimoto(
        [frozenset({"c"}), frozenset({"b"}), frozenset({"zz"})], {"a"}, fps
    )
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] is None


def test_overlap_violation_detected(resolver) -> None:
    train = _frame([("1", "rs1", "G1", "10"), ("2", "rs2", "G2", "11")])
    test = _frame([("1", "rs3", "G3", "12")], start=9)
    report = overlap_report(resolver, train, test, Scenario.DRUG_COLD)
    assert report.shared_drugs == 1
    assert "shared_drugs" in report.violations
    assert overlap_report(resolver, train, test, Scenario.RANDOM).violations == []


def test_exact_duplicates_counted(resolver) -> None:
    train = _frame([("1", "rs1", "G1", "10")])
    test = _frame([("1", "rs1", "G1", "99")], start=3)
    report = overlap_report(resolver, train, test, Scenario.PMID)
    assert report.exact_duplicates == 1
    assert report.exact_duplicates_pct == pytest.approx(100.0)
    assert report.violations == []


def test_tanimoto_bins() -> None:
    from src.evaluation.diagnostics import tanimoto_bin

    assert tanimoto_bin(None) == "no structure"
    assert tanimoto_bin(0.1) == "[0.0, 0.3)"
    assert tanimoto_bin(0.95) == "[0.9, 1.0]"
    assert tanimoto_bin(1.0) == "[0.9, 1.0]"
