"""Tests for src.data.library.admet — the predicted ADMET/CYP profile.

These cover the pure-Python surface (endpoint layout, record parsing, column
projection, the provider lookup, and the Parquet cache logic) without importing
``admet_ai`` — the heavy D-MPNN ensemble is exercised only at build time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from src.data.library.admet import (
    ADMET_ENDPOINTS,
    DRUG_ADMET_DIM,
    AdmetProvider,
    _select_columns,
    load_or_build_admet_table,
    records_from_rows,
    zero_admet_vector,
)

if TYPE_CHECKING:
    from pathlib import Path

_CID_COL = "cid"
_PERCENTILE_SUFFIX = "_drugbank_approved_percentile"
# Regression endpoints sourced from the DrugBank percentile rather than a raw value.
_REGRESSION = {
    "Solubility_AqSolDB",
    "Lipophilicity_AstraZeneca",
    "HydrationFreeEnergy_FreeSolv",
    "Caco2_Wang",
    "PPBR_AZ",
    "VDss_Lombardo",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Half_Life_Obach",
    "LD50_Zhu",
}


def _admet_table(rows: dict[int, float]) -> pl.DataFrame:
    """Build a profile table (cid + all endpoints) with each row filled by a constant."""
    cids = list(rows)
    data: dict[str, list] = {_CID_COL: cids}
    for endpoint in ADMET_ENDPOINTS:
        data[endpoint] = [rows[c] for c in cids]
    return pl.DataFrame(data)


class TestLayout:
    def test_dim_matches_endpoint_count(self) -> None:
        assert DRUG_ADMET_DIM == 41
        assert len(ADMET_ENDPOINTS) == DRUG_ADMET_DIM

    def test_endpoints_unique(self) -> None:
        assert len(set(ADMET_ENDPOINTS)) == len(ADMET_ENDPOINTS)


class TestRecordsFromRows:
    def test_keeps_valid_pairs(self) -> None:
        rows = [{"cid": "123", "smiles": "CCO"}, {"cid": 7, "smiles": "c1ccccc1"}]
        assert records_from_rows(rows) == [(123, "CCO"), (7, "c1ccccc1")]

    def test_drops_non_integer_cid(self) -> None:
        assert records_from_rows([{"cid": "abc", "smiles": "CCO"}]) == []
        assert records_from_rows([{"cid": None, "smiles": "CCO"}]) == []

    def test_drops_empty_smiles(self) -> None:
        assert records_from_rows([{"cid": "1", "smiles": ""}]) == []
        assert records_from_rows([{"cid": "1", "smiles": None}]) == []
        assert records_from_rows([{"cid": "1"}]) == []


class TestSelectColumns:
    def test_regression_uses_percentile_over_100(self) -> None:
        raw = pl.DataFrame(
            {e: [1.0] for e in ADMET_ENDPOINTS if e not in _REGRESSION}
            | {f"{e}{_PERCENTILE_SUFFIX}": [50.0] for e in _REGRESSION}
        )
        out = _select_columns(raw)
        assert out.columns == list(ADMET_ENDPOINTS)
        for endpoint in _REGRESSION:
            assert out[endpoint][0] == pytest.approx(0.5)

    def test_classification_passes_through(self) -> None:
        raw = pl.DataFrame(
            {e: [0.3] for e in ADMET_ENDPOINTS if e not in _REGRESSION}
            | {f"{e}{_PERCENTILE_SUFFIX}": [10.0] for e in _REGRESSION}
        )
        out = _select_columns(raw)
        classification = next(e for e in ADMET_ENDPOINTS if e not in _REGRESSION)
        assert out[classification][0] == pytest.approx(0.3)

    def test_missing_expected_column_raises(self) -> None:
        raw = pl.DataFrame({e: [0.0] for e in ADMET_ENDPOINTS[:-1]})
        with pytest.raises(KeyError):
            _select_columns(raw)


class TestAdmetProvider:
    def test_hit_returns_row_shape(self) -> None:
        provider = AdmetProvider(_admet_table({42: 0.7}))
        vec = provider.vector_for(42)
        assert vec.shape == (1, DRUG_ADMET_DIM)
        assert vec[0, 0].item() == pytest.approx(0.7)
        assert provider.misses == 0
        assert 42 in provider

    def test_miss_returns_zeros_and_counts(self) -> None:
        provider = AdmetProvider(_admet_table({42: 0.7}))
        vec = provider.vector_for(999)
        assert vec.shape == (1, DRUG_ADMET_DIM)
        assert vec.abs().sum().item() == 0.0
        assert provider.misses == 1
        assert 999 not in provider

    def test_null_provider_is_all_zeros(self) -> None:
        provider = AdmetProvider.null()
        assert provider.vector_for(1).abs().sum().item() == 0.0
        assert provider.misses == 1


def test_zero_admet_vector_shape() -> None:
    vec = zero_admet_vector()
    assert vec.shape == (1, DRUG_ADMET_DIM)
    assert vec.abs().sum().item() == 0.0


class TestCache:
    def test_reuses_cache_on_schema_match(self, tmp_path: Path) -> None:
        cache = tmp_path / "admet.parquet"
        _admet_table({1: 0.1, 2: 0.2}).write_parquet(cache)

        # A non-empty cache with the right schema must be returned untouched, so
        # the (mocked-out) compute path is never reached.
        def _fail(*_a: object, **_k: object) -> object:
            raise AssertionError("compute_admet_table should not run on cache hit")

        import src.data.library.admet as admet_mod

        original = admet_mod.compute_admet_table
        admet_mod.compute_admet_table = _fail  # type: ignore[assignment]
        try:
            table = load_or_build_admet_table([], cache)
        finally:
            admet_mod.compute_admet_table = original  # type: ignore[assignment]
        assert table.height == 2
        assert set(table.columns) == {_CID_COL, *ADMET_ENDPOINTS}

    def test_recomputes_on_schema_mismatch(self, tmp_path: Path) -> None:
        cache = tmp_path / "admet.parquet"
        pl.DataFrame({_CID_COL: [1], "stale_col": [0.0]}).write_parquet(cache)

        import src.data.library.admet as admet_mod

        sentinel = _admet_table({5: 0.5})
        original = admet_mod.compute_admet_table
        admet_mod.compute_admet_table = lambda *a, **k: sentinel  # type: ignore[assignment]
        try:
            table = load_or_build_admet_table([], cache)
        finally:
            admet_mod.compute_admet_table = original  # type: ignore[assignment]
        assert set(table.columns) == {_CID_COL, *ADMET_ENDPOINTS}
        # The recomputed table is also written back to the cache.
        assert set(pl.read_parquet(cache).columns) == {_CID_COL, *ADMET_ENDPOINTS}
