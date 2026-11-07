"""Tests for src.data.library.admet — the predicted ADMET/CYP profile.

These cover the pure-Python surface (endpoint layout, record parsing, column
projection, the provider lookup, and the Parquet cache logic) without importing
``admet_ai`` — the heavy D-MPNN ensemble is exercised only at build time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from src.data.library import admet as admet_mod
from src.data.library.admet import (
    ADMET_COLUMNS,
    ADMET_ENDPOINTS,
    DRUG_ADMET_DIM,
    AdmetProvider,
    _select_columns,
    read_admet_table,
    update_admet_table,
    zero_admet_vector,
)

if TYPE_CHECKING:
    from pathlib import Path

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
ASPIRIN = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
ETHANOL = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


def _admet_table(rows: dict[str, float]) -> pl.DataFrame:
    """A profile table (inchikey + all endpoints), each row filled by a constant."""
    keys = list(rows)
    data: dict[str, list] = {"inchikey": keys}
    for endpoint in ADMET_ENDPOINTS:
        data[endpoint] = [rows[k] for k in keys]
    return pl.DataFrame(data)


class TestLayout:
    def test_dim_matches_endpoint_count(self) -> None:
        assert DRUG_ADMET_DIM == 41
        assert len(ADMET_ENDPOINTS) == DRUG_ADMET_DIM
        assert ADMET_COLUMNS == ("inchikey", *ADMET_ENDPOINTS)

    def test_endpoints_unique(self) -> None:
        assert len(set(ADMET_ENDPOINTS)) == len(ADMET_ENDPOINTS)


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
        provider = AdmetProvider(_admet_table({ASPIRIN: 0.7}))
        vec = provider.vector_for(ASPIRIN)
        assert vec.shape == (1, DRUG_ADMET_DIM)
        assert vec[0, 0].item() == pytest.approx(0.7)
        assert provider.misses == 0
        assert ASPIRIN in provider
        assert len(provider) == 1

    def test_miss_returns_zeros_and_counts(self) -> None:
        provider = AdmetProvider(_admet_table({ASPIRIN: 0.7}))
        vec = provider.vector_for(ETHANOL)
        assert vec.shape == (1, DRUG_ADMET_DIM)
        assert vec.abs().sum().item() == 0.0
        assert provider.misses == 1
        assert ETHANOL not in provider

    def test_null_provider_is_all_zeros(self) -> None:
        provider = AdmetProvider.null()
        assert provider.vector_for(ASPIRIN).abs().sum().item() == 0.0
        assert provider.misses == 1

    def test_load_without_a_table_is_null(self, tmp_path: Path) -> None:
        assert len(AdmetProvider.load(tmp_path / "missing.parquet")) == 0


def test_zero_admet_vector_shape() -> None:
    vec = zero_admet_vector()
    assert vec.shape == (1, DRUG_ADMET_DIM)
    assert vec.abs().sum().item() == 0.0


class TestUpdate:
    """``update_admet_table`` predicts only what the cache lacks (no ADMET-AI here)."""

    @pytest.fixture
    def predicted(self, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake(records):  # one constant profile per requested compound
            keys = [k for k, _ in records]
            calls.append(keys)
            return _admet_table(dict.fromkeys(keys, 0.5))

        monkeypatch.setattr(admet_mod, "compute_admet_table", fake)
        return calls

    def test_full_cache_is_reused(self, tmp_path: Path, predicted) -> None:
        cache = tmp_path / "admet.parquet"
        _admet_table({ASPIRIN: 0.1, ETHANOL: 0.2}).write_parquet(cache)
        table = update_admet_table([(ASPIRIN, "CC(=O)Oc1ccccc1C(=O)O")], cache)
        assert predicted == []
        assert table.height == 2

    def test_only_missing_compounds_are_predicted(
        self, tmp_path: Path, predicted
    ) -> None:
        cache = tmp_path / "admet.parquet"
        _admet_table({ASPIRIN: 0.1}).write_parquet(cache)
        records = [(ASPIRIN, "CC(=O)Oc1ccccc1C(=O)O"), (ETHANOL, "CCO")]
        table = update_admet_table(records, cache)
        assert predicted == [[ETHANOL]]
        assert sorted(table["inchikey"].to_list()) == sorted([ASPIRIN, ETHANOL])
        stored = read_admet_table(cache)
        assert stored is not None
        assert (
            stored.filter(pl.col("inchikey") == ASPIRIN)[ADMET_ENDPOINTS[0]][0] == 0.1
        )

    def test_force_recomputes_everything(self, tmp_path: Path, predicted) -> None:
        cache = tmp_path / "admet.parquet"
        _admet_table({ASPIRIN: 0.1}).write_parquet(cache)
        update_admet_table([(ASPIRIN, "CC(=O)Oc1ccccc1C(=O)O")], cache, force=True)
        assert predicted == [[ASPIRIN]]

    def test_cache_with_other_columns_is_recomputed(
        self, tmp_path: Path, predicted
    ) -> None:
        cache = tmp_path / "admet.parquet"
        pl.DataFrame({"cid": [1], "stale_col": [0.0]}).write_parquet(cache)
        assert read_admet_table(cache) is None
        table = update_admet_table([(ASPIRIN, "CCO")], cache)
        assert predicted == [[ASPIRIN]]
        assert tuple(table.columns) == ADMET_COLUMNS
        assert tuple(pl.read_parquet(cache).columns) == ADMET_COLUMNS


def test_unparseable_smiles_never_reach_the_model() -> None:
    table = admet_mod.compute_admet_table([(ASPIRIN, "not a smiles")])
    assert table.height == 0
    assert tuple(table.columns) == ADMET_COLUMNS
