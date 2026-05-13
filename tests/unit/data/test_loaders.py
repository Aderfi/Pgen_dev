"""Tests for src.data.loaders."""

from pathlib import Path

import pytest

from src.data.loaders import DEFAULT_NULL_VALUES, TRAIN_DATA_SCHEMA, TabularLoader


@pytest.fixture
def tsv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.tsv"
    p.write_text(
        "drugs_cid\tdrugs\tgene\tgenotype\n"
        "2244\taspirin\tCYP2D6\trs3892097\n"
        "1234\tcaffeine\tCYP1A2\trs762551\n"
    )
    return p


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.csv"
    p.write_text("a,b\n1,foo\n2,bar\n")
    return p


class TestTabularLoader:
    def test_loads_tsv_with_default_separator(self, tsv_file: Path) -> None:
        df = TabularLoader.load(tsv_file)
        assert df.shape == (2, 4)
        assert set(df.columns) == {"drugs_cid", "drugs", "gene", "genotype"}
        assert df["drugs_cid"].to_list() == ["2244", "1234"]

    def test_csv_uses_comma(self, csv_file: Path) -> None:
        df = TabularLoader.load(csv_file, schema=None)
        assert df.shape == (2, 2)
        assert df["b"].to_list() == ["foo", "bar"]

    def test_explicit_separator_overrides(self, tmp_path: Path) -> None:
        # Pipe-separated file with .csv extension — only an explicit
        # separator should parse it correctly.
        p = tmp_path / "weird.csv"
        p.write_text("a|b\n1|foo\n2|bar\n")
        df = TabularLoader.load(p, schema=None, separator="|")
        assert df["b"].to_list() == ["foo", "bar"]

    def test_columns_projection(self, tsv_file: Path) -> None:
        df = TabularLoader.load(tsv_file, columns=["drugs_cid", "gene"])
        assert df.columns == ["drugs_cid", "gene"]

    def test_nulls_unified(self, tmp_path: Path) -> None:
        p = tmp_path / "nulls.tsv"
        p.write_text(
            "a\tb\n"
            "x\tNA\n"
            "y\t\n"  # empty cell
            "z\tnull\n"
        )
        df = TabularLoader.load(p, schema=None)
        # All three rows should have null in 'b'.
        assert df["b"].null_count() == 3

    def test_default_null_values_complete(self) -> None:
        for token in ("", "NA", "NaN", "null", "N/A"):
            assert token in DEFAULT_NULL_VALUES

    def test_train_data_schema_is_dict(self) -> None:
        # If someone accidentally turns it into a list, schema= passes will fail.
        assert isinstance(TRAIN_DATA_SCHEMA, dict)
        assert "drugs_cid" in TRAIN_DATA_SCHEMA
