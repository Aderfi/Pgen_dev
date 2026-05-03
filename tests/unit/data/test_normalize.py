"""Tests for src.data.normalize."""

import polars as pl
import pytest

from src.data.normalize import MultiLabelNormalizer, Stratifier


class TestMultiLabelNormalizer:
    def test_dedup_and_sort(self) -> None:
        df = pl.DataFrame({"tags": ["c|a|b|a", "b|a"]})
        out = df.with_columns(
            MultiLabelNormalizer.normalize_expr("tags").alias("out")
        )["out"].to_list()
        assert out == ["a|b|c", "a|b"]

    def test_strips_whitespace(self) -> None:
        df = pl.DataFrame({"tags": ["  a  | b "]})
        out = df.with_columns(
            MultiLabelNormalizer.normalize_expr("tags").alias("out")
        )["out"].to_list()
        assert out == ["a|b"]

    def test_drops_empty_segments(self) -> None:
        df = pl.DataFrame({"tags": ["a||b||"]})
        out = df.with_columns(
            MultiLabelNormalizer.normalize_expr("tags").alias("out")
        )["out"].to_list()
        assert out == ["a|b"]

    def test_handles_nulls(self) -> None:
        df = pl.DataFrame({"tags": ["a|b", None, ""]})
        out = df.with_columns(
            MultiLabelNormalizer.normalize_expr("tags").alias("out")
        )["out"].to_list()
        assert out == ["a|b", "", ""]

    def test_custom_delimiter(self) -> None:
        df = pl.DataFrame({"tags": ["a,b,a"]})
        out = df.with_columns(
            MultiLabelNormalizer.normalize_expr("tags", delimiter=",").alias("out")
        )["out"].to_list()
        assert out == ["a,b"]


class TestStratifier:
    def test_concatenates_columns(self) -> None:
        df = pl.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})
        out = Stratifier.add_stratify_column(df, ["a", "b"])
        assert out["_stratify"].to_list() == ["x_1", "y_2"]

    def test_no_columns_is_passthrough(self) -> None:
        df = pl.DataFrame({"a": ["x", "y"]})
        out = Stratifier.add_stratify_column(df, [])
        assert "_stratify" not in out.columns

    def test_missing_columns_skipped(self) -> None:
        df = pl.DataFrame({"a": ["x", "y"]})
        out = Stratifier.add_stratify_column(df, ["a", "missing"])
        # Only 'a' survived the filter — output has just one column's value.
        assert out["_stratify"].to_list() == ["x", "y"]

    def test_all_columns_missing_passthrough(self) -> None:
        df = pl.DataFrame({"a": ["x"]})
        out = Stratifier.add_stratify_column(df, ["nope1", "nope2"])
        assert "_stratify" not in out.columns

    def test_constant_name(self) -> None:
        # Name is part of the contract — pipeline uses it for sklearn split.
        assert Stratifier.OUTPUT_COLUMN == "_stratify"
