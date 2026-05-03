"""Column-level normalization helpers.

Two small utilities extracted from the legacy ``DataLoaderUtils``:

- ``MultiLabelNormalizer`` — Polars expression builder that canonicalizes a
  pipe-delimited multi-label column (split → strip → unique → sort → join).
- ``Stratifier`` — adds a synthetic ``_stratify`` column derived from one or
  more existing columns, suitable for ``sklearn.model_selection.train_test_split``.

Both are stateless. They expose static methods rather than instance methods to
discourage callers from leaking state across calls.
"""

from __future__ import annotations

import polars as pl


class MultiLabelNormalizer:
    """Build Polars expressions that canonicalize multi-label string columns.

    The default delimiter is ``|`` to match the rest of the project (PharmGKB
    multi-value columns). The expression is referentially transparent and can
    be safely composed inside ``with_columns`` for parallel execution.
    """

    DEFAULT_DELIMITER: str = "|"

    @staticmethod
    def normalize_expr(col_name: str, delimiter: str = DEFAULT_DELIMITER) -> pl.Expr:
        """Return an expression that produces a canonical pipe-delimited string."""
        return (
            pl.col(col_name)
            .cast(pl.String)
            .fill_null("")
            .str.split(delimiter)
            .list.eval(pl.element().str.strip_chars())
            .list.eval(pl.element().filter(pl.element() != ""))
            .list.unique()
            .list.sort()
            .list.join(delimiter)
        )


class Stratifier:
    """Compose a stratification column from one or more existing columns."""

    OUTPUT_COLUMN: str = "_stratify"

    @staticmethod
    def add_stratify_column(
        df: pl.DataFrame, stratify_cols: list[str]
    ) -> pl.DataFrame:
        """Append ``_stratify = col1_col2_...`` for stratified train/test splits.

        - Returns the DataFrame unchanged if no columns are requested.
        - Silently drops columns that don't exist (callers may pass optional
          stratification keys; absence is not an error).
        """
        if not stratify_cols:
            return df
        valid = [c for c in stratify_cols if c in df.columns]
        if not valid:
            return df
        return df.with_columns(
            _stratify=pl.concat_str(valid, separator="_", ignore_nulls=True).alias(
                Stratifier.OUTPUT_COLUMN
            )
        )
