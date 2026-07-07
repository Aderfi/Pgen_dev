"""Tests for src.data.cleaning."""

import polars as pl

from src.data.cleaning import PharmacogenomicCleaner


class TestPharmacogenomicCleaner:
    def _df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "gene": ["CYP2D6", "CYP2C19", None, "  ", "CYP2D6"],
                "genotype": ["rs3892097", "rs4244285", "rs1", "rs2", ""],
                "alleles": ["*4", "*2", "", "", ""],
            }
        )

    def test_drops_invalid_rows(self) -> None:
        out = PharmacogenomicCleaner().clean(self._df())
        # Rows with null/empty gene or empty genotype are dropped (3 of 5).
        assert len(out) == 2

    def test_keeps_gene_and_genotype(self) -> None:
        out = PharmacogenomicCleaner().clean(self._df())
        assert "gene" in out.columns
        assert "genotype" in out.columns
        # No synthetic join key is built anymore.
        assert "geno_key" not in out.columns
        assert set(out["gene"].to_list()) == {"CYP2D6", "CYP2C19"}

    def test_strips_ref_seq_prefix(self) -> None:
        df = pl.DataFrame({"gene": ["CYP2C9"], "genotype": ["REF_SEQ|rs1057910"]})
        out = PharmacogenomicCleaner().clean(df)
        assert out["genotype"].to_list() == ["rs1057910"]

    def test_stratify_column_added(self) -> None:
        out = PharmacogenomicCleaner().clean(self._df(), stratify_col="gene")
        assert "_stratify" in out.columns

    def test_multilabel_normalized(self) -> None:
        df = pl.DataFrame(
            {
                "gene": ["CYP2D6"],
                "genotype": ["rs3892097"],
                "effect_type": ["Toxicity ; Efficacy"],
            }
        )
        out = PharmacogenomicCleaner(multi_label_cols=["effect_type"]).clean(df)
        assert "effect_type" in out.columns

    def test_alleles_column_optional(self) -> None:
        df = pl.DataFrame({"gene": ["CYP2D6"], "genotype": ["rs3892097"]})
        out = PharmacogenomicCleaner().clean(df)
        assert len(out) == 1
