"""Tests for src.data.cleaning."""

import polars as pl

from src.data.cleaning import GenoKeyBuilder, PharmacogenomicCleaner

# Toy rsID table: forces deterministic outputs without depending on the
# real catalog (which can drift as data/dicts/star_alleles.tsv evolves).
TOY_RSID_TO_LABELS = {
    "rs3892097": ["CYP2D6*4"],
    "rs4244285": ["CYP2C19*2"],
}


class TestGenoKeyBuilder:
    def test_star_allele_in_alleles_column(self) -> None:
        gkb = GenoKeyBuilder(TOY_RSID_TO_LABELS)
        keys = gkb.keys_for(gene="CYP2D6", genotype="", alleles="*4/*1")
        assert "CYP2D6_*4" in keys

    def test_rsid_resolved_via_lookup(self) -> None:
        gkb = GenoKeyBuilder(TOY_RSID_TO_LABELS)
        keys = gkb.keys_for(gene="CYP2D6", genotype="rs3892097", alleles="")
        assert keys == ["CYP2D6_*4"]

    def test_unknown_rsid_falls_back(self) -> None:
        gkb = GenoKeyBuilder(TOY_RSID_TO_LABELS)
        keys = gkb.keys_for(gene="CYP2D6", genotype="rs9999999", alleles="")
        assert keys == ["CYP2D6_rs9999999"]

    def test_multiple_rsids_pipe_separated(self) -> None:
        gkb = GenoKeyBuilder(TOY_RSID_TO_LABELS)
        keys = gkb.keys_for(
            gene="CYP2C19", genotype="rs4244285|rs3892097", alleles=""
        )
        assert "CYP2C19_*4" in keys
        # rs4244285 → CYP2C19*2 (matches gene)
        assert "CYP2C19_*2" in keys

    def test_default_uses_real_catalog(self) -> None:
        # If the catalog ever fails to load, this constructor would raise.
        gkb = GenoKeyBuilder()
        # rs3892097 → CYP2D6*4 in the real catalog.
        keys = gkb.keys_for(gene="CYP2D6", genotype="rs3892097", alleles="")
        assert "CYP2D6_*4" in keys


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
        cleaner = PharmacogenomicCleaner(
            key_builder=GenoKeyBuilder(TOY_RSID_TO_LABELS)
        )
        out = cleaner.clean(self._df())
        # Rows with null/empty gene or empty genotype are dropped.
        assert len(out) <= 2

    def test_geno_key_present(self) -> None:
        cleaner = PharmacogenomicCleaner(
            key_builder=GenoKeyBuilder(TOY_RSID_TO_LABELS)
        )
        out = cleaner.clean(self._df())
        assert "geno_key" in out.columns
        keys = set(out["geno_key"].to_list())
        # The two valid input rows yield CYP2D6_*4 and CYP2C19_*2.
        assert "CYP2D6_*4" in keys
        assert "CYP2C19_*2" in keys

    def test_stratify_column_added(self) -> None:
        cleaner = PharmacogenomicCleaner(
            key_builder=GenoKeyBuilder(TOY_RSID_TO_LABELS)
        )
        out = cleaner.clean(self._df(), stratify_col="gene")
        assert "_stratify" in out.columns

    def test_alleles_column_optional(self) -> None:
        df = pl.DataFrame(
            {"gene": ["CYP2D6"], "genotype": ["rs3892097"]}
        )
        cleaner = PharmacogenomicCleaner(
            key_builder=GenoKeyBuilder(TOY_RSID_TO_LABELS)
        )
        out = cleaner.clean(df)
        assert "geno_key" in out.columns
