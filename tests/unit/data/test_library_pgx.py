"""Tests for src.data.library.pgx."""


from src.data.library.pgx import parse_haplotype_label


class TestParseHaplotypeLabel:
    def test_gene_underscore_n_becomes_star(self) -> None:
        assert parse_haplotype_label("CYP2D6", "CYP2D6_4") == "*4"
        assert parse_haplotype_label("CYP2D6", "CYP2D6_10") == "*10"

    def test_bare_n_becomes_star(self) -> None:
        assert parse_haplotype_label("CYP2D6", "4") == "*4"

    def test_already_starred(self) -> None:
        assert parse_haplotype_label("CYP2D6", "*4") == "*4"

    def test_letter_suffix(self) -> None:
        assert parse_haplotype_label("CYP2D6", "CYP2D6_2A") == "*2A"

    def test_dpyd_rsid_resolved_via_catalog(self) -> None:
        # rs3918290 → DPYD*2A in the catalog at data/dicts/star_alleles.tsv
        assert parse_haplotype_label("DPYD", "rs3918290") == "*2A"

    def test_unknown_rsid_passes_through(self) -> None:
        assert parse_haplotype_label("CYP2D6", "rs999999999") == "rs999999999"

    def test_hgvs_kept(self) -> None:
        # HGVS-style names start with 'c.' which doesn't match any digit/star rule.
        # We accept the resulting form even if not strictly canonical — the
        # important thing is that it's stable and round-trippable.
        result = parse_haplotype_label("DPYD", "c.2846A>T")
        assert result.startswith("*") or "c." in result
