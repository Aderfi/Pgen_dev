"""Tests for src.data.library.genes."""

import pytest

from src.data.library.genes import GENE_EDGE_DIM, GENE_NODE_DIM, safe_variant_filename


class TestSafeVariantFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("*4", "star4"),
            ("rs1234", "rs1234"),
            ("*1xN", "star1xN"),
            ("c.2846A>T", "c.2846A>T"),  # > and < survive — no filesystem reserves them on Linux
            ("foo:bar/baz|qux", "foo_bar_baz_qux"),
        ],
    )
    def test_handles_common_variants(self, raw: str, expected: str) -> None:
        assert safe_variant_filename(raw) == expected


class TestGeneSchemaConstants:
    def test_dimensions_match_models_toml(self) -> None:
        # These two values are mirrored in src/config/data/models.toml under
        # [TwoTowerGAT].geno_node_features / geno_attrs_features. Changing
        # either silently would invalidate every trained model in
        # src/pgen_model/.
        assert GENE_NODE_DIM == 9
        assert GENE_EDGE_DIM == 3
