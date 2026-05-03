"""Tests for src.data.library.drugs.

The schema dimensions are tested explicitly because changing them silently
would invalidate every previously-trained model — they must stay 25/7.
"""

import pytest
import torch

from src.data.library.drugs import (
    DRUG_EDGE_DIM,
    DRUG_NODE_DIM,
    safe_filename,
    smiles_to_graph,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SIMPLE_METHANE = "C"


class TestSafeFilename:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("simple", "simple"),
            ("weird:name/with*chars", "weird_name_with_chars"),
            ("with spaces", "with_spaces"),
            ("trailing space    ", "trailing_space"),
            ('quote"slash\\bar|gt>', "quote_slash_bar_gt_"),
        ],
    )
    def test_normalizes_unsafe_chars(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected

    def test_caps_length(self) -> None:
        long = "a" * 500
        assert len(safe_filename(long, max_length=100)) == 100


class TestSmilesToGraph:
    def test_aspirin_dimensions(self) -> None:
        g = smiles_to_graph(ASPIRIN)
        assert g is not None
        assert g.x.shape[1] == DRUG_NODE_DIM == 25
        assert g.edge_attr.shape[1] == DRUG_EDGE_DIM == 7
        assert g.edge_index.shape[0] == 2
        # 13 heavy atoms, 13 bonds (× 2 directions = 26 edges)
        assert g.x.shape[0] == 13
        assert g.edge_index.shape[1] == 26

    def test_methane_no_bonds(self) -> None:
        g = smiles_to_graph(SIMPLE_METHANE)
        assert g is not None
        assert g.x.shape == (1, DRUG_NODE_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_attr.shape == (0, DRUG_EDGE_DIM)

    @pytest.mark.parametrize("bad", ["not_smiles!!!", "", "   ", None])
    def test_invalid_returns_none(self, bad) -> None:
        assert smiles_to_graph(bad) is None  # type: ignore[arg-type]

    def test_aromatic_flag_set(self) -> None:
        # Benzene — every atom is aromatic.
        g = smiles_to_graph("c1ccccc1")
        assert g is not None
        # Feature layout: atomic_num(1) + degree(1) + one-hot degree(5)
        # + charge(5) + hybridization(3) + Hs(5) + chiral(3) = 23, then aromatic at index 23.
        aromatic_col = g.x[:, 23]
        assert (aromatic_col > 0).all(), "every benzene atom should be aromatic"

    def test_dtype_consistency(self) -> None:
        g = smiles_to_graph(ASPIRIN)
        assert g is not None
        assert g.x.dtype == torch.float
        assert g.edge_index.dtype == torch.long
        assert g.edge_attr.dtype == torch.float
