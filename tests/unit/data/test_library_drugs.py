"""Tests for src.data.library.drugs.

The schema dimensions are tested explicitly because changing them silently
would invalidate every previously-trained model — they must stay 61/18/1038.
"""

import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem

from src.data.library.drugs import (
    DRUG_EDGE_DIM,
    DRUG_GLOBAL_DIM,
    DRUG_NODE_DIM,
    molecular_descriptors,
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
        assert g.x.shape[1] == DRUG_NODE_DIM == 61
        assert g.edge_attr.shape[1] == DRUG_EDGE_DIM == 18
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
        # Feature layout: element(13) + degree(7) + valence(7) + charge[scalar+1h](5)
        # + hybridization(7) + Hs(5) + chirality(4) = 48, then is_aromatic at index 48.
        aromatic_col = g.x[:, 48]
        assert (aromatic_col > 0).all(), "every benzene atom should be aromatic"

    def test_dtype_consistency(self) -> None:
        g = smiles_to_graph(ASPIRIN)
        assert g is not None
        assert g.x.dtype == torch.float
        assert g.edge_index.dtype == torch.long
        assert g.edge_attr.dtype == torch.float


class TestGlobalDescriptors:
    """Per-molecule QSAR + ECFP global vector attached as ``global_feats``."""

    def test_descriptor_vector_length(self) -> None:
        mol = Chem.MolFromSmiles(ASPIRIN)
        vec = molecular_descriptors(mol)
        assert len(vec) == DRUG_GLOBAL_DIM == 1038

    def test_global_feats_attached_to_graph(self) -> None:
        g = smiles_to_graph(ASPIRIN)
        assert g is not None
        assert hasattr(g, "global_feats")
        assert g.global_feats.shape == (1, DRUG_GLOBAL_DIM)
        assert g.global_feats.dtype == torch.float

    def test_no_nan_or_inf(self) -> None:
        vec = torch.tensor(molecular_descriptors(Chem.MolFromSmiles(ASPIRIN)))
        assert torch.isfinite(vec).all()

    def test_similar_molecules_have_similar_vectors(self) -> None:
        # Homologs (hexanol/heptanol) should be far closer than hexanol/pyridine.
        hexanol = torch.tensor(molecular_descriptors(Chem.MolFromSmiles("CCCCCCO")))
        heptanol = torch.tensor(molecular_descriptors(Chem.MolFromSmiles("CCCCCCCO")))
        pyridine = torch.tensor(molecular_descriptors(Chem.MolFromSmiles("c1ccncc1")))
        sim_homolog = F.cosine_similarity(hexanol, heptanol, dim=0)
        sim_distinct = F.cosine_similarity(hexanol, pyridine, dim=0)
        assert sim_homolog > sim_distinct
        assert sim_homolog > 0.8
