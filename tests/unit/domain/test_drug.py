"""Tests for src.domain.drug."""

import pytest

from src.domain.drug import Drug

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_CANONICAL = "CC(=O)Oc1ccccc1C(=O)O"


class TestDrug:
    def test_minimal_construction(self) -> None:
        d = Drug(name="aspirin", cid=2244, smiles=ASPIRIN_SMILES)
        # Without `from_smiles` the SMILES is preserved as given (only stripped).
        assert d.smiles == ASPIRIN_SMILES
        assert d.molecule is None
        assert d.graph is None

    def test_strip_smiles_whitespace(self) -> None:
        d = Drug(name="aspirin", cid=2244, smiles=f"  {ASPIRIN_SMILES}  ")
        assert d.smiles == ASPIRIN_SMILES

    def test_empty_smiles_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Drug(name="x", cid=1, smiles="   ")

    def test_zero_cid_rejected(self) -> None:
        with pytest.raises(ValueError):
            Drug(name="x", cid=0, smiles="C")

    def test_negative_cid_rejected(self) -> None:
        with pytest.raises(ValueError):
            Drug(name="x", cid=-1, smiles="C")

    def test_from_smiles_canonicalizes(self) -> None:
        d = Drug.from_smiles(name="aspirin", cid=2244, smiles=ASPIRIN_SMILES)
        assert d.smiles == ASPIRIN_CANONICAL
        assert d.molecule is not None
        # RDKit Mol exposes GetNumAtoms; quick sanity check.
        assert d.molecule.GetNumAtoms() == 13

    def test_from_smiles_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="could not parse"):
            Drug.from_smiles(name="bogus", cid=1, smiles="not_a_smiles_string!!!")

    def test_ph_group_optional(self) -> None:
        d = Drug.from_smiles(name="x", cid=1, smiles="C")
        assert d.ph_group is None
        d2 = Drug.from_smiles(name="aspirin", cid=2244, smiles=ASPIRIN_SMILES, ph_group="N02BA01")
        assert d2.ph_group == "N02BA01"
