"""Tests for src.domain.drug."""

import pytest

from src.domain.schemas.drug import Drug

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_CANONICAL = "CC(=O)Oc1ccccc1C(=O)O"


class TestDrug:
    def test_minimal_construction(self) -> None:
        d = Drug(name="aspirin", chembl_id="CHEMBL25", smiles=ASPIRIN_SMILES)
        # Without `from_smiles` the SMILES is preserved as given (only stripped).
        assert d.smiles == ASPIRIN_SMILES
        assert d.molecule is None
        assert d.graph is None

    def test_strip_smiles_whitespace(self) -> None:
        d = Drug(name="aspirin", chembl_id="CHEMBL25", smiles=f"  {ASPIRIN_SMILES}  ")
        assert d.smiles == ASPIRIN_SMILES

    def test_empty_smiles_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Drug(name="x", chembl_id="CHEMBL1", smiles="   ")

    def test_chembl_id_is_normalised(self) -> None:
        assert (
            Drug(name="x", chembl_id=" chembl25 ", smiles="C").chembl_id == "CHEMBL25"
        )

    @pytest.mark.parametrize("bad", ["2244", "CHEMBL", "CHEMBLX1", ""])
    def test_malformed_chembl_id_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            Drug(name="x", chembl_id=bad, smiles="C")

    def test_from_smiles_canonicalizes(self) -> None:
        d = Drug.from_smiles(
            name="aspirin", chembl_id="CHEMBL25", smiles=ASPIRIN_SMILES
        )
        assert d.smiles == ASPIRIN_CANONICAL
        assert d.molecule is not None
        # RDKit Mol exposes GetNumAtoms; quick sanity check.
        assert d.molecule.GetNumAtoms() == 13

    def test_from_smiles_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="could not parse"):
            Drug.from_smiles(
                name="bogus", chembl_id="CHEMBL1", smiles="not_a_smiles_string!!!"
            )

    def test_ph_group_optional(self) -> None:
        d = Drug.from_smiles(name="x", chembl_id="CHEMBL1", smiles="C")
        assert d.ph_group is None
        d2 = Drug.from_smiles(
            name="aspirin",
            chembl_id="CHEMBL25",
            smiles=ASPIRIN_SMILES,
            ph_group="N02BA01",
        )
        assert d2.ph_group == "N02BA01"
