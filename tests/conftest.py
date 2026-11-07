"""Shared pytest fixtures for Pharmagen test suite."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mock_encoder():
    """Mock sklearn LabelEncoder."""
    encoder = MagicMock()
    encoder.classes_ = np.array(["A", "B", "C"])
    encoder.transform.return_value = np.array([0, 1, 2])
    encoder.inverse_transform.return_value = np.array(["A", "B", "C"])
    return encoder


# --------------------------------------------------------------------------- #
# A tiny Modephere drug library (2D only), built once per session
# --------------------------------------------------------------------------- #

#: Drugs of the test library: ChEMBL ID -> SMILES.
LIBRARY_DRUGS = {"CHEMBL25": "CC(=O)Oc1ccccc1C(=O)O", "CHEMBL545": "CCO"}


def make_modephere_library(root, drugs, *, three_d=()):
    """A real Modephere store at ``root`` with ``drugs`` (ChEMBL ID -> SMILES).

    Compounds are 2D-only except the IDs in ``three_d`` (a few conformers each).
    Returns ``(LibraryStore, {chembl_id: inchikey})``. Each compound is built by
    Modephere and stamped as resolved against ChEMBL, as a staged-release build
    would.
    """
    from modephere.library import LibraryStore, PipelineOptions, build_compound
    from modephere.types import ChemblMatch, ConformerParams

    flat = PipelineOptions(build_3d=False)
    fast = PipelineOptions(
        conformers=ConformerParams(num_conformers=4, min_conformers=1, max_conformers=4)
    )
    store = LibraryStore(root)
    arts = []
    for chembl_id, smiles in drugs.items():
        art = build_compound(smiles, fast if chembl_id in three_d else flat)
        resolved = art.compound.resolved.model_copy(
            update={
                "chembl_id": chembl_id,
                "chembl_record_id": chembl_id,
                "chembl_match": ChemblMatch.STRUCTURE_INCHIKEY,
                "chembl_release": "chembl_37",
            }
        )
        compound = art.compound.model_copy(update={"resolved": resolved})
        art = art.model_copy(update={"compound": compound})
        store.save(art)
        arts.append(art)
    store.upsert(arts)
    return store, {a.compound.resolved.chembl_id: a.inchikey for a in arts}


#: 3D aspirin, 2D-only ethanol, thiotepa (no pharmacophore feature: K = 0).
PHARM_DRUGS = {
    "CHEMBL25": "CC(=O)Oc1ccccc1C(=O)O",
    "CHEMBL545": "CCO",
    "CHEMBL671": "S=P(N1CC1)(N1CC1)N1CC1",
}


@pytest.fixture(scope="session")
def pharm_library(tmp_path_factory):
    """``(LibraryStore, {chembl_id: inchikey})``: 3D, 2D-only and K = 0 drugs."""
    return make_modephere_library(
        tmp_path_factory.mktemp("pharm_library"), PHARM_DRUGS, three_d={"CHEMBL25"}
    )


@pytest.fixture(scope="session")
def modephere_library(tmp_path_factory):
    """``(LibraryStore, {chembl_id: inchikey})`` of a real 2D Modephere store."""
    return make_modephere_library(
        tmp_path_factory.mktemp("modephere_library"), LIBRARY_DRUGS
    )
