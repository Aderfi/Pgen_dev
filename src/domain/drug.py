"""Drug model — small molecule with SMILES and (optionally) a graph encoding.

The SMILES validator is opt-in: by default we trust the caller, but
``Drug.from_smiles()`` runs RDKit's canonicalizer and rejects unparseable
strings. Use that constructor when accepting external input.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data as PyGData


class Drug(BaseModel):
    """A drug molecule keyed by PubChem CID.

    `molecule` (RDKit Mol) and `graph` (PyG Data) are optional so callers can
    construct a Drug from a database row before chemistry/graph derivation
    has run. The `from_smiles` classmethod populates everything in one shot.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        ..., description="Cleaned drug name; used for filenames and display."
    )
    cid: PositiveInt = Field(..., description="PubChem Compound ID.")
    smiles: str = Field(..., description="Canonical SMILES string.")
    ph_group: str | None = Field(
        default=None, description="ATC/pharmacological group code."
    )
    molecule: Mol | None = None
    graph: PyGData | None = None

    @field_validator("smiles", mode="before")
    @classmethod
    def _strip_smiles(cls, v: str) -> str:
        if not isinstance(v, str):
            msg = f"smiles must be str, got {type(v).__name__}"
            raise TypeError(msg)
        stripped = v.strip()
        if not stripped:
            msg = "smiles cannot be empty"
            raise ValueError(msg)
        return stripped

    @classmethod
    def from_smiles(
        cls,
        *,
        name: str,
        cid: int,
        smiles: str,
        ph_group: str | None = None,
        graph: PyGData | None = None,
        **extra: Any,
    ) -> Drug:
        """Build a Drug from a SMILES string, validating with RDKit.

        Raises ValueError if the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            msg = (
                f"RDKit could not parse SMILES {smiles!r} for drug {name!r} (CID {cid})"
            )
            raise ValueError(msg)
        canonical = Chem.MolToSmiles(mol)
        return cls(
            name=name,
            cid=cid,
            smiles=canonical,
            ph_group=ph_group,
            molecule=mol,
            graph=graph,
            **extra,
        )
