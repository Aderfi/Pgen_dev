from pydantic import BaseModel, ConfigDict, PositiveInt
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data as PyGData


class Drug(BaseModel):
    """Pydantic model for a drug molecule.

    Note: Phase 2 will move this to src/domain/drug.py with full validation.
    Kept here in Phase 1 just to make the type-check valid.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    cid: PositiveInt
    smiles: str
    ph_group: str
    molecule: Mol
    graph: PyGData | None = None
