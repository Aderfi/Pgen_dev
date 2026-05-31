"""Graph metadata models.

These describe a PyG ``Data`` object without holding it — useful for indexes,
catalogues, and FastAPI responses where shipping the full tensor is wasteful.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from torch_geometric.data import Data as PyGData


class GraphKind(str, Enum):
    DRUG = "drug"
    GENE = "gene"


class GraphMetadata(BaseModel):
    """Lightweight description of a stored graph artifact.

    `path` points at the .pt file produced by the library builder. It's a
    Path, not a string, so downstream code gets the typed object for free.
    """

    model_config = ConfigDict(frozen=True)

    kind: GraphKind
    identifier: str = Field(..., description="Drug CID or 'GENE/variant' key.")
    path: Path
    num_nodes: NonNegativeInt
    num_edges: NonNegativeInt
    feature_dim: PositiveInt
    edge_dim: NonNegativeInt = 0
    source: str | None = Field(
        default=None, description="Builder version or pipeline run id."
    )


class GraphPair(BaseModel):
    """A drug graph paired with a gene/genotype graph — the input shape for the
    Two-Tower model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    drug: PyGData
    gene: PyGData
    drug_id: str
    gene_id: str
