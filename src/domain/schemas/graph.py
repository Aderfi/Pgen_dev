"""Graph metadata models.

These describe a PyG ``Data`` object without holding it — useful for indexes,
catalogues, and FastAPI responses where shipping the full tensor is wasteful.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt
from torch_geometric.data import Data as PyGData

from src.domain.base import GraphDomainModel

GraphKind = Literal["drug", "gene"]


class GraphMetadata(GraphDomainModel):
    """Lightweight description of a stored graph artifact.

    `path` points at the .pt file produced by the library builder. It's a
    Path, not a string, so downstream code gets the typed object for free.
    """

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


class GraphPair(GraphDomainModel):
    """A drug graph paired with a gene/genotype graph — the input shape for the
    Two-Tower model.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # holds live PyG tensors

    drug: PyGData
    gene: PyGData
    drug_id: str
    gene_id: str
