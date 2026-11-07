"""A row's drug set: molecules -> entities -> one drug-side vector per row.

An entity is one drug (one molecule) or one ATC class (its member molecules).
Entities are the **mean** of their molecules — class members are alternatives,
and a mean keeps the class size from scaling the vector — plus a learned
entity-kind embedding (drug / class). A row is an attention-weighted sum of
its entities: a combination is co-administration, so its drugs are weighed,
not averaged blindly.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.utils import scatter, softmax

#: Entity kinds (``src.data.cache.ENTITY_DRUG`` / ``ENTITY_CLASS``).
NUM_ENTITY_KINDS = 2


class DrugSetPooling(nn.Module):
    """``mol_emb [M, d]`` -> ``(entity_emb [N_ent, d], row_emb [num_rows, d])``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(NUM_ENTITY_KINDS, dim)
        nn.init.normal_(self.kind_embedding.weight, std=0.02)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(
        self,
        mol_emb: Tensor,
        mol_to_entity: Tensor,
        entity_kind: Tensor,
        entity_to_row: Tensor,
        num_rows: int,
    ) -> tuple[Tensor, Tensor]:
        num_entities = entity_to_row.numel()
        entities = scatter(
            mol_emb, mol_to_entity, dim=0, dim_size=num_entities, reduce="mean"
        )
        entities = entities + self.kind_embedding(entity_kind)
        alpha = softmax(self.gate(entities), entity_to_row, num_nodes=num_rows)
        # a single-entity row is that entity (alpha = 1)
        rows = scatter(
            alpha * entities, entity_to_row, dim=0, dim_size=num_rows, reduce="sum"
        )
        return entities, rows


__all__ = ["NUM_ENTITY_KINDS", "DrugSetPooling"]
