"""Compositional label contrastive loss for the two-tower GNN model.

The model composes a label embedding ``outputs["_z"]`` from its predicted
components. This loss pulls that composed embedding toward the compositional
embedding of the true label tuple (``target_emb``) and pushes it away from the
other labels present in the same batch, using an InfoNCE-style cosine
contrastive objective.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CompositionalLabelLoss(nn.Module):
    """InfoNCE-style cosine contrastive loss over compositional label embeddings.

    Each row of ``z`` (the model's composed label embedding) is treated as an
    anchor whose positive is the same-row entry of ``target_emb`` (the
    compositional embedding of the true label tuple); the other rows of
    ``target_emb`` in the batch serve as in-batch negatives.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z: Tensor, target_emb: Tensor) -> Tensor:
        z_norm = F.normalize(z, dim=-1)
        target_norm = F.normalize(target_emb, dim=-1)
        sim = (z_norm @ target_norm.t()) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)
