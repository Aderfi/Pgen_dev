"""Focal binary cross-entropy for the new two-tower GNN model."""

import torch
from torch import Tensor
from torch.nn import functional as F


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    pos_weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Focal binary cross-entropy.

    Adverse pharmacogenomic events are rare (often <1%). Plain BCE converges to
    predicting the majority class with a flattering AUROC that means nothing;
    focal loss down-weights the easy negatives that dominate the gradient.
    Always evaluate these tasks with AUPRC, not AUROC.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        bce = bce * (1.0 - p_t).pow(gamma)
    if reduction == "mean":
        return bce.mean()
    if reduction == "sum":
        return bce.sum()
    return bce
