"""Loss functions for Pharmagen training.

The current two-tower GNN model trains with ``MultiTaskLoss`` (learned
per-task uncertainty weighting over ``focal_bce_with_logits`` / cross-entropy
/ smooth-L1 per axis kind) and ``CompositionalLabelLoss`` (the composed-label
embedding loss). The legacy DeepFM-era loss classes (``FocalLoss``,
``AdaptiveFocalLoss``, ``AsymmetricLoss``, ``PolyLoss``,
``MultiTaskUncertaintyLoss``) have been removed — they had no importers left
outside this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from src.model.architectures.config import AxisSpec

__all__ = [
    "CompositionalLabelLoss",
    "MultiTaskLoss",
    "focal_bce_with_logits",
]


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


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learned homoscedastic uncertainty weighting.

    Summing raw task losses lets whichever task has the largest natural scale
    dominate the gradient. This implements Kendall et al.'s uncertainty
    weighting: each task gets a learned log-variance, so the optimiser balances
    the tasks instead of the practitioner guessing weights.

    Targets are supplied as a dict of task-name -> tensor. Missing tasks are
    skipped, and NaN entries are masked out, so partially labelled cohorts work
    without building separate models.
    """

    def __init__(self, targets: dict[str, AxisSpec]) -> None:
        super().__init__()
        self.specs = {n: s for n, s in targets.items() if s.enabled}
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.specs}
        )
        for name, spec in self.specs.items():
            if spec.kind == "binary" and spec.pos_weight is not None:
                self.register_buffer(
                    f"pos_weight_{name}", torch.tensor(float(spec.pos_weight))
                )
            if spec.kind == "multiclass" and spec.class_weights is not None:
                self.register_buffer(
                    f"class_weight_{name}", torch.tensor(list(spec.class_weights))
                )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total = torch.zeros((), device=next(iter(outputs.values())).device)
        per_task: dict[str, Tensor] = {}

        for name, spec in self.specs.items():
            if name not in outputs or name not in targets:
                continue
            logits, y = outputs[name], targets[name]

            if spec.kind in ("binary", "regression"):
                logits = logits.view(-1, spec.dim)
                y = y.view(-1, spec.dim).to(logits.dtype)
                mask = ~torch.isnan(y)
                if not mask.any():
                    continue
                logits, y = logits[mask], y[mask]
            else:
                y = y.view(-1).long()
                mask = y >= 0  # convention: -1 marks an unlabelled sample
                if not mask.any():
                    continue
                logits, y = logits[mask], y[mask]

            if spec.kind == "binary":
                pw = getattr(self, f"pos_weight_{name}", None)
                loss = focal_bce_with_logits(
                    logits, y, gamma=spec.focal_gamma, pos_weight=pw
                )
                scale = 1.0
            elif spec.kind == "multiclass":
                cw = getattr(self, f"class_weight_{name}", None)
                loss = F.cross_entropy(logits, y, weight=cw)
                scale = 1.0
            else:
                loss = F.smooth_l1_loss(logits, y)
                scale = 0.5  # Kendall's Gaussian-likelihood factor

            log_var = self.log_vars[name]
            total = total + scale * torch.exp(-log_var) * loss + 0.5 * log_var
            per_task[name] = loss.detach()

        return total, per_task


class CompositionalLabelLoss(nn.Module):
    """InfoNCE-style cosine contrastive loss over compositional label embeddings.

    The model composes a label embedding ``outputs["_z"]`` from its predicted
    components. This loss pulls that composed embedding toward the compositional
    embedding of the true label tuple (``target_emb``) and pushes it away from
    the other labels present in the same batch, using an InfoNCE-style cosine
    contrastive objective.

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
