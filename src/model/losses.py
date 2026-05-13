"""Custom loss functions for Pharmagen multi-task training.

- :class:`FocalLoss` — single-label imbalanced classification.
- :class:`AdaptiveFocalLoss` — focal loss with per-batch gamma adjustment.
- :class:`AsymmetricLoss` — multi-label loss that down-weights easy negatives.
- :class:`PolyLoss` — polynomial-expansion loss (Leng et al., ICLR 2022).
- :class:`MultiTaskUncertaintyLoss` — learned per-task weighting (Kendall & Gal, 2018).
"""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss for imbalanced single-label classification."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class AdaptiveFocalLoss(FocalLoss):
    """Focal loss that adjusts gamma based on the batch accuracy."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean()
            self.gamma = float(3.0 - acc * 2.0)  # range [1.0, 3.0]
        return super().forward(logits, targets)


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification.

    Reduces the weight of easy negatives.
    Reference: Ben-Baruch et al., "Asymmetric Loss for Multi-Label Classification".
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        xs_neg = (1 - probs + self.clip).clamp(max=1) if self.clip > 0 else 1 - probs

        loss_pos = -targets * torch.log(probs.clamp(min=self.eps)) * ((1 - probs) ** self.gamma_pos)
        loss_neg = -(1 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * ((1 - xs_neg) ** self.gamma_neg)
        return (loss_pos + loss_neg).mean()


class PolyLoss(nn.Module):
    """Poly-1 loss: polynomial expansion of cross-entropy.

    Reference: Leng et al., ICLR 2022.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.smoothing)
        poly = ce + self.epsilon * (1 - torch.exp(-ce))
        if self.reduction == "mean":
            return poly.mean()
        if self.reduction == "sum":
            return poly.sum()
        return poly


class MultiTaskUncertaintyLoss(nn.Module):
    """Learned per-task loss weighting.

    Reference: Kendall & Gal, NeurIPS 2018.
    ``L = L_i * exp(-σ_i) + σ_i`` where σ_i are learnable log-uncertainty parameters.
    """

    def __init__(self, tasks: list[str]):
        super().__init__()
        self.log_sigmas = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(1)) for t in tasks}
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total: torch.Tensor = sum(  # type: ignore[assignment]
            loss * torch.exp(-self.log_sigmas[task]) + self.log_sigmas[task]
            for task, loss in losses.items()
        )
        return total
