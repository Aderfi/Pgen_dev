"""Multi-task loss with learned homoscedastic uncertainty weighting."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.architectures.config import TaskSpec

from .focal import focal_bce_with_logits


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

    def __init__(self, targets: dict[str, TaskSpec]) -> None:
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
