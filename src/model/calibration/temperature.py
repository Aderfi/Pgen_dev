"""Per-task temperature scaling for post-hoc probability calibration."""

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from src.model.architectures.config import TaskSpec


class TemperatureScaler(nn.Module):
    """Per-task temperature scaling for post-hoc probability calibration.

    A clinically usable model needs probabilities that are calibrated, not
    merely correctly ranked: "12% risk of therapeutic failure" has to mean 12%.
    Fit this on a held-out calibration split (never on train, never on test) and
    report expected calibration error alongside AUPRC.
    """

    def __init__(self, targets: dict[str, TaskSpec]) -> None:
        super().__init__()
        self.log_temp = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(()))
                for name, spec in targets.items()
                if spec.enabled and spec.kind in ("binary", "multiclass")
            }
        )

    def forward(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        scaled = dict(outputs)
        for name, log_t in self.log_temp.items():
            if name in scaled:
                scaled[name] = scaled[name] / torch.exp(log_t).clamp(min=1e-3)
        return scaled

    def fit(
        self,
        logits: dict[str, Tensor],
        labels: dict[str, Tensor],
        targets: dict[str, TaskSpec],
        max_iter: int = 200,
    ) -> None:
        """Optimise one temperature per task on a calibration split (LBFGS)."""
        for name, param in self.log_temp.items():
            if name not in logits or name not in labels:
                continue
            kind = targets[name].kind
            opt = torch.optim.LBFGS([param], lr=0.05, max_iter=max_iter)

            def closure() -> Tensor:
                opt.zero_grad()
                t = torch.exp(param).clamp(min=1e-3)
                if kind == "binary":
                    loss = F.binary_cross_entropy_with_logits(
                        logits[name] / t, labels[name].view_as(logits[name]).float()
                    )
                else:
                    loss = F.cross_entropy(
                        logits[name] / t, labels[name].view(-1).long()
                    )
                loss.backward()
                return loss

            opt.step(closure)  # type: ignore[arg-type]
