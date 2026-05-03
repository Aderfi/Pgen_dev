"""Back-compat shim for the legacy ``PGenTrainer`` class.

Old callers (``src.pipeline``, ``src.modeling.engine.tuner``) construct a
trainer like::

    trainer = PGenTrainer(model, optimizer, scheduler, device, ...,
                           from_optuna=True | False)

That code path keeps working: ``PGenTrainer`` is now a tiny factory that
returns either :class:`StandardTrainer` or :class:`OptunaTrialTrainer` based
on the ``from_optuna`` flag.

New code should import the explicit subclass directly:

    from src.modeling.training import StandardTrainer, OptunaTrialTrainer
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Set
from typing import Any

import torch
from torch import nn

from src.modeling.training import OptunaTrialTrainer, StandardTrainer, TrainingLoop
from src.utils.losses import MultiTaskUncertaintyLoss


__all__ = [
    "OptunaTrialTrainer",
    "PGenTrainer",
    "StandardTrainer",
    "TrainingLoop",
]


def PGenTrainer(  # noqa: N802 — back-compat alias for class name
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    target_cols: MutableSequence[str],
    multi_label_cols: Set[str],
    params: Mapping[str, Any],
    uncertainty_module: MultiTaskUncertaintyLoss | None = None,
    from_optuna: bool = False,
) -> TrainingLoop:
    """Pick the right trainer subclass based on the legacy ``from_optuna`` flag."""
    if from_optuna:
        return OptunaTrialTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            target_cols=target_cols,
            multi_label_cols=multi_label_cols,
            params=params,
            uncertainty_module=uncertainty_module,
        )
    return StandardTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        target_cols=target_cols,
        multi_label_cols=multi_label_cols,
        params=params,
        uncertainty_module=uncertainty_module,
    )
