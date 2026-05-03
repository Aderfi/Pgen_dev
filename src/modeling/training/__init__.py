"""Trainers — split from the old PGenTrainer god-class.

Use the explicit subclasses in new code:

    from src.modeling.training import StandardTrainer, OptunaTrialTrainer

    trainer = StandardTrainer(model, optimizer, scheduler, ...)
    best_loss = trainer.fit(train_loader, val_loader, epochs=100, patience=10)

The legacy ``PGenTrainer`` factory (``src.modeling.engine.trainer``) keeps
working for code paths that pass ``from_optuna=True/False``.
"""

from src.modeling.training.loop import TrainingLoop
from src.modeling.training.optuna_trainer import OptunaTrialTrainer
from src.modeling.training.standard import StandardTrainer

__all__ = ["OptunaTrialTrainer", "StandardTrainer", "TrainingLoop"]
