"""Trainers for the Pharmagen training pipeline.

    from src.model.training import StandardTrainer, OptunaTrialTrainer

    trainer = StandardTrainer(model, optimizer, scheduler, ...)
    best_loss = trainer.fit(train_loader, val_loader, epochs=100, patience=10)
"""

from src.model.training.loop import TrainingLoop
from src.model.training.optuna_trainer import OptunaTrialTrainer
from src.model.training.standard import StandardTrainer

__all__ = ["OptunaTrialTrainer", "StandardTrainer", "TrainingLoop"]
