# metrics.py
# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani

from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class TaskEvaluator:
    """
    Evaluator Engine.

    PATRÓN: Adapter.
    Adapta la salida del DataLoader (Collater específico) a la entrada esperada por el Modelo
    y calcula métricas agnósticas a la tarea (Single-label vs Multi-label).
    """

    def __init__(self, device: torch.device):
        self.device = device

    def _compute_single_task_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, is_multilabel: bool
    ) -> dict[str, float]:
        """
        Calcula métricas estandarizadas.
        """
        if is_multilabel:
            return {
                "f1_macro": float(
                    f1_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "f1_samples": float(
                    f1_score(y_true, y_pred, average="samples", zero_division=0)
                ),
            }

        # Para clasificación multiclase simple
        return {
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "acc": float((y_pred == y_true).mean()),
        }

    def evaluate(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        target_cols: list[str],
        multi_label_cols: set[str],
        threshold: float = 0.5,
    ) -> dict[str, dict[str, float]]:
        """
        Ejecuta el loop de evaluación respetando la estructura de grafos Two-Tower.
        """
        model.eval()

        # Contenedores para acumular predicciones y targets por tarea
        all_preds: dict[str, list[torch.Tensor]] = {c: [] for c in target_cols}
        all_targets: dict[str, list[torch.Tensor]] = {c: [] for c in target_cols}

        with torch.no_grad():
            for batch in data_loader:
                # 1. Desempaquetado Seguro (Alineado con DoubleTowerCollater)
                # Se asume que el batch contiene las claves 'drug_batch' y 'haplo_batch'
                if not ("drug_batch" in batch and "haplo_batch" in batch):
                    raise ValueError(
                        "El batch no contiene la estructura de grafos requerida (drug_batch, haplo_batch)."
                    )

                drug_data = batch["drug_batch"].to(self.device)
                haplo_data = batch["haplo_batch"].to(self.device)

                # Targets suelen venir en un sub-diccionario 'targets' según tu Collater
                batch_targets = {
                    k: v.to(self.device) for k, v in batch.get("targets", {}).items()
                }

                # 2. Inferencia (Forward Pass explícito)
                # Coincide con la firma de PharmagenTwoTower.forward
                outputs = model(drug_data=drug_data, haplo_data=haplo_data)

                # 3. Procesamiento de Salidas
                for col in target_cols:
                    if col not in batch_targets:
                        continue

                    logits = outputs[col]
                    true_v = batch_targets[col]

                    # Decisión basada en tipo de tarea
                    if col in multi_label_cols:
                        # Multi-label: Sigmoid + Umbral
                        pred = (torch.sigmoid(logits) > threshold).float()
                    else:
                        # Single-label / Multi-class: Argmax
                        pred = torch.argmax(logits, dim=1)

                    all_preds[col].append(pred.cpu())
                    all_targets[col].append(true_v.cpu())

        # 4. Agregación y Cálculo
        metrics = {}
        for col in target_cols:
            if not all_preds[col]:
                continue

            y_p = torch.cat(all_preds[col]).numpy()
            y_t = torch.cat(all_targets[col]).numpy()

            metrics[col] = self._compute_single_task_metrics(
                y_true=y_t, y_pred=y_p, is_multilabel=(col in multi_label_cols)
            )

        return metrics
