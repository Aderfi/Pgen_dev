"""Loss functions for Pharmagen training.

Re-exports the legacy DeepFM-era losses (``legacy.py``) alongside the new
two-tower GNN loss (``multitask.py`` / ``focal.py``) so existing importers
keep working while new code adopts the new-model API.
"""

from .focal import focal_bce_with_logits
from .legacy import (
    AdaptiveFocalLoss,
    AsymmetricLoss,
    FocalLoss,
    MultiTaskUncertaintyLoss,
    PolyLoss,
)
from .multitask import MultiTaskLoss

__all__ = [
    "AdaptiveFocalLoss",
    "AsymmetricLoss",
    "FocalLoss",
    "MultiTaskLoss",
    "MultiTaskUncertaintyLoss",
    "PolyLoss",
    "focal_bce_with_logits",
]
