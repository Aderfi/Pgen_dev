"""Loss functions for Pharmagen training.

The current two-tower GNN model trains with ``MultiTaskLoss`` (learned
per-task uncertainty weighting over ``focal_bce_with_logits`` / cross-entropy
/ smooth-L1 per axis kind) and ``CompositionalLabelLoss`` (the composed-label
embedding loss). The legacy DeepFM-era loss classes (``FocalLoss``,
``AdaptiveFocalLoss``, ``AsymmetricLoss``, ``PolyLoss``,
``MultiTaskUncertaintyLoss``) have been removed — they had no importers left
outside this package.
"""

from .compositional import CompositionalLabelLoss
from .focal import focal_bce_with_logits
from .multitask import MultiTaskLoss

__all__ = [
    "CompositionalLabelLoss",
    "MultiTaskLoss",
    "focal_bce_with_logits",
]
