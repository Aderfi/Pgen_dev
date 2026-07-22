"""Configuration dataclasses for :class:`PharmagenTwoTower`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence


ConvType = Literal["gine", "gatv2"]
TaskKind = Literal["binary", "multiclass", "regression"]


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Declarative description of one prediction head.

    Args:
        dim: Output dimensionality (1 for binary / regression, C for multiclass).
        kind: Determines the loss and the calibration strategy.
        pos_weight: Positive-class weight for binary tasks. Set this to
            ``n_negative / n_positive``; with adverse-event rates below 1% an
            unweighted BCE collapses to the majority class.
        focal_gamma: If > 0, use focal loss instead of plain BCE for binary
            tasks. ``gamma=2.0`` is the usual starting point.
        class_weights: Per-class weights for multiclass tasks.
        enabled: Allows switching a head off without changing the config shape.
    """

    dim: int
    kind: TaskKind = "binary"
    pos_weight: float | None = None
    focal_gamma: float = 0.0
    class_weights: Sequence[float] | None = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Main model config
# ---------------------------------------------------------------------------


@dataclass
class PharmagenConfig:
    """Configuration for :class:`PharmagenTwoTower`.

    The ``use_*`` flags exist so that ablations are a config change rather than a
    code change. Run at minimum:
      * GNN only            (use_drug_global=False, use_drug_admet=False)
      * descriptors only    (use_mol_gnn=False)
      * both
    If the GNN adds no measurable delta over descriptors alone, prefer gradient
    boosting on descriptors: it is easier to validate and to defend clinically.
    """

    # --- Drug (molecular) tower ---
    drug_in_features: int = 0
    drug_edge_dim: int | None = None
    drug_hidden_dim: int = 256

    # --- Polypharmacy (patient-level drug-drug interaction) graph ---
    ddi_edge_dim: int | None = None
    polypharmacy_layers: int = 2

    # --- Genotype tower ---
    geno_in_features: int = 0
    geno_edge_dim: int | None = None
    geno_hidden_dim: int = 256

    # --- Shared ---
    embedding_dim: int = 256
    num_layers: int = 4
    heads: int = 4
    dropout: float = 0.2

    # --- Auxiliary per-molecule / per-genotype descriptor blocks (0 disables) ---
    drug_global_dim: int = 0  # QSAR physchem + ECFP + Murcko scaffold hash
    drug_admet_dim: int = 0  # predicted ADMET / CYP interaction profile
    geno_global_dim: int = 0  # graph-level PGx function vector (activity scores)

    # --- Ablation / structural switches ---
    use_mol_gnn: bool = True
    use_polypharmacy: bool = True
    use_cross_attention: bool = True

    # --- Tasks ---
    targets: dict[str, TaskSpec] = field(default_factory=dict)


__all__ = ["ConvType", "TaskKind", "TaskSpec", "PharmagenConfig"]
