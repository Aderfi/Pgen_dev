"""Configuration models for :class:`PharmagenTwoTower`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ConvType = Literal["gine", "gatv2"]
TaskKind = Literal["binary", "multiclass", "regression", "ordinal"]


# ---------------------------------------------------------------------------
# Axis specification
# ---------------------------------------------------------------------------


class AxisSpec(BaseModel):
    """Declarative description of one prediction head.

    Args:
        name: Human-readable identifier for the axis/head.
        dim: Output dimensionality (1 for binary / regression, C for multiclass).
        kind: Determines the loss and the calibration strategy.
        embedding_dim: Size of any auxiliary embedding associated with this axis.
        pos_weight: Positive-class weight for binary tasks. Set this to
            ``n_negative / n_positive``; with adverse-event rates below 1% an
            unweighted BCE collapses to the majority class.
        focal_gamma: If > 0, use focal loss instead of plain BCE for binary
            tasks. ``gamma=2.0`` is the usual starting point.
        class_weights: Per-class weights for multiclass tasks.
        enabled: Allows switching a head off without changing the config shape.
    """

    name: str = ""
    dim: int
    kind: TaskKind = "binary"
    embedding_dim: int = 32
    pos_weight: float | None = None
    focal_gamma: float = 0.0
    class_weights: list[float] | None = None
    enabled: bool = True

    @field_validator("dim")
    @classmethod
    def _dim_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("dim must be >= 1")
        return value


# Backward-compatible alias; kept for one release so `losses`/`calibration`
# modules that import `TaskSpec` keep working unchanged.
TaskSpec = AxisSpec


# ---------------------------------------------------------------------------
# Main model config
# ---------------------------------------------------------------------------


class PharmagenConfig(BaseModel):
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
    # Passthrough cap on co-medications sampled per patient; not consumed by
    # the model yet -- reserved for the dataset-side neighbour sampling once
    # `DoubleTowerDataset._build_poly_drug_data` is model-ready (see Task D4
    # scope note in the task brief).
    poly_max_neighbors: int = 8

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
    use_polypharmacy: bool = False
    use_cross_attention: bool = False

    # --- Axes (prediction heads) ---
    axes: dict[str, AxisSpec] = Field(default_factory=dict)

    # --- Factorized-axes composition (see heads/axis_heads.py + heads/compose.py) ---
    label_out_dim: int = 128
    use_compositional_output: bool = True

    @field_validator("axes")
    @classmethod
    def _axes_not_empty(cls, value: dict[str, AxisSpec]) -> dict[str, AxisSpec]:
        if not value:
            raise ValueError(
                "axes must not be empty: the model needs at least one head."
            )
        return value


__all__ = ["ConvType", "TaskKind", "TaskSpec", "AxisSpec", "PharmagenConfig"]
