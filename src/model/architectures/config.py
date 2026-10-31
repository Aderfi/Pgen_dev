"""Configuration models for :class:`PharmagenTwoTower`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from src.config.models import TrainingParams
    from src.data.cache import GraphDims

ConvType = Literal["gine", "gatv2"]
TaskKind = Literal["binary", "multiclass", "regression", "ordinal"]


# ---------------------------------------------------------------------------
# Axis specification
# ---------------------------------------------------------------------------


class AxisSpec(BaseModel):
    """Declarative description of one prediction head.

    Args:
        name: Identifier for the axis/head.
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

    # --- Genotype tower (HGVS variant graph, src/data/genotype) ---
    geno_in_features: int = 0
    geno_edge_dim: int | None = None
    geno_hidden_dim: int = 256
    # Rows of the gene-identity embedding added to gene hub nodes (0 disables).
    gene_vocab_size: int = 0

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

    # --- Factorized-axes composition (see axis_heads.py + compose.py) ---
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

    @classmethod
    def from_dims(
        cls,
        *,
        dims: GraphDims,
        drug_dim: int,
        geno_dim: int,
        axes: dict[str, AxisSpec],
        params: TrainingParams,
        switches: dict[str, bool] | None = None,
        gene_vocab_size: int = 0,
    ) -> PharmagenConfig:
        """Assemble a config from dataset-inferred dims + fitted axis specs.

        `drug_dim`, `geno_dim`, and `axes` are only known once real graphs and
        fitted target encoders exist, so they can't live in a static
        TOML-backed model the way `params` does — this classmethod is where
        they and `params` (architecture hyperparameters) meet.

        Args:
            dims: Graph feature widths (edge dims + auxiliary descriptor
                widths; see :class:`~src.data.cache.GraphDims`).
            drug_dim: Drug-tower node feature width, inferred from a real graph.
            geno_dim: Genotype-tower node feature width, inferred from a real
                graph.
            axes: One :class:`AxisSpec` per prediction head (see
                ``src.model.architectures.assembly.infer_axis_specs``).
            params: Architecture hyperparameters (embedding_dim, hidden_dim,
                n_layers, heads, dropout_rate).
            switches: Structural ablation flags (``use_polypharmacy``,
                ``use_cross_attention``). Absent keys default to ``False``.
            gene_vocab_size: Gene-embedding rows (see
                ``src.data.genotype.gene_vocab_size``); 0 disables it.
        """
        switches = switches or {}
        return cls(
            drug_in_features=drug_dim,
            drug_edge_dim=dims.drug_edges or None,
            drug_hidden_dim=params.hidden_dim,
            drug_global_dim=dims.drug_global,
            drug_admet_dim=dims.drug_admet,
            geno_in_features=geno_dim,
            geno_edge_dim=dims.geno_edges or None,
            geno_hidden_dim=params.hidden_dim,
            gene_vocab_size=gene_vocab_size,
            # The geno tower's aux "global" branch is fed the graph-level PGx
            # function vector (``geno_function``).
            geno_global_dim=dims.geno_function,
            embedding_dim=params.embedding_dim,
            num_layers=params.n_layers,
            heads=params.heads,
            dropout=params.dropout_rate,
            use_polypharmacy=switches.get("use_polypharmacy", False),
            use_cross_attention=switches.get("use_cross_attention", False),
            axes=axes,
        )


__all__ = ["ConvType", "TaskKind", "AxisSpec", "PharmagenConfig"]
