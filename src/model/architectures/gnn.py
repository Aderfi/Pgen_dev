"""
Pharmacogenomic two-tower graph neural network.

Architecture overview
---------------------

    SMILES ─► atom/bond graph ──[GINEConv x L]──► molecule embedding
                                                        │
                              (+ physchem/ECFP, ADMET) ──┤  fused per molecule
                                                        ▼
                    polypharmacy graph (nodes = drugs, edges = DDI)
                                   ──[GATv2Conv x L]──► drug node set
                                                        │
    genotype graph (nodes = variants/genes) ──[GATv2Conv x L]──► gene node set
                                                        │
                              cross-attention  drug ⇄ gene
                                                        ▼
                    [d, g, d*g, |d-g|] ─► interaction MLP ─► multi-task heads

Design notes that motivate the structure
----------------------------------------
* GINE for molecules: sum aggregation + MLP update reaches the WL-1 expressivity
  bound and consumes categorical bond features natively.
* GATv2 for genotype / polypharmacy: those graphs have noisy, heterogeneous and
  partly spurious edges, so learned per-edge weighting is worth its cost.
* Sum-based readout (concatenated with mean and max) preserves node cardinality,
  which mean pooling destroys — molecule size is a real pharmacokinetic signal.
* Explicit multiplicative interaction terms: pharmacogenomics is conditional by
  nature (a CYP genotype only matters for its substrates), and MLPs learn
  multiplicative interactions poorly from plain concatenation.
* Cross-attention yields a drug x gene attention matrix, which is the only
  clinically defensible explanation this model can produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor, cat, nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_dense_batch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data


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
# Shared building blocks
# ---------------------------------------------------------------------------


def _branch_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Project an auxiliary descriptor vector into the embedding space."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
    )


def _masked_pool(x: Tensor, mask: Tensor) -> Tensor:
    """Sum + mean pooling over a padded ``[B, L, D]`` tensor, returning ``[B, 2D]``."""
    m = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
    summed = (x * m).sum(dim=1)  # [B, D]
    counts = m.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return cat([summed, summed / counts], dim=-1)


class GraphTower(nn.Module):
    """Generic message-passing tower, parameterised by convolution type.

    A single input projection puts every layer at the same width, so all
    residual skips are identity connections and there are no dimension
    bookkeeping traps when switching between GINE and GATv2.

    Uses pre-normalisation (norm -> conv -> residual add), which stays stable if
    the tower is ever deepened beyond 5-6 layers.

    Returns both node-level and graph-level embeddings: the node-level output is
    what cross-attention consumes, the graph-level output is the pooled residual.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        conv_type: ConvType = "gine",
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int | None = None,
    ) -> None:
        super().__init__()

        if conv_type not in ("gine", "gatv2"):
            raise ValueError(
                f"Unknown conv_type '{conv_type}'. Expected 'gine' or 'gatv2'."
            )
        if conv_type == "gatv2" and hidden_channels % heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"heads ({heads}) when conv_type='gatv2'."
            )

        self.conv_type: ConvType = conv_type
        self.dropout: float = dropout
        self.edge_dim: int | None = edge_dim
        dim = hidden_channels

        self.input_proj = nn.Linear(in_channels, dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gine":
                # The update function must be a universal approximator: this MLP
                # is what makes the GIN aggregation injective. A bare Linear
                # would silently forfeit the WL-1 expressivity guarantee.
                mlp = nn.Sequential(
                    nn.Linear(dim, 2 * dim),
                    nn.BatchNorm1d(2 * dim),
                    nn.ELU(),
                    nn.Linear(2 * dim, dim),
                )
                conv: nn.Module = GINEConv(
                    nn=mlp,
                    edge_dim=edge_dim,
                    train_eps=True,  # let the model weight self vs. neighbour sum
                )
            else:
                conv = GATv2Conv(
                    dim,
                    dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,  # DropAttention on the attention coefficients
                )
            self.convs.append(conv)
            self.norms.append(GraphNorm(dim))

        # Node-level projection, consumed by the cross-attention module.
        self.node_proj = nn.Linear(dim, out_channels)

        # Triple readout: add preserves cardinality (molecule size), mean gives a
        # size-invariant view, max captures dominant substructures.
        self.post_pool_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_channels),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:          node features          [num_nodes, in_channels]
            edge_index: graph connectivity     [2, num_edges]
            edge_attr:  optional edge features [num_edges, edge_dim]
            batch:      node -> graph mapping  [num_nodes]

        Returns:
            node_emb:  [num_nodes, out_channels]
            graph_emb: [num_graphs, out_channels]
        """
        if self.edge_dim is not None and edge_attr is None:
            raise ValueError(
                f"This tower was built with edge_dim={self.edge_dim} but "
                "edge_attr is None. Either supply edge features or rebuild the "
                "tower with edge_dim=None."
            )
        if self.edge_dim is None:
            edge_attr = None

        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h = norm(x, batch)
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # identity residual: widths match by construction

        graph_emb = cat(
            [
                global_add_pool(x, batch),
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )
        return self.node_proj(x), self.post_pool_mlp(graph_emb)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between the drug set and the gene set.

    Replaces plain concatenation of two pooled vectors. Concatenation forces the
    downstream MLP to rediscover which gene matters for which drug from data;
    cross-attention makes that correspondence a first-class, inspectable object.

    The returned attention matrix is [B, n_drugs, n_genes] and is the model's
    clinical explanation surface.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.drug_to_gene = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.gene_to_drug = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_drug = nn.LayerNorm(dim)
        self.norm_gene = nn.LayerNorm(dim)

    def forward(
        self,
        drug_nodes: Tensor,
        drug_index: Tensor,
        gene_nodes: Tensor,
        gene_index: Tensor,
        num_patients: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            drug_nodes: [num_drugs_in_batch, dim]
            drug_index: drug -> patient mapping [num_drugs_in_batch]
            gene_nodes: [num_genes_in_batch, dim]
            gene_index: gene -> patient mapping [num_genes_in_batch]

        Returns:
            drug_pooled: [B, 2*dim]
            gene_pooled: [B, 2*dim]
            attn:        [B, L_drug, L_gene] attention weights (head-averaged)
        """
        d, d_mask = to_dense_batch(drug_nodes, drug_index, batch_size=num_patients)
        g, g_mask = to_dense_batch(gene_nodes, gene_index, batch_size=num_patients)

        # A fully padded row would make softmax produce NaN. Patients with no
        # drugs or no genotyped genes should be filtered upstream, but keep the
        # forward pass numerically safe regardless.
        d_key_pad = ~d_mask
        g_key_pad = ~g_mask
        d_key_pad[d_key_pad.all(dim=1), 0] = False
        g_key_pad[g_key_pad.all(dim=1), 0] = False

        d_att, attn = self.drug_to_gene(
            query=d, key=g, value=g, key_padding_mask=g_key_pad, need_weights=True
        )
        g_att, _ = self.gene_to_drug(
            query=g, key=d, value=d, key_padding_mask=d_key_pad, need_weights=False
        )

        d = self.norm_drug(d + d_att)
        g = self.norm_gene(g + g_att)

        return _masked_pool(d, d_mask), _masked_pool(g, g_mask), attn


# ---------------------------------------------------------------------------
# Main model
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


class PharmagenTwoTower(nn.Module):
    """Two-tower pharmacogenomic outcome model.

    Expected inputs
    ---------------
    ``drug_data`` (PyG ``Batch`` of molecular graphs):
        x, edge_index, edge_attr, batch  -- atoms -> molecules
        mol_to_patient : [num_molecules]  molecule -> patient index.
            If absent, one molecule per patient is assumed.
        ddi_edge_index : [2, E]  edges over *molecule* indices (batch-global).
            If absent, molecules of the same patient are fully connected.
        ddi_edge_attr  : [E, ddi_edge_dim]  optional interaction-type features.
        global_feats   : [num_molecules, drug_global_dim]
        admet_feats    : [num_molecules, drug_admet_dim]

    ``geno_data`` (PyG ``Batch`` of genotype graphs, one per patient):
        x, edge_index, edge_attr, batch  -- variants/genes -> patients
        geno_function : [num_patients, geno_global_dim]
    """

    def __init__(self, config: PharmagenConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config

        if not cfg.targets:
            raise ValueError("config.targets is empty: the model has no heads.")

        dim = cfg.embedding_dim

        # --- Drug tower: GINE ---------------------------------------------
        # Molecular graphs are small, edges are categorical and meaningful, and
        # WL-1 expressivity matters for distinguishing scaffolds. GINE wins here;
        # GATv2's softmax would reintroduce mean-like aggregation.
        if cfg.use_mol_gnn:
            self.drug_tower = GraphTower(
                in_channels=cfg.drug_in_features,
                hidden_channels=cfg.drug_hidden_dim,
                out_channels=dim,
                conv_type="gine",
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                edge_dim=cfg.drug_edge_dim,
            )

        # --- Auxiliary drug branches (fused at MOLECULE level) -------------
        # Fusing before the polypharmacy graph means the DDI layer reasons over
        # chemically complete drug representations, not topology-only ones.
        if cfg.drug_global_dim > 0:
            self.drug_global_mlp = _branch_mlp(cfg.drug_global_dim, dim, cfg.dropout)
        if cfg.drug_admet_dim > 0:
            self.drug_admet_mlp = _branch_mlp(cfg.drug_admet_dim, dim, cfg.dropout)

        n_drug_branches = (
            int(cfg.use_mol_gnn)
            + int(cfg.drug_global_dim > 0)
            + int(cfg.drug_admet_dim > 0)
        )
        if n_drug_branches == 0:
            raise ValueError(
                "The drug tower has no active branch: enable use_mol_gnn or "
                "provide drug_global_dim / drug_admet_dim."
            )
        self.drug_fuse = (
            nn.Linear(dim * n_drug_branches, dim)
            if n_drug_branches > 1
            else nn.Identity()
        )

        # --- Polypharmacy tower: GATv2 -------------------------------------
        # Nodes are the patient's drugs; edges are known DDIs. This is where
        # combination risk (e.g. clopidogrel + omeprazole in a CYP2C19 IM) can
        # be represented at all -- a per-drug model cannot express it.
        if cfg.use_polypharmacy:
            self.poly_tower = GraphTower(
                in_channels=dim,
                hidden_channels=dim,
                out_channels=dim,
                conv_type="gatv2",
                num_layers=cfg.polypharmacy_layers,
                heads=cfg.heads,
                dropout=cfg.dropout,
                edge_dim=cfg.ddi_edge_dim,
            )

        # --- Genotype tower: GATv2 -----------------------------------------
        # Gene/variant graphs carry noisy, partly spurious edges (pathway and
        # interaction databases are far from clean), so learned edge weighting
        # is worth its cost here.
        self.geno_tower = GraphTower(
            in_channels=cfg.geno_in_features,
            hidden_channels=cfg.geno_hidden_dim,
            out_channels=dim,
            conv_type="gatv2",
            num_layers=cfg.num_layers,
            heads=cfg.heads,
            dropout=cfg.dropout,
            edge_dim=cfg.geno_edge_dim,
        )

        if cfg.geno_global_dim > 0:
            self.geno_global_mlp = _branch_mlp(cfg.geno_global_dim, dim, cfg.dropout)
            self.geno_fuse = nn.Linear(dim * 2, dim)

        # --- Cross-attention -----------------------------------------------
        if cfg.use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                dim, heads=cfg.heads, dropout=cfg.dropout
            )
            # _masked_pool returns 2*dim per side.
            self.drug_side_proj = nn.Linear(dim * 2, dim)
            self.gene_side_proj = nn.Linear(dim * 2, dim)

        # --- Interaction & heads -------------------------------------------
        # [d, g, d*g, |d-g|]: the explicit product term encodes the conditional
        # nature of pharmacogenomics (a CYP genotype only matters for its
        # substrates), which a plain concatenation forces the MLP to learn from
        # scratch and typically learns badly.
        combined_dim = dim * 4

        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
        )
        head_in = combined_dim // 2

        self.heads = nn.ModuleDict()
        for name, spec in cfg.targets.items():
            if not spec.enabled:
                continue
            self.heads[name] = nn.Sequential(
                nn.Linear(head_in, head_in // 2),
                nn.ELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(head_in // 2, spec.dim),
            )

        self._init_own_weights()

    # -- initialisation ----------------------------------------------------

    def _init_own_weights(self) -> None:
        """Initialise only our own Linear layers.

        A blanket ``self.apply(...)`` would overwrite the Glorot initialisation
        that PyG applies inside GATv2Conv / GINEConv, which was chosen
        deliberately by those authors. Kaiming is also better matched to the
        ELU/ReLU activations used here than Xavier, which assumes tanh-like
        symmetric activations.
        """
        skip = tuple(
            m
            for m in self.modules()
            if isinstance(m, (GATv2Conv, GINEConv, nn.MultiheadAttention))
        )
        skipped_params = {id(p) for m in skip for p in m.parameters()}

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if id(module.weight) in skipped_params:
                continue
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    # -- input validation --------------------------------------------------

    @staticmethod
    def _require_attr(
        data: Data,
        attr: str,
        expected_rows: int,
        expected_dim: int,
        config_field: str,
        data_name: str,
    ) -> Tensor:
        """Fetch a required graph-level attribute and validate its shape.

        This is the single most common silent bug in PyG models: a graph-level
        feature stored as a 1-D ``[D]`` tensor gets concatenated along dim 0 at
        collate time, producing ``[B*D]`` instead of ``[B, D]``. Downstream
        Linear layers may not error if the sizes happen to line up, and the
        model then trains on garbage. Always store these as ``[1, D]``.
        """
        value = getattr(data, attr, None)
        if value is None:
            raise ValueError(
                f"{data_name} is missing '{attr}' but the model was configured "
                f"with {config_field}={expected_dim}."
            )
        if value.dim() != 2 or tuple(value.shape) != (expected_rows, expected_dim):
            raise ValueError(
                f"{data_name}.{attr} must have shape "
                f"[{expected_rows}, {expected_dim}], got {tuple(value.shape)}. "
                f"Store it as a [1, {expected_dim}] tensor on each Data object "
                "so PyG collates it into a [B, D] matrix."
            )
        return value

    # -- polypharmacy graph construction ----------------------------------

    @staticmethod
    def _complete_intra_patient_edges(mol_to_patient: Tensor) -> Tensor:
        """Fully connect the molecules belonging to the same patient.

        Fallback when no curated DDI edge list is supplied. With <= ~20 drugs per
        patient the quadratic cost is negligible, and it lets the attention layer
        decide which pairs matter instead of leaving the graph disconnected.
        """
        idx = torch.arange(mol_to_patient.numel(), device=mol_to_patient.device)
        same = mol_to_patient.unsqueeze(0) == mol_to_patient.unsqueeze(1)
        same &= idx.unsqueeze(0) != idx.unsqueeze(1)  # drop self-loops
        src, dst = same.nonzero(as_tuple=True)
        return torch.stack([src, dst], dim=0)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        drug_data: Data,
        geno_data: Data,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        """Multi-task forward pass.

        Returns a dict of task-name -> logits/predictions. When
        ``return_attention`` is set, the drug x gene attention matrix is added
        under the key ``"_attention"`` for interpretability and audit.
        """
        cfg = self.config

        for name, data in (("drug_data", drug_data), ("geno_data", geno_data)):
            if (
                getattr(data, "x", None) is None
                or getattr(data, "edge_index", None) is None
            ):
                raise ValueError(
                    f"{name} is missing required attributes (x, edge_index)"
                )
            if getattr(data, "batch", None) is None:
                raise ValueError(
                    f"{name}.batch is None: pass a PyG Batch, not a single Data."
                )

        num_patients = int(geno_data.batch.max().item()) + 1

        # ---------------- Drug side: molecule level ----------------
        mol_to_patient: Tensor | None = getattr(drug_data, "mol_to_patient", None)
        num_molecules = int(drug_data.batch.max().item()) + 1
        if mol_to_patient is None:
            if num_molecules != num_patients:
                raise ValueError(
                    "drug_data.mol_to_patient is required when the number of "
                    f"molecules ({num_molecules}) differs from the number of "
                    f"patients ({num_patients})."
                )
            mol_to_patient = torch.arange(num_molecules, device=drug_data.x.device)
        mol_to_patient = mol_to_patient.long()

        branches: list[Tensor] = []
        if cfg.use_mol_gnn:
            _, mol_emb = self.drug_tower(
                x=drug_data.x,
                edge_index=drug_data.edge_index,
                edge_attr=getattr(drug_data, "edge_attr", None),
                batch=drug_data.batch,
            )
            branches.append(mol_emb)

        if cfg.drug_global_dim > 0:
            branches.append(
                self.drug_global_mlp(
                    self._require_attr(
                        drug_data,
                        "global_feats",
                        num_molecules,
                        cfg.drug_global_dim,
                        "drug_global_dim",
                        "drug_data",
                    )
                )
            )
        if cfg.drug_admet_dim > 0:
            branches.append(
                self.drug_admet_mlp(
                    self._require_attr(
                        drug_data,
                        "admet_feats",
                        num_molecules,
                        cfg.drug_admet_dim,
                        "drug_admet_dim",
                        "drug_data",
                    )
                )
            )

        drug_nodes = self.drug_fuse(
            cat(branches, dim=1) if len(branches) > 1 else branches[0]
        )

        # ---------------- Drug side: polypharmacy level ----------------
        if cfg.use_polypharmacy:
            ddi_edge_index = getattr(drug_data, "ddi_edge_index", None)
            ddi_edge_attr = getattr(drug_data, "ddi_edge_attr", None)
            if ddi_edge_index is None:
                if cfg.ddi_edge_dim is not None:
                    raise ValueError(
                        "config.ddi_edge_dim is set but drug_data.ddi_edge_index "
                        "is missing; cannot fall back to a complete graph with "
                        "typed edges."
                    )
                ddi_edge_index = self._complete_intra_patient_edges(mol_to_patient)
                ddi_edge_attr = None
            drug_nodes, drug_graph_emb = self.poly_tower(
                x=drug_nodes,
                edge_index=ddi_edge_index,
                edge_attr=ddi_edge_attr,
                batch=mol_to_patient,
            )
        else:
            drug_graph_emb = global_add_pool(
                drug_nodes, mol_to_patient, size=num_patients
            )

        # ---------------- Genotype side ----------------
        gene_nodes, geno_emb = self.geno_tower(
            x=geno_data.x,
            edge_index=geno_data.edge_index,
            edge_attr=getattr(geno_data, "edge_attr", None),
            batch=geno_data.batch,
        )

        if cfg.geno_global_dim > 0:
            geno_function = self._require_attr(
                geno_data,
                "geno_function",
                num_patients,
                cfg.geno_global_dim,
                "geno_global_dim",
                "geno_data",
            )
            geno_emb = self.geno_fuse(
                cat([geno_emb, self.geno_global_mlp(geno_function)], dim=1)
            )

        # ---------------- Fusion ----------------
        attention: Tensor | None = None
        if cfg.use_cross_attention:
            drug_pooled, gene_pooled, attention = self.cross_attention(
                drug_nodes=drug_nodes,
                drug_index=mol_to_patient,
                gene_nodes=gene_nodes,
                gene_index=geno_data.batch,
                num_patients=num_patients,
            )
            d = self.drug_side_proj(drug_pooled) + drug_graph_emb
            g = self.gene_side_proj(gene_pooled) + geno_emb
        else:
            d, g = drug_graph_emb, geno_emb

        combined = cat([d, g, d * g, (d - g).abs()], dim=1)
        interacted = self.interaction_mlp(combined)

        outputs: dict[str, Tensor] = {
            name: head(interacted) for name, head in self.heads.items()
        }
        if return_attention and attention is not None:
            outputs["_attention"] = attention
        return outputs

    def count_parameters(self) -> dict[str, int]:
        """Parameter counts per top-level submodule, for ablation reporting."""
        counts = {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in self.named_children()
        }
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    pos_weight: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Focal binary cross-entropy.

    Adverse pharmacogenomic events are rare (often <1%). Plain BCE converges to
    predicting the majority class with a flattering AUROC that means nothing;
    focal loss down-weights the easy negatives that dominate the gradient.
    Always evaluate these tasks with AUPRC, not AUROC.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        bce = bce * (1.0 - p_t).pow(gamma)
    if reduction == "mean":
        return bce.mean()
    if reduction == "sum":
        return bce.sum()
    return bce


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learned homoscedastic uncertainty weighting.

    Summing raw task losses lets whichever task has the largest natural scale
    dominate the gradient. This implements Kendall et al.'s uncertainty
    weighting: each task gets a learned log-variance, so the optimiser balances
    the tasks instead of the practitioner guessing weights.

    Targets are supplied as a dict of task-name -> tensor. Missing tasks are
    skipped, and NaN entries are masked out, so partially labelled cohorts work
    without building separate models.
    """

    def __init__(self, targets: dict[str, TaskSpec]) -> None:
        super().__init__()
        self.specs = {n: s for n, s in targets.items() if s.enabled}
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in self.specs}
        )
        for name, spec in self.specs.items():
            if spec.kind == "binary" and spec.pos_weight is not None:
                self.register_buffer(
                    f"pos_weight_{name}", torch.tensor(float(spec.pos_weight))
                )
            if spec.kind == "multiclass" and spec.class_weights is not None:
                self.register_buffer(
                    f"class_weight_{name}", torch.tensor(list(spec.class_weights))
                )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        total = torch.zeros((), device=next(iter(outputs.values())).device)
        per_task: dict[str, Tensor] = {}

        for name, spec in self.specs.items():
            if name not in outputs or name not in targets:
                continue
            logits, y = outputs[name], targets[name]

            if spec.kind in ("binary", "regression"):
                logits = logits.view(-1, spec.dim)
                y = y.view(-1, spec.dim).to(logits.dtype)
                mask = ~torch.isnan(y)
                if not mask.any():
                    continue
                logits, y = logits[mask], y[mask]
            else:
                y = y.view(-1).long()
                mask = y >= 0  # convention: -1 marks an unlabelled sample
                if not mask.any():
                    continue
                logits, y = logits[mask], y[mask]

            if spec.kind == "binary":
                pw = getattr(self, f"pos_weight_{name}", None)
                loss = focal_bce_with_logits(
                    logits, y, gamma=spec.focal_gamma, pos_weight=pw
                )
                scale = 1.0
            elif spec.kind == "multiclass":
                cw = getattr(self, f"class_weight_{name}", None)
                loss = F.cross_entropy(logits, y, weight=cw)
                scale = 1.0
            else:
                loss = F.smooth_l1_loss(logits, y)
                scale = 0.5  # Kendall's Gaussian-likelihood factor

            log_var = self.log_vars[name]
            total = total + scale * torch.exp(-log_var) * loss + 0.5 * log_var
            per_task[name] = loss.detach()

        return total, per_task


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


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


__all__ = [
    "TaskSpec",
    "PharmagenConfig",
    "GraphTower",
    "CrossAttentionFusion",
    "PharmagenTwoTower",
    "MultiTaskLoss",
    "TemperatureScaler",
    "focal_bce_with_logits",
]
