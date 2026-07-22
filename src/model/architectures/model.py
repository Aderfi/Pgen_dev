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

from typing import TYPE_CHECKING

import torch
from torch import Tensor, cat, nn
from torch_geometric.nn import (
    GATv2Conv,
    GINEConv,
    global_add_pool,
)

from .fusion.cross_attention import CrossAttentionFusion
from .towers.blocks import branch_mlp
from .towers.graph_tower import GraphTower

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from .config import PharmagenConfig


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


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
            self.drug_global_mlp = branch_mlp(cfg.drug_global_dim, dim, cfg.dropout)
        if cfg.drug_admet_dim > 0:
            self.drug_admet_mlp = branch_mlp(cfg.drug_admet_dim, dim, cfg.dropout)

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
            self.geno_global_mlp = branch_mlp(cfg.geno_global_dim, dim, cfg.dropout)
            self.geno_fuse = nn.Linear(dim * 2, dim)

        # --- Cross-attention -----------------------------------------------
        if cfg.use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                dim, heads=cfg.heads, dropout=cfg.dropout
            )
            # masked_pool returns 2*dim per side.
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


__all__ = ["PharmagenTwoTower"]
