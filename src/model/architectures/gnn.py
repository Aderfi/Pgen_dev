from torch import Tensor, cat, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.norm import GraphNorm


class GATv2Tower(nn.Module):
    """GATv2-based graph encoding tower."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int | None = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_layers: int = num_layers
        self.pooling: str = pooling
        self.dropout: float = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()  # graph normalization layers
        self.skips = nn.ModuleList()

        curr_in = in_channels
        for i in range(num_layers):
            out_dim = hidden_channels * heads
            self.convs.append(
                GATv2Conv(
                    curr_in,
                    hidden_channels,
                    heads=heads,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.norms.append(GraphNorm(out_dim))
            # Identity skip when dims already match — saves a Linear layer of params.
            if curr_in != out_dim:
                self.skips.append(nn.Linear(curr_in, out_dim))
            else:
                self.skips.append(nn.Identity())

            curr_in = out_dim

        self.post_pool_mlp = nn.Sequential(
            nn.Linear(curr_in, curr_in),
            nn.ELU(),
            nn.Linear(curr_in, out_channels),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        batch: Tensor,
    ) -> Tensor:
        """
        Args:
            x:          node features          [num_nodes, num_features]
            edge_index: graph connectivity     [2, num_edges]
            edge_attr:  optional edge features [num_edges, edge_dim]
            batch:      node→graph mapping     [num_nodes]
        """
        for conv, skip, norm in zip(self.convs, self.skips, self.norms):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = x + skip(x_in)
            x = norm(x, batch)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            x_graph = global_mean_pool(x, batch)
        elif self.pooling == "add":
            x_graph = global_add_pool(x, batch)
        elif self.pooling == "max":
            x_graph = global_max_pool(x, batch)
        else:
            raise ValueError(
                f"Unknown pooling strategy '{self.pooling}'. Expected 'mean', 'add', or 'max'."
            )

        return self.post_pool_mlp(x_graph)


class PharmagenTwoTower(nn.Module):
    def __init__(
        self,
        # Drug tower configuration
        drug_in_features: int,
        drug_edge_dim: int,
        drug_hidden_dim: int,
        # Genotype tower configuration
        geno_in_features: int,
        geno_edge_dim: int,
        geno_hidden_dim: int,
        # Global / shared
        embedding_dim: int,
        target_dims: dict[str, int],
        # Hyperparameters
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        # Per-molecule global descriptor vector (QSAR physchem + ECFP); 0 disables.
        drug_global_dim: int = 0,
    ):
        super().__init__()
        self.target_dims: dict[str, int] = target_dims
        self.drug_global_dim = drug_global_dim

        # --- Drug Tower ---
        self.drug_tower = GATv2Tower(
            in_channels=drug_in_features,
            hidden_channels=drug_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=drug_edge_dim,
            dropout=dropout,
            pooling="mean",
        )

        # --- Drug global-descriptor branch (fused into the drug embedding) ---
        if drug_global_dim > 0:
            self.drug_global_mlp = nn.Sequential(
                nn.Linear(drug_global_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, embedding_dim),
                nn.ELU(),
            )
            # Fuse graph embedding + descriptor embedding back to embedding_dim,
            # so the downstream interaction/heads are unchanged.
            self.drug_fuse = nn.Linear(embedding_dim * 2, embedding_dim)

        # --- Genotype Tower ---
        self.geno_tower: GATv2Tower = GATv2Tower(
            in_channels=geno_in_features,
            hidden_channels=geno_hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            heads=heads,
            edge_dim=geno_edge_dim,
            dropout=dropout,
            pooling="add",
        )

        # --- Interaction & Prediction Heads ---
        combined_dim = embedding_dim * 2

        self.interaction_mlp = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.LayerNorm(combined_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, combined_dim),
            nn.ELU(),
        )

        self.heads = nn.ModuleDict()
        for target_name, out_dim in target_dims.items():
            self.heads[target_name] = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ELU(),
                nn.Linear(combined_dim // 2, out_dim),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, drug_data: Data, geno_data: Data) -> dict[str, Tensor]:
        """
        Dynamic multi-task forward pass.

        Args:
            drug_data: Batch of molecular graphs (PyG Batch).
            geno_data: Batch of genotype graphs (PyG Batch).
        """
        if not (hasattr(drug_data, "x") and hasattr(drug_data, "edge_index")):
            raise ValueError(
                "drug_data is missing required graph attributes (x, edge_index)"
            )
        if not (hasattr(geno_data, "x") and hasattr(geno_data, "edge_index")):
            raise ValueError(
                "geno_data is missing required graph attributes (x, edge_index)"
            )

        drug_emb = self.drug_tower(
            x=drug_data.x,
            edge_index=drug_data.edge_index,
            edge_attr=getattr(drug_data, "edge_attr", None),
            batch=drug_data.batch,
        )

        # Fuse the per-molecule global descriptor vector into the drug embedding.
        if self.drug_global_dim > 0:
            global_feats = getattr(drug_data, "global_feats", None)
            if global_feats is None:
                raise ValueError(
                    "drug_data is missing 'global_feats' but the model was built "
                    f"with drug_global_dim={self.drug_global_dim}"
                )
            global_emb = self.drug_global_mlp(global_feats)
            drug_emb = self.drug_fuse(cat([drug_emb, global_emb], dim=1))

        geno_emb = self.geno_tower(
            x=geno_data.x,
            edge_index=geno_data.edge_index,
            edge_attr=getattr(geno_data, "edge_attr", None),
            batch=geno_data.batch,
        )

        combined = cat([drug_emb, geno_emb], dim=1)
        interacted = self.interaction_mlp(combined)

        outputs = {}
        for target_name, head_layer in self.heads.items():
            outputs[target_name] = head_layer(interacted)

        return outputs
