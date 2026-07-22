from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, cat, nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import GraphNorm

from .blocks import make_conv

ConvType = Literal["gine", "gatv2"]


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
            conv = make_conv(conv_type, dim, heads, dropout, edge_dim)
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
