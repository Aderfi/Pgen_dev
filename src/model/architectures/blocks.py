from __future__ import annotations

import torch
from torch import Tensor, cat, nn
from torch_geometric.nn import GATv2Conv, GINEConv


def branch_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
        nn.ELU(),
    )


def masked_pool(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return cat([summed, summed / counts], dim=-1)


def make_conv(
    conv_type: str, dim: int, heads: int, dropout: float, edge_dim: int | None
) -> nn.Module:
    if conv_type == "gine":
        mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ELU(),
            nn.Linear(2 * dim, dim),
        )
        return GINEConv(nn=mlp, edge_dim=edge_dim, train_eps=True)
    if conv_type == "gatv2":
        return GATv2Conv(
            dim,
            dim // heads,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown conv_type '{conv_type}'. Expected 'gine' or 'gatv2'.")
