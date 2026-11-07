"""Pharmacophore branch of the drug tower.

Reads Modephere's pharmacophore graph of each molecule (``pharm_x`` feature
points, ``pharm_edge_attr`` topological point-pair features, ``pharm_global``)
plus the 3D positions of the points in one conformer (``pharm_pos``).

Every pair of points of a molecule exchanges messages (K is 5-30), SchNet
style: a continuous filter computed from a radial-basis expansion of the
pair's distance (zeroed when the molecule has no conformer, ``has_3d = 0``)
together with the pair's 2D edge features where the pharmacophore graph has
one. Distances only, so the embedding is invariant to rotations and
translations. A molecule without any pharmacophore point (K = 0) gets a
learned ``empty`` vector.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


def _pairs(batch: Tensor, num_molecules: int) -> tuple[Tensor, Tensor]:
    """Every ordered pair ``(i, j), i != j`` of points in the same molecule.

    ``batch`` (point -> molecule) must be sorted, as PyG batching gives it.
    """
    counts = torch.bincount(batch, minlength=num_molecules)
    ptr = torch.cumsum(counts, 0) - counts
    partners = counts[batch]  # pairs (incl. self) each point starts
    src = torch.repeat_interleave(
        torch.arange(batch.numel(), device=batch.device), partners
    )
    block = torch.cumsum(partners, 0) - partners
    local = torch.arange(src.numel(), device=batch.device) - block[src]
    dst = ptr[batch[src]] + local
    keep = src != dst
    return src[keep], dst[keep]


class _RadialBasis(nn.Module):
    """Gaussian expansion of distances in ``[0, cutoff]``."""

    def __init__(self, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, num_rbf))
        self.gamma = 0.5 / (cutoff / max(num_rbf - 1, 1)) ** 2

    def forward(self, distance: Tensor) -> Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(-1) - self.centers) ** 2)


class _Interaction(nn.Module):
    """One continuous-filter convolution + residual update."""

    def __init__(self, hidden: int, edge_in: int) -> None:
        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.lin_in = nn.Linear(hidden, hidden)
        self.update = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self, h: Tensor, src: Tensor, dst: Tensor, edge: Tensor, weight: Tensor
    ) -> Tensor:
        message = self.lin_in(h)[src] * self.filter(edge) * weight.unsqueeze(-1)
        agg = scatter(message, dst, dim=0, dim_size=h.size(0), reduce="sum")
        return self.norm(h + self.update(agg))


class PharmacophoreEncoder(nn.Module):
    """Pharmacophore graph (+ conformer positions) -> one vector per molecule.

    Args:
        node_dim / edge_dim / global_dim: ``PHARMACOPHORE_GRAPH_V1`` widths
            (11 / 13 / 258).
        out_dim: Output width (the drug embedding width).
        num_layers: Interaction blocks.
        num_rbf: Radial basis functions over ``[0, cutoff]`` Å.
        cutoff: Distance where the smooth cosine cutoff reaches zero.
        hidden_dim: Width of the point features (pairs are quadratic in K, so
            this stays below ``out_dim``).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        out_dim: int,
        *,
        num_layers: int = 3,
        num_rbf: int = 32,
        cutoff: float = 15.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.edge_dim = edge_dim
        self.rbf = _RadialBasis(num_rbf, cutoff)
        self.embed = nn.Linear(node_dim, hidden_dim)
        edge_in = num_rbf + edge_dim + 1  # + "has a 2D edge" flag
        self.layers = nn.ModuleList(
            _Interaction(hidden_dim, edge_in) for _ in range(num_layers)
        )
        self.global_proj = nn.Linear(global_dim, hidden_dim) if global_dim else None
        readout_in = hidden_dim * 2 + (hidden_dim if global_dim else 0) + 1
        self.readout = nn.Sequential(
            nn.Linear(readout_in, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        #: Output of a molecule without pharmacophore points (K = 0).
        self.empty = nn.Parameter(torch.zeros(out_dim))

    def _edges(
        self,
        src: Tensor,
        dst: Tensor,
        pharm_edge_index: Tensor,
        pharm_edge_attr: Tensor,
        num_points: int,
    ) -> Tensor:
        """2D edge features on the pair list (zeros + flag 0 where none)."""
        feats = pharm_edge_attr.new_zeros(src.numel(), self.edge_dim + 1)
        if pharm_edge_index.numel() and src.numel():
            pair_key = src * num_points + dst  # ascending: src-major, then dst
            edge_key = pharm_edge_index[0] * num_points + pharm_edge_index[1]
            pos = torch.searchsorted(pair_key, edge_key).clamp(max=src.numel() - 1)
            hit = pair_key[pos] == edge_key
            feats[pos[hit], : self.edge_dim] = pharm_edge_attr[hit].to(feats.dtype)
            feats[pos[hit], self.edge_dim] = 1.0
        return feats

    def forward(
        self,
        pharm_x: Tensor,
        pharm_edge_index: Tensor,
        pharm_edge_attr: Tensor,
        pharm_pos: Tensor,
        pharm_global: Tensor,
        has_3d: Tensor,
        pharm_batch: Tensor,
        num_molecules: int,
    ) -> Tensor:
        """``[num_molecules, out_dim]``; ``pharm_batch`` maps points to molecules."""
        has_3d = has_3d.reshape(-1).to(pharm_x.dtype)
        k = pharm_x.size(0)
        h = self.embed(pharm_x)
        src, dst = _pairs(pharm_batch, num_molecules)
        flag3d = has_3d[pharm_batch[src]]
        distance = (pharm_pos[src] - pharm_pos[dst]).norm(dim=-1) * flag3d
        # smooth cutoff where 3D is known; every pair counts without it
        envelope = 0.5 * (torch.cos(distance * math.pi / self.cutoff) + 1.0)
        envelope = envelope * (distance < self.cutoff).to(envelope.dtype)
        weight = flag3d * envelope + (1.0 - flag3d)
        edge = torch.cat(
            [
                self.rbf(distance) * flag3d.unsqueeze(-1),
                self._edges(src, dst, pharm_edge_index, pharm_edge_attr, k),
            ],
            dim=-1,
        )
        for layer in self.layers:
            h = layer(h, src, dst, edge, weight)

        mean = scatter(h, pharm_batch, dim=0, dim_size=num_molecules, reduce="mean")
        maxi = h.new_zeros(num_molecules, h.size(1))
        if k:
            maxi = maxi.index_reduce(
                0, pharm_batch, h, reduce="amax", include_self=False
            )
        parts = [mean, maxi]
        if self.global_proj is not None:
            parts.append(self.global_proj(pharm_global))
        parts.append(has_3d.unsqueeze(-1))
        out = self.readout(torch.cat(parts, dim=-1))
        points = torch.bincount(pharm_batch, minlength=num_molecules)
        return torch.where((points > 0).unsqueeze(-1), out, self.empty.expand_as(out))


__all__ = ["PharmacophoreEncoder"]
