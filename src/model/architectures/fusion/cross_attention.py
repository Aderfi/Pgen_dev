"""Bidirectional cross-attention fusion between drug and genotype towers."""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from ..towers.blocks import masked_pool


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

        return masked_pool(d, d_mask), masked_pool(g, g_mask), attn
