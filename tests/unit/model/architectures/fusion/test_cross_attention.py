import torch

from src.model.architectures.fusion.cross_attention import CrossAttentionFusion


def test_cross_attention_pooled_and_attn_shapes():
    fusion = CrossAttentionFusion(dim=8, heads=2, dropout=0.0)
    drug_nodes = torch.randn(3, 8)
    drug_index = torch.tensor([0, 0, 1])
    gene_nodes = torch.randn(2, 8)
    gene_index = torch.tensor([0, 1])
    d, g, attn = fusion(drug_nodes, drug_index, gene_nodes, gene_index, num_patients=2)
    assert d.shape == (2, 16) and g.shape == (2, 16)
    assert attn.shape[0] == 2  # [B, L_drug, L_gene]
