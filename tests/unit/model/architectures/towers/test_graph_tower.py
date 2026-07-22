import torch

from src.model.architectures.towers.graph_tower import GraphTower


def _tiny_graph(feat_dim=6):
    x = torch.randn(4, feat_dim)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    batch = torch.zeros(4, dtype=torch.long)
    return x, edge_index, batch


def test_graph_tower_gine_output_shapes():
    tower = GraphTower(6, 16, 8, conv_type="gine", num_layers=2, edge_dim=None)
    x, edge_index, batch = _tiny_graph()
    node_emb, graph_emb = tower(x, edge_index, None, batch)
    assert node_emb.shape == (4, 8)
    assert graph_emb.shape == (1, 8)


def test_graph_tower_gatv2_divisibility_guard():
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        GraphTower(6, 17, 8, conv_type="gatv2", heads=4)
