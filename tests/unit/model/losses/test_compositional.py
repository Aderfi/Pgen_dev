import torch

from src.model.losses.compositional import CompositionalLabelLoss


def test_contrastive_lower_when_aligned():
    loss = CompositionalLabelLoss(temperature=0.1)
    target = torch.randn(8, 12)
    aligned = loss(target.clone(), target)
    misaligned = loss(torch.randn(8, 12), target)
    assert aligned < misaligned
