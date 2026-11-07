"""DrugSetPooling: molecules -> entities (mean) -> rows (attention)."""

from __future__ import annotations

import torch

from src.model.architectures.drug_set import DrugSetPooling


def _inputs():
    torch.manual_seed(0)
    mol = torch.randn(5, 8)
    mol_to_entity = torch.tensor([0, 1, 1, 1, 2])
    entity_kind = torch.tensor([0, 1, 0])
    entity_to_row = torch.tensor([0, 0, 1])
    return mol, mol_to_entity, entity_kind, entity_to_row


def test_drug_set_pooling_means_class_members():
    pool = DrugSetPooling(8)
    with torch.no_grad():
        pool.kind_embedding.weight.zero_()
    mol, m2e, kind, e2r = _inputs()
    entities, rows = pool(mol, m2e, kind, e2r, num_rows=2)
    assert entities.shape == (3, 8) and rows.shape == (2, 8)
    assert torch.allclose(entities[1], mol[1:4].mean(0))
    assert torch.allclose(entities[0], mol[0])


def test_kind_embedding_tells_classes_from_drugs():
    pool = DrugSetPooling(8)
    mol = torch.ones(2, 8)
    entities, _ = pool(
        mol, torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1]), 2
    )
    assert not torch.allclose(entities[0], entities[1])


def test_single_entity_row_is_that_entity():
    pool = DrugSetPooling(8)
    mol, m2e, kind, e2r = _inputs()
    entities, rows = pool(mol, m2e, kind, e2r, num_rows=2)
    assert torch.allclose(rows[1], entities[2], atol=1e-6)


def test_rows_pool_their_entities_only():
    pool = DrugSetPooling(8)
    mol, m2e, kind, e2r = _inputs()
    _, rows_a = pool(mol, m2e, kind, e2r, num_rows=2)
    mol_b = mol.clone()
    mol_b[4] += 5.0  # molecule of row 1
    _, rows_b = pool(mol_b, m2e, kind, e2r, num_rows=2)
    assert torch.allclose(rows_a[0], rows_b[0])
    assert not torch.allclose(rows_a[1], rows_b[1])
