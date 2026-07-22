import torch

from tests.unit.model.architectures.test_statedict_golden import _cfg


def test_forward_emits_axis_logits_and_z(make_batch):
    from src.model.architectures import PharmagenTwoTower

    model = PharmagenTwoTower(_cfg())
    drug_batch, geno_batch = make_batch()  # fixture: minimal PyG Batches
    out = model(drug_batch, geno_batch)
    assert "pheno" in out and "_z" in out
    assert out["_z"].shape[0] == out["pheno"].shape[0]
