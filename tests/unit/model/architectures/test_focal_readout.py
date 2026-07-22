"""Focal-anchored polypharmacy readout: synthetic-batch unit tests.

These tests deliberately hand-build minimal PyG ``Batch`` objects instead of
going through ``DoubleTowerDataset._build_poly_drug_data`` — that dataset path
is not model-ready yet (it packs molecule descriptors into ``x`` instead of
atom-level features + ``global_feats``, and the full atom -> molecule ->
patient batching needs real DDI data). The model-level contract (``x``,
``edge_index``, ``batch``, ``mol_to_patient``, ``ddi_edge_index``,
``is_focal``) is exercised directly here.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from src.model.architectures import PharmagenConfig, PharmagenTwoTower, TaskSpec

DRUG_IN_FEATURES = 6
DRUG_EDGE_DIM = 4
GENO_IN_FEATURES = 5


def _mol_graph() -> Data:
    """A tiny 3-atom molecule graph. GINEConv requires edge_attr."""
    return Data(
        x=torch.randn(3, DRUG_IN_FEATURES),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, DRUG_EDGE_DIM),
    )


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, GENO_IN_FEATURES),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
    )


def _poly_drug_batch() -> Batch:
    """2 patients x 2 molecules each (mol 0/2 focal, mol 1/3 neighbour)."""
    mols = [_mol_graph(), _mol_graph(), _mol_graph(), _mol_graph()]
    drug_batch = Batch.from_data_list(mols)
    drug_batch.mol_to_patient = torch.tensor([0, 0, 1, 1])
    # Both directions, per patient, over molecule-global indices.
    drug_batch.ddi_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    drug_batch.is_focal = torch.tensor([1, 0, 1, 0])
    return drug_batch


def _geno_batch(n_patients: int = 2) -> Batch:
    return Batch.from_data_list([_geno_graph() for _ in range(n_patients)])


def _single_mol_drug_batch(n_patients: int = 2) -> Batch:
    """One molecule per patient — the pre-D4 (Phase C) shape."""
    return Batch.from_data_list([_mol_graph() for _ in range(n_patients)])


def _poly_cfg(use_polypharmacy: bool = True, use_cross_attention: bool = True):
    return PharmagenConfig(
        drug_in_features=DRUG_IN_FEATURES,
        drug_edge_dim=DRUG_EDGE_DIM,
        drug_hidden_dim=16,
        ddi_edge_dim=None,
        geno_in_features=GENO_IN_FEATURES,
        geno_edge_dim=None,
        geno_hidden_dim=16,
        embedding_dim=16,
        num_layers=2,
        heads=2,
        dropout=0.0,
        drug_global_dim=0,
        drug_admet_dim=0,
        geno_global_dim=0,
        use_polypharmacy=use_polypharmacy,
        use_cross_attention=use_cross_attention,
        axes={
            "pheno": TaskSpec(dim=3, kind="multiclass"),
            "outcome": TaskSpec(dim=1, kind="binary"),
        },
    )


def _zero_molecules(drug_batch: Batch, mol_indices: list[int]) -> Batch:
    """Return a copy of `drug_batch` with atom features of given molecules zeroed."""
    clone = drug_batch.clone()
    atom_mask = torch.isin(clone.batch, torch.tensor(mol_indices))
    clone.x = clone.x.clone()
    clone.x[atom_mask] = 0.0
    return clone


def test_focal_readout_forward_runs_and_produces_axis_logits_and_z():
    torch.manual_seed(0)
    model = PharmagenTwoTower(_poly_cfg())
    drug_batch = _poly_drug_batch()
    geno_batch = _geno_batch()

    out = model(drug_batch, geno_batch)

    assert "pheno" in out
    assert "outcome" in out
    assert "_z" in out
    assert out["pheno"].shape == (2, 3)
    assert out["outcome"].shape == (2, 1)
    assert out["_z"].shape[0] == 2


def test_focal_dominance_zeroing_focal_perturbs_output_more_than_neighbour():
    torch.manual_seed(0)
    model = PharmagenTwoTower(_poly_cfg())
    model.eval()
    geno_batch = _geno_batch()

    # Reuse the SAME base random batch for all three forwards below --
    # calling `_poly_drug_batch()` again would resample `torch.randn` and
    # compare apples to oranges.
    base_batch = _poly_drug_batch()
    base_out = model(base_batch, geno_batch)

    focal_zeroed = _zero_molecules(base_batch, [0, 2])
    neighbour_zeroed = _zero_molecules(base_batch, [1, 3])

    focal_out = model(focal_zeroed, geno_batch)
    neighbour_out = model(neighbour_zeroed, geno_batch)

    focal_delta = (base_out["_z"] - focal_out["_z"]).norm().item()
    neighbour_delta = (base_out["_z"] - neighbour_out["_z"]).norm().item()

    assert focal_delta > neighbour_delta


def test_focal_readout_engages_a_different_path_than_plain_pooling():
    """The focal-anchored path must actually change drug_graph_emb.

    Compares the full model output with `is_focal` present against the same
    batch with `is_focal` removed (falls back to the pre-D4 pooled path).
    """
    torch.manual_seed(0)
    model = PharmagenTwoTower(_poly_cfg())
    model.eval()
    geno_batch = _geno_batch()

    base_batch = _poly_drug_batch()
    with_focal = model(base_batch, geno_batch)

    no_focal_batch = base_batch.clone()
    del no_focal_batch.is_focal
    without_focal = model(no_focal_batch, geno_batch)

    assert not torch.allclose(with_focal["_z"], without_focal["_z"])


def test_switches_off_behaviour_is_unchanged_single_molecule_batch():
    """use_polypharmacy=False must be identical to the pre-D4 (Phase C) path."""
    torch.manual_seed(0)
    model = PharmagenTwoTower(
        _poly_cfg(use_polypharmacy=False, use_cross_attention=False)
    )
    drug_batch = _single_mol_drug_batch()
    geno_batch = _geno_batch()

    out = model(drug_batch, geno_batch)

    assert set(out) >= {"pheno", "outcome", "_z"}
    assert out["pheno"].shape == (2, 3)
    assert out["outcome"].shape == (2, 1)


def test_switches_off_ignores_is_focal_if_present():
    """is_focal on the batch must be a no-op when use_polypharmacy=False."""
    torch.manual_seed(0)
    model = PharmagenTwoTower(
        _poly_cfg(use_polypharmacy=False, use_cross_attention=False)
    )
    model.eval()
    geno_batch = _geno_batch()

    drug_batch = _single_mol_drug_batch()
    out_without = model(drug_batch, geno_batch)

    drug_batch_with_focal = drug_batch.clone()
    drug_batch_with_focal.is_focal = torch.tensor([1, 1])
    out_with = model(drug_batch_with_focal, geno_batch)

    assert torch.allclose(out_without["_z"], out_with["_z"])
