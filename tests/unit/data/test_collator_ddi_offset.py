import torch
from torch_geometric.data import Data

from src.data.collator import DoubleTowerCollater


def _sample(n_mol, edges):
    drug = Data(x=torch.randn(n_mol, 6), edge_index=torch.empty(2, 0, dtype=torch.long))
    drug.mol_to_patient = torch.zeros(n_mol, dtype=torch.long)
    drug.ddi_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    geno = Data(x=torch.randn(1, 5), edge_index=torch.empty(2, 0, dtype=torch.long))
    return {"drug_data": drug, "geno_data": geno, "targets": {"pheno": torch.tensor(0)}}


def test_ddi_edges_offset_across_batch():
    # patient 0: 2 molecules, edge (0,1); patient 1: 2 molecules, edge (0,1)
    batch = DoubleTowerCollater()([_sample(2, [[0, 1]]), _sample(2, [[0, 1]])])
    ddi = batch["drug_batch"].ddi_edge_index
    # second patient's edge must be offset to (2,3), not (0,1)
    assert ddi.max().item() == 3
    assert batch["drug_batch"].mol_to_patient.tolist() == [0, 0, 1, 1]


def test_ddi_edges_offset_across_batch_uneven_molecule_counts():
    # patient 0: 3 molecules, edge (0,2); patient 1: 2 molecules, edge (0,1).
    # A naive fixed-offset (e.g. always +2) would get this wrong; the offset
    # must track each sample's actual molecule count.
    batch = DoubleTowerCollater()([_sample(3, [[0, 2]]), _sample(2, [[0, 1]])])
    ddi = batch["drug_batch"].ddi_edge_index
    # second patient's edge (0,1) must be offset to (3,4)
    assert ddi[:, 1].tolist() == [3, 4]
    assert batch["drug_batch"].mol_to_patient.tolist() == [0, 0, 0, 1, 1]
