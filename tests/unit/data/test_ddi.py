import torch

from src.data.ddi import (
    DDI_EDGE_DIM,
    DDIGraph,
    build_ddi_artifact,
)


def test_build_and_query(tmp_path):
    csv = tmp_path / "edges.csv"
    csv.write_text(
        "drug_a,drug_b,category,severity\nCHEMBL1,CHEMBL2,PK,3.0\nCHEMBL1,CHEMBL3,PD,1.0\n"
    )
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    g = DDIGraph.load(out)
    neighbors, edge_attr = g.neighbors("CHEMBL1", k=5)
    assert set(neighbors) == {"CHEMBL2", "CHEMBL3"}
    assert edge_attr.shape[0] == len(neighbors)
    assert edge_attr.shape[1] == DDI_EDGE_DIM


def test_unknown_drug_returns_empty(tmp_path):
    csv = tmp_path / "edges.csv"
    csv.write_text("drug_a,drug_b,category,severity\nCHEMBL1,CHEMBL2,PK,3.0\n")
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    neighbors, edge_attr = DDIGraph.load(out).neighbors("CHEMBL999", k=5)
    assert neighbors == []
    assert edge_attr.shape == (0, DDI_EDGE_DIM)


def test_out_of_vocab_category_maps_to_unknown_bucket(tmp_path):
    """A drifted/unseen category must land in the frozen 'unknown' slot, not an
    all-zero one-hot indistinguishable from padding."""
    csv = tmp_path / "edges.csv"
    csv.write_text(
        "drug_a,drug_b,category,severity\nCHEMBL1,CHEMBL2,DRIFTED_KIND,2.0\n"
    )
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    _, edge_attr = DDIGraph.load(out).neighbors("CHEMBL1", k=5)
    one_hot = edge_attr[0, :-1]
    assert one_hot.sum().item() == 1.0  # exactly one category slot set (unknown)
    assert edge_attr[0, -1].item() == 2.0  # severity preserved
