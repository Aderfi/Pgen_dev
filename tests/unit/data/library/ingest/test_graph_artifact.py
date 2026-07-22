import torch

from src.data.library.ingest.graph_artifact import DDIGraph, build_ddi_artifact


def test_build_and_query(tmp_path):
    csv = tmp_path / "edges.csv"
    csv.write_text("cid_a,cid_b,category,severity\n1,2,PK,3.0\n1,3,PD,1.0\n")
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    g = DDIGraph.load(out)
    neighbors, edge_attr = g.neighbors(1, k=5)
    assert set(neighbors) == {2, 3}
    assert edge_attr.shape[0] == len(neighbors)
