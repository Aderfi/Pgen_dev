"""Tests for src.data.polypharmacy.PseudoPatientBuilder."""

from pathlib import Path

import torch
from torch_geometric.data.data import Data

from src.data.cache import GraphCache
from src.data.library.ingest.graph_artifact import (
    DDI_EDGE_DIM,
    DDIGraph,
    build_ddi_artifact,
)
from src.data.polypharmacy import PseudoPatientBuilder


def _tiny_drug_graph(cid: str) -> Data:
    g = Data(
        x=torch.randn(2, 4),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, 3),
    )
    g.cid = cid
    return g


def _ddi_graph(tmp_path: Path) -> DDIGraph:
    """Focal cid 1, neighbours 2 and 3."""
    csv = tmp_path / "edges.csv"
    csv.write_text("cid_a,cid_b,category,severity\n1,2,PK,3.0\n1,3,PD,1.0\n")
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    return DDIGraph.load(out)


def _cache_missing_neighbor_3(tmp_path: Path) -> GraphCache:
    """Molecular graphs on disk for cid 1 (focal) and cid 2 only — cid 3 has
    a DDI edge but no graph, so it must be dropped by the builder."""
    drug_index = {}
    for cid in ("1", "2"):
        p = tmp_path / f"{cid}.pt"
        torch.save(_tiny_drug_graph(cid), p)
        drug_index[cid] = p
    return GraphCache(drug_index=drug_index, inference_mode=True)


def test_focal_is_always_index_zero(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("1")

    assert sample["molecules"][0].cid == "1"
    assert sample["is_focal"][0].item() == 1


def test_mol_to_patient_is_all_zero_single_patient(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("1")

    mol_to_patient = sample["mol_to_patient"]
    assert mol_to_patient.shape == (len(sample["molecules"]),)
    assert torch.all(mol_to_patient == 0)


def test_neighbor_without_graph_is_dropped_and_indices_remapped(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("1")

    # Only focal (cid 1) + kept neighbour (cid 2) survive; cid 3 dropped.
    molecules = sample["molecules"]
    assert len(molecules) == 2
    assert [m.cid for m in molecules] == ["1", "2"]

    edge_index = sample["ddi_edge_index"]
    assert edge_index.dtype == torch.long
    assert edge_index.shape[1] == 2  # (0,1) and (1,0)
    assert torch.all(edge_index < len(molecules))
    assert torch.all(edge_index >= 0)


def test_ddi_edge_attr_aligned_with_edge_index(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("1")

    edge_index = sample["ddi_edge_index"]
    edge_attr = sample["ddi_edge_attr"]
    assert edge_attr.shape == (edge_index.shape[1], DDI_EDGE_DIM)
    # Both directions carry the same (duplicated) edge-attr row.
    assert torch.equal(edge_attr[0], edge_attr[1])


def test_no_kept_neighbors_yields_empty_edges(tmp_path):
    ddi = _ddi_graph(tmp_path)
    # Cache with only the focal drug — every neighbour gets dropped.
    drug_index = {"1": tmp_path / "1.pt"}
    torch.save(_tiny_drug_graph("1"), drug_index["1"])
    cache = GraphCache(drug_index=drug_index)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("1")

    assert len(sample["molecules"]) == 1
    assert sample["ddi_edge_index"].shape == (2, 0)
    assert sample["ddi_edge_attr"].shape == (0, DDI_EDGE_DIM)
    assert sample["is_focal"].tolist() == [1]


def test_unknown_focal_cid_still_returns_focal_only_sample(tmp_path):
    """A focal cid absent from the DDI graph has no neighbours, but the
    focal molecule graph is still built (or placeholder'd) at index 0."""
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("999")

    assert len(sample["molecules"]) == 1
    assert sample["is_focal"].tolist() == [1]
    assert sample["ddi_edge_index"].shape == (2, 0)
