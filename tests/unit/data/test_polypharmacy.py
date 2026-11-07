"""Tests for src.data.polypharmacy.PseudoPatientBuilder."""

from typing import TYPE_CHECKING

import torch
from torch_geometric.data.data import Data

from src.data.cache import make_empty_graph
from src.data.ddi import (
    DDI_EDGE_DIM,
    DDIGraph,
    build_ddi_artifact,
)
from src.data.polypharmacy import PseudoPatientBuilder

if TYPE_CHECKING:
    from pathlib import Path


class _StubCache:
    """The two members PseudoPatientBuilder uses: ``drug_index`` and ``get_drug``."""

    def __init__(self, drug_ids: tuple[str, ...]) -> None:
        self.drug_index = {d: f"KEY-{d}" for d in drug_ids}

    def get_drug(self, drug_id: str) -> Data:
        if drug_id not in self.drug_index:
            return make_empty_graph("drug", graph_id=drug_id)
        g = Data(
            x=torch.randn(2, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros(2, 3),
        )
        g.drug_id = drug_id
        return g


def _write_ddi(tmp_path: Path, rows: str) -> DDIGraph:
    csv = tmp_path / "edges.csv"
    csv.write_text("drug_a,drug_b,category,severity\n" + rows)
    out = tmp_path / "ddi_graph.pt"
    build_ddi_artifact(csv, out)
    return DDIGraph.load(out)


def _ddi_graph(tmp_path: Path) -> DDIGraph:
    """Focal CHEMBL1, neighbours CHEMBL2 and CHEMBL3."""
    return _write_ddi(tmp_path, "CHEMBL1,CHEMBL2,PK,3.0\nCHEMBL1,CHEMBL3,PD,1.0\n")


def _cache_missing_neighbor_3(tmp_path: Path) -> _StubCache:  # noqa: ARG001
    """Graphs for CHEMBL1 (focal) and CHEMBL2 only — CHEMBL3 has a DDI edge but
    no graph, so it must be dropped by the builder."""
    return _StubCache(("CHEMBL1", "CHEMBL2"))


def test_focal_is_always_index_zero(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL1")

    assert sample["molecules"][0].drug_id == "CHEMBL1"
    assert sample["is_focal"][0].item() == 1


def test_mol_to_patient_is_all_zero_single_patient(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL1")

    mol_to_patient = sample["mol_to_patient"]
    assert mol_to_patient.shape == (len(sample["molecules"]),)
    assert torch.all(mol_to_patient == 0)


def test_neighbor_without_graph_is_dropped_and_indices_remapped(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL1")

    # Only focal (CHEMBL1) + kept neighbour (CHEMBL2) survive; CHEMBL3 dropped.
    molecules = sample["molecules"]
    assert len(molecules) == 2
    assert [m.drug_id for m in molecules] == ["CHEMBL1", "CHEMBL2"]

    edge_index = sample["ddi_edge_index"]
    assert edge_index.dtype == torch.long
    assert edge_index.shape[1] == 2  # (0,1) and (1,0)
    assert torch.all(edge_index < len(molecules))
    assert torch.all(edge_index >= 0)


def test_ddi_edge_attr_aligned_with_edge_index(tmp_path):
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL1")

    edge_index = sample["ddi_edge_index"]
    edge_attr = sample["ddi_edge_attr"]
    assert edge_attr.shape == (edge_index.shape[1], DDI_EDGE_DIM)
    # Both directions carry the same (duplicated) edge-attr row.
    assert torch.equal(edge_attr[0], edge_attr[1])


def test_no_kept_neighbors_yields_empty_edges(tmp_path):
    ddi = _ddi_graph(tmp_path)
    # Cache with only the focal drug — every neighbour gets dropped.
    cache = _StubCache(("CHEMBL1",))
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL1")

    assert len(sample["molecules"]) == 1
    assert sample["ddi_edge_index"].shape == (2, 0)
    assert sample["ddi_edge_attr"].shape == (0, DDI_EDGE_DIM)
    assert sample["is_focal"].tolist() == [1]


def test_unknown_focal_drug_still_returns_focal_only_sample(tmp_path):
    """A focal drug absent from the DDI graph has no neighbours, but the
    focal molecule graph is still built (or placeholder'd) at index 0."""
    ddi = _ddi_graph(tmp_path)
    cache = _cache_missing_neighbor_3(tmp_path)
    builder = PseudoPatientBuilder(ddi, cache, max_neighbors=8)

    sample = builder.build("CHEMBL999")

    assert len(sample["molecules"]) == 1
    assert sample["is_focal"].tolist() == [1]
    assert sample["ddi_edge_index"].shape == (2, 0)


def test_self_interaction_row_does_not_duplicate_focal(tmp_path):
    """A self-loop row (drug_a == drug_b) in the DDI export must not add the
    focal drug as its own neighbour."""
    ddi = _write_ddi(tmp_path, "CHEMBL1,CHEMBL1,PK,3.0\nCHEMBL1,CHEMBL2,PD,1.0\n")
    cache = _StubCache(("CHEMBL1", "CHEMBL2"))

    sample = PseudoPatientBuilder(ddi, cache, max_neighbors=8).build("CHEMBL1")

    # Only CHEMBL2 is a real neighbour; the self-loop is filtered out.
    assert len(sample["molecules"]) == 2
    assert sample["is_focal"].tolist() == [1, 0]
    assert int(sample["ddi_edge_index"].max()) == 1  # local indices 0 and 1 only
