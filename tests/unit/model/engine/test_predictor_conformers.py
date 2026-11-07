"""Inference averages predictions over the lowest-energy conformers."""

from __future__ import annotations

import polars as pl
import pytest
import torch

from src.data.cache import GraphCache
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.genotype import GenotypeGraphBuilder, HgvsCatalog
from src.model.engine.predictor import (
    PGenPredictor,
    average_predictions,
    conformer_passes,
)


class _Cache:
    def __init__(self, conformers: dict[str, int]) -> None:
        self.conformers = conformers

    def entity_members(self, key: str) -> list[str]:
        return [key] if key in self.conformers else []

    def n_conformers(self, drug_id: str) -> int:
        return self.conformers[drug_id]


def test_conformer_passes_bounded_by_ensemble():
    cache = _Cache({"ONE": 1, "FIVE": 5, "FLAT": 0})
    assert conformer_passes(cache, [["ONE"]], 3) == 1
    assert conformer_passes(cache, [["FIVE"]], 3) == 3
    assert conformer_passes(cache, [["FLAT"]], 3) == 1
    assert conformer_passes(cache, [["FLAT"], ["ONE", "FIVE"]], 10) == 5
    assert conformer_passes(cache, [["MISSING"]], 3) == 1


def test_average_predictions_averages_probabilities():
    a = {"multi": torch.tensor([[0.0]]), "cls": torch.tensor([[2.0, 0.0]])}
    b = {"multi": torch.tensor([[2.0]]), "cls": torch.tensor([[0.0, 2.0]])}
    out = average_predictions([a, b], multi_label_cols={"multi"})
    p = (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(2.0))) / 2
    assert torch.sigmoid(out["multi"]).item() == pytest.approx(p.item())
    assert torch.softmax(out["cls"], -1).tolist()[0] == pytest.approx([0.5, 0.5])
    assert average_predictions([a], multi_label_cols=set()) is a


class _SpyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions: list[torch.Tensor] = []

    def forward(self, drug_batch, geno_batch):  # noqa: ANN001, ANN201
        self.positions.append(drug_batch.pharm_pos.clone())
        return {"t": torch.zeros(geno_batch.num_graphs, 2)}


def _predictor(k: int) -> PGenPredictor:
    predictor = object.__new__(PGenPredictor)
    predictor.device = torch.device("cpu")
    predictor.collater = DoubleTowerCollater(inference_mode=True)
    predictor.target_cols = ["t"]
    predictor.multi_label_cols = set()
    predictor.inference_conformers = k
    predictor.model = _SpyModel()
    return predictor


def test_forward_runs_one_pass_per_conformer(pharm_library):
    store, keys = pharm_library
    cache = GraphCache(keys, store=store, inference_mode=True)
    n = cache.n_conformers("CHEMBL25")
    assert n >= 2
    df = pl.DataFrame({"drugs_chembl": ["CHEMBL25"], "genotype": ["rs1"]})
    ds = DoubleTowerDataset(
        df,
        "drugs_chembl",
        "genotype",
        [],
        [],
        genotype_builder=GenotypeGraphBuilder(HgvsCatalog(), {}),
        drug_cache=cache,
        inference_mode=True,
    )
    predictor = _predictor(k=2)
    out = predictor._forward_dataset(ds, batch_size=8)
    assert out["t"].shape == (1, 2)
    seen = predictor.model.positions
    assert len(seen) == 2
    assert not torch.equal(seen[0], seen[1])  # conformers 0 and 1
    assert ds.cache is cache  # the dataset's cache is restored

    flat = DoubleTowerDataset(
        pl.DataFrame({"drugs_chembl": ["CHEMBL545"], "genotype": ["rs1"]}),
        "drugs_chembl",
        "genotype",
        [],
        [],
        genotype_builder=GenotypeGraphBuilder(HgvsCatalog(), {}),
        drug_cache=cache,
        inference_mode=True,
    )
    predictor = _predictor(k=3)
    predictor._forward_dataset(flat, batch_size=8)
    assert len(predictor.model.positions) == 1  # 2D-only: one pass


def test_predict_single_rejects_unknown_drugs(pharm_library):
    # final review #5: never answer from the placeholder graph
    from src.core import UnknownDrugError
    from src.data.cleaning import PharmacogenomicCleaner

    store, keys = pharm_library
    predictor = _predictor(k=1)
    predictor.feature_cols = ["drugs_chembl", "genotype"]
    predictor.cleaner = PharmacogenomicCleaner()
    predictor.dims = GraphCache(keys).dims
    predictor._genotype_builder = GenotypeGraphBuilder(HgvsCatalog(), {})
    predictor._drug_cache = GraphCache(keys, store=store, inference_mode=True)
    from sklearn.preprocessing import LabelEncoder

    predictor.encoders = {"t": LabelEncoder().fit(["a", "b"])}
    predictor.label_table = None
    sample = {"gene": "CYP2D6", "genotype": "rs1"}
    with pytest.raises(UnknownDrugError, match="CHEMBL999"):
        predictor.predict_single(sample | {"drugs_chembl": "CHEMBL999"})
    with pytest.raises(UnknownDrugError, match="ATC:N02X"):
        predictor.predict_single(sample | {"drugs_chembl": "ATC:N02X"})
    assert predictor.predict_single(sample | {"drugs_chembl": "CHEMBL25"}) is not None


def test_api_maps_unknown_drugs_to_422(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import deps
    from src.api.main import create_app
    from src.core import UnknownDrugError

    class _Predictor:
        def predict_single(self, features):  # noqa: ANN001, ANN201
            raise UnknownDrugError("no readable drug: CHEMBL999")

    class _Registry:
        def get(self, name):  # noqa: ANN001, ANN201
            return _Predictor()

    app = create_app()
    app.dependency_overrides[deps.get_registry] = _Registry
    body = {
        "drug_ids": ["CHEMBL999"],
        "allele": {"gene": {"symbol": "CYP2D6"}, "allele": "4"},
    }
    r = TestClient(app).post("/v1/predict", json=body)
    assert r.status_code == 422
    assert "CHEMBL999" in r.json()["detail"]
