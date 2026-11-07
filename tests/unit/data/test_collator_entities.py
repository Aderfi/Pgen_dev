"""Rows as sets of drug entities: resolution, dataset items and collation."""

from __future__ import annotations

import logging
import random

import polars as pl
import pytest
import torch

from src.data.cache import ENTITY_CLASS, ENTITY_DRUG, GraphCache
from src.data.collator import DoubleTowerCollater
from src.data.datasets import DoubleTowerDataset
from src.data.genotype import GenotypeGraphBuilder, HgvsCatalog

# ATC "X" = {aspirin, ethanol}; "E" = members not in the library
MEMBERS = {
    "X": frozenset({"CHEMBL545", "CHEMBL25"}),
    "E": frozenset({"CHEMBL999"}),
    "BIG": frozenset({"CHEMBL25", "CHEMBL545", "CHEMBL671"}),
}


@pytest.fixture
def cache(pharm_library) -> GraphCache:
    store, keys = pharm_library
    return GraphCache(keys, store=store, class_members=MEMBERS, rng=random.Random(0))


def _dataset(cache: GraphCache, keys: list[str], **kw) -> DoubleTowerDataset:
    df = pl.DataFrame({"drugs_chembl": keys, "genotype": ["rs1"] * len(keys)})
    return DoubleTowerDataset(
        df,
        "drugs_chembl",
        "genotype",
        [],
        [],
        genotype_builder=GenotypeGraphBuilder(HgvsCatalog(), {}),
        drug_cache=cache,
        **kw,
    )


def test_get_entity(cache) -> None:
    kind, drugs = cache.get_entity("CHEMBL25")
    assert (kind, len(drugs)) == (ENTITY_DRUG, 1)
    assert cache.get_entity("CHEMBL999") == (ENTITY_DRUG, [])
    kind, drugs = cache.get_entity("ATC:X")
    assert (kind, len(drugs)) == (ENTITY_CLASS, 2)
    assert cache.get_entity("ATC:E") == (ENTITY_CLASS, [])
    view = cache.without_members(["CHEMBL25"])
    assert len(view.get_entity("ATC:X")[1]) == 1
    assert len(view.get_entity("CHEMBL25")[1]) == 1  # single drugs unaffected


def test_row_entities_from_keys(cache) -> None:
    item = _dataset(cache, ["CHEMBL25|ATC:X"])[0]
    assert [kind for kind, _ in item["drug_data"]] == [ENTITY_DRUG, ENTITY_CLASS]
    assert [len(mols) for _, mols in item["drug_data"]] == [1, 2]


def test_row_with_only_empty_class_gets_dummy(cache, caplog) -> None:
    with caplog.at_level(logging.INFO):
        ds = _dataset(cache, ["ATC:E", "CHEMBL25|CHEMBL999"])
    assert "1 rows have no readable drug (dummy graph)" in caplog.text
    ((kind, mols),) = ds[0]["drug_data"]
    assert kind == ENTITY_DRUG and len(mols) == 1
    assert mols[0].name == "dummy_drug"
    assert len(ds[1]["drug_data"]) == 1  # the unbuilt drug is skipped
    assert ds.rows_without_drug == 1


def test_class_cap_samples_in_training_only(cache) -> None:
    train = _dataset(cache, ["ATC:BIG"], max_class_members=2)
    assert len(train[0]["drug_data"][0][1]) == 2
    infer = _dataset(cache, ["ATC:BIG"], max_class_members=2, inference_mode=True)
    assert len(infer[0]["drug_data"][0][1]) == 3


def _sample(entities, n_targets: int = 0) -> dict:
    geno = GenotypeGraphBuilder(HgvsCatalog(), {}).build("CYP2D6", "rs1", None, None)
    return {"drug_data": entities, "geno_data": geno, "targets": {}}


def test_collate_entity_indices(cache) -> None:
    a = cache.get_entity("CHEMBL25")
    b = cache.get_entity("CHEMBL545")
    cd = cache.get_entity("ATC:X")
    batch = DoubleTowerCollater()([_sample([a]), _sample([b, cd])])
    drug = batch["drug_batch"]
    assert drug.mol_to_entity.tolist() == [0, 1, 2, 2]
    assert drug.entity_to_row.tolist() == [0, 1, 1]
    assert drug.entity_kind.tolist() == [ENTITY_DRUG, ENTITY_DRUG, ENTITY_CLASS]
    assert drug.num_graphs == 4


def test_collate_mixed_pharmacophores(cache) -> None:
    entities = [cache.get_entity(k) for k in ("CHEMBL25", "CHEMBL545", "CHEMBL671")]
    batch = DoubleTowerCollater()([_sample(entities)])
    drug = batch["drug_batch"]
    total_k = sum(m.pharm_x.size(0) for _, mols in entities for m in mols)
    assert drug.pharm_x.size(0) == total_k
    assert drug.pharm_x_batch.shape == (total_k,)
    assert drug.pharm_pos.shape == (total_k, 3)
    assert drug.pharm_global.shape == (3, 258)
    assert drug.has_3d.tolist() == [1.0, 0.0, 0.0]
    assert int(drug.pharm_edge_index.max()) < total_k
    # point -> molecule index follows the molecules' point counts
    counts = torch.bincount(drug.pharm_x_batch, minlength=3).tolist()
    assert counts == [m.pharm_x.size(0) for _, mols in entities for m in mols]


def test_collate_real_then_dummy_drug(cache) -> None:
    # final review #1: PyG collates on the first item's keys; a dummy without
    # the real drug's attributes (z) broke [real, dummy] batches
    ds = _dataset(cache, ["CHEMBL25", "CHEMBL999", "ATC:E"])
    for order in ([0, 1, 2], [1, 0, 2]):
        batch = DoubleTowerCollater()([ds[i] for i in order])
        assert batch["drug_batch"].num_graphs == 3
        assert batch["drug_batch"].z.numel() == batch["drug_batch"].x.size(0)


def test_sampling_is_seeded_from_settings(pharm_library) -> None:
    # final review #7: conformer and member sampling are reproducible
    store, keys = pharm_library

    def draws() -> list[float]:
        c = GraphCache(keys, store=store, class_members=MEMBERS)
        ds = _dataset(c, ["ATC:BIG"], max_class_members=2)
        members = [m.x.size(0) for m in ds[0]["drug_data"][0][1]]
        return [
            *members,
            *(c.get_drug("CHEMBL25").pharm_pos.sum().item() for _ in range(5)),
        ]

    assert draws() == draws()


def test_worker_init_reseeds_per_worker(cache, monkeypatch) -> None:
    import torch.utils.data as tud

    from src.data.datasets import seed_worker

    ds = _dataset(cache, ["CHEMBL25"])
    states = []
    for wid, seed in ((0, 100), (1, 101)):
        info = type("W", (), {"dataset": ds, "id": wid, "seed": seed})()
        monkeypatch.setattr(tud, "get_worker_info", lambda info=info: info)
        seed_worker(wid)
        states.append((ds._rng.random(), ds.cache._rng.random()))
    assert states[0] != states[1]
