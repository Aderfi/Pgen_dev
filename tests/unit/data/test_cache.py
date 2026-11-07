"""Tests for src.data.cache — drug graphs served from a Modephere library."""

from __future__ import annotations

import polars as pl
import pytest
from modephere.types.layouts import MOLECULE_GRAPH_V1

from src.data.cache import GraphCache, GraphDims, make_empty_graph
from src.data.library.admet import ADMET_ENDPOINTS, DRUG_ADMET_DIM, AdmetProvider
from src.data.library.bridge import chembl_index


def _admet(inchikey: str, value: float) -> AdmetProvider:
    return AdmetProvider(
        pl.DataFrame({"inchikey": [inchikey]} | {e: [value] for e in ADMET_ENDPOINTS})
    )


class TestGraphDims:
    def test_drug_widths_are_modephere_layout(self) -> None:
        dims = GraphDims()
        assert dims.drug_features == MOLECULE_GRAPH_V1.node_dim
        assert dims.drug_edges == MOLECULE_GRAPH_V1.edge_dim
        assert dims.drug_global == MOLECULE_GRAPH_V1.global_dim
        assert dims.drug_admet == DRUG_ADMET_DIM


class TestMakeEmptyGraph:
    def test_drug_dims(self) -> None:
        g = make_empty_graph("drug", graph_id="CHEMBL25")
        assert g.x.shape == (1, MOLECULE_GRAPH_V1.node_dim)
        assert g.edge_attr.shape == (0, MOLECULE_GRAPH_V1.edge_dim)
        assert g.global_feats.shape == (1, MOLECULE_GRAPH_V1.global_dim)
        assert g.admet_feats.shape == (1, DRUG_ADMET_DIM)
        assert g.name == "dummy_drug"
        assert g.drug_id == "CHEMBL25"

    def test_geno_dims(self) -> None:
        g = make_empty_graph("geno", graph_id="CYP2D6")
        assert g.x.shape == (1, 45)
        assert g.edge_attr.shape == (0, 3)
        assert g.geno_function.shape == (1, 6)
        assert g.gene_token.tolist() == [1]  # UNK gene hub
        assert g.gene == "CYP2D6"

    def test_custom_dims(self) -> None:
        dims = GraphDims(drug_features=10, drug_edges=2, geno_features=5, geno_edges=1)
        g = make_empty_graph("drug", dims=dims)
        assert g.x.shape == (1, 10)
        assert g.edge_attr.shape == (0, 2)

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown graph kind"):
            make_empty_graph("nope")  # type: ignore[arg-type]


class TestGraphCacheLookup:
    def test_drug_is_modephere_molecule_graph_with_admet(
        self, modephere_library
    ) -> None:
        store, keys = modephere_library
        aspirin = keys["CHEMBL25"]
        cache = GraphCache(keys, store=store, admet=_admet(aspirin, 0.25))
        g = cache.get_drug("CHEMBL25")
        assert g.x.shape == (13, MOLECULE_GRAPH_V1.node_dim)  # aspirin heavy atoms
        assert g.edge_attr.shape[1] == MOLECULE_GRAPH_V1.edge_dim
        assert g.global_feats.shape == (1, MOLECULE_GRAPH_V1.global_dim)
        assert g.z.shape == (13,)
        assert g.admet_feats[0, 0].item() == pytest.approx(0.25)

    def test_second_lookup_is_a_cache_hit(self, modephere_library) -> None:
        store, keys = modephere_library
        cache = GraphCache(keys, store=store)
        cache.get_drug("CHEMBL25")
        cache.get_drug("CHEMBL25")
        assert cache.stats()["drug_misses"] == 1
        assert cache.stats()["drug_hits"] == 1

    def test_drug_without_admet_profile_gets_zeros(self, modephere_library) -> None:
        store, keys = modephere_library
        g = GraphCache(keys, store=store).get_drug("CHEMBL545")
        assert g.admet_feats.abs().sum().item() == 0.0

    def test_unknown_drug_returns_empty(self) -> None:
        cache = GraphCache(drug_index={})
        g = cache.get_drug("CHEMBL0")
        assert g.x.shape == (1, MOLECULE_GRAPH_V1.node_dim)
        assert cache.stats()["drug_misses"] == 1

    def test_unreadable_compound_returns_empty(self, tmp_path) -> None:
        from modephere.library import LibraryStore

        store = LibraryStore(tmp_path)
        store.compounds_dir.mkdir()
        store.path_for("XXXXXXXXXXXXXX-YYYYYYYYYY-N").write_bytes(b"not safetensors")
        cache = GraphCache({"CHEMBL1": "XXXXXXXXXXXXXX-YYYYYYYYYY-N"}, store=store)
        assert cache.get_drug("CHEMBL1").x.shape == (1, MOLECULE_GRAPH_V1.node_dim)


class TestPreload:
    def test_preload_drugs_caches(self, modephere_library) -> None:
        store, keys = modephere_library
        cache = GraphCache(keys, store=store)
        cache.preload_drugs(["CHEMBL25", "CHEMBL0"])
        assert cache.cached_drug_count == 1
        cache.get_drug("CHEMBL25")
        assert cache.stats()["drug_hits"] == 1


class TestInferenceMode:
    def test_identity_kept_at_inference(self, modephere_library) -> None:
        store, keys = modephere_library
        g = GraphCache(keys, store=store, inference_mode=True).get_drug("CHEMBL25")
        assert g.drug_id == "CHEMBL25"
        assert g.inchikey == keys["CHEMBL25"]

    def test_identity_stripped_for_training(self, modephere_library) -> None:
        store, keys = modephere_library
        g = GraphCache(keys, store=store).get_drug("CHEMBL25")
        assert "drug_id" not in g
        assert "inchikey" not in g

    def test_cached_graph_is_not_mutated_by_inference_copies(
        self, modephere_library
    ) -> None:
        store, keys = modephere_library
        cache = GraphCache(keys, store=store, inference_mode=True)
        cache.get_drug("CHEMBL25").x.zero_()
        assert cache.get_drug("CHEMBL25").x.abs().sum().item() > 0


def test_chembl_index_reads_the_store_registry(modephere_library) -> None:
    store, keys = modephere_library
    assert chembl_index(store) == keys
