"""Tests for src.data.cache."""

from pathlib import Path

import pytest
import torch
from torch_geometric.data.data import Data

from src.data.cache import GraphCache, GraphDims, make_empty_graph


@pytest.fixture
def saved_drug(tmp_path: Path) -> Path:
    """Round-trip a tiny drug-shaped graph to disk."""
    g = Data(
        x=torch.randn(3, 25),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_attr=torch.zeros(3, 7),
    )
    g.cid = "2244"
    g.name = "aspirin"
    g.smiles = "CC(=O)Oc1ccccc1C(=O)O"
    p = tmp_path / "2244_aspirin.pt"
    torch.save(g, p)
    return p


@pytest.fixture
def saved_variant(tmp_path: Path) -> Path:
    g = Data(
        x=torch.randn(5, 9),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.zeros(4, 3),
    )
    g.variant_name = "*4"
    p = tmp_path / "CYP2D6_star4.pt"
    torch.save(g, p)
    return p


class TestMakeEmptyGraph:
    def test_drug_dims(self) -> None:
        g = make_empty_graph("drug")
        assert g.x.shape == (1, 61)
        assert g.edge_attr.shape == (0, 18)
        assert g.name == "dummy_drug"

    def test_geno_dims(self) -> None:
        g = make_empty_graph("geno", graph_id="CYP2D6_*4")
        assert g.x.shape == (1, 9)
        assert g.edge_attr.shape == (0, 3)
        assert g.variant_name == "CYP2D6_*4"

    def test_custom_dims(self) -> None:
        dims = GraphDims(drug_features=10, drug_edges=2, geno_features=5, geno_edges=1)
        g = make_empty_graph("drug", dims=dims)
        assert g.x.shape == (1, 10)
        assert g.edge_attr.shape == (0, 2)

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="unknown graph kind"):
            make_empty_graph("nope")  # type: ignore[arg-type]


class TestGraphCacheLookup:
    def test_drug_cache_hit(self, saved_drug: Path) -> None:
        cache = GraphCache(
            drug_index={"2244": saved_drug},
            variant_index={},
        )
        first = cache.get_drug("2244")
        assert first.x.shape == (3, 25)
        # Hit on the second call (now cached or re-read)
        cache.get_drug("2244")
        # First call was a miss, second a miss too (cache only fills via preload).
        # But the stats track hits — check we got at least one read.
        stats = cache.stats()
        assert stats["drug_misses"] == 2  # never preloaded → both misses

    def test_unknown_drug_returns_empty(self) -> None:
        cache = GraphCache(drug_index={}, variant_index={})
        g = cache.get_drug("does_not_exist")
        assert g.x.shape == (1, 61)
        assert cache.stats()["drug_misses"] == 1

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a torch file")
        cache = GraphCache(drug_index={"123": bad}, variant_index={})
        g = cache.get_drug("123")
        assert g.x.shape == (1, 61)  # placeholder

    def test_variant_lookup_via_underscore_split(self, saved_variant: Path) -> None:
        cache = GraphCache(
            drug_index={},
            variant_index={"CYP2D6": {"star4": saved_variant}},
        )
        g = cache.get_variant("CYP2D6_star4")
        assert g.x.shape == (5, 9)

    def test_variant_lookup_no_underscore_returns_empty(self) -> None:
        cache = GraphCache(drug_index={}, variant_index={})
        g = cache.get_variant("malformed")  # no underscore
        assert g.x.shape == (1, 9)


class TestPreload:
    def test_preload_drugs_caches(self, saved_drug: Path) -> None:
        cache = GraphCache(drug_index={"2244": saved_drug}, variant_index={})
        cache.preload_drugs(["2244"])
        assert cache.cached_drug_count == 1
        cache.get_drug("2244")
        # Now we should see a hit.
        assert cache.stats()["drug_hits"] == 1

    def test_preload_variants_caches(self, saved_variant: Path) -> None:
        cache = GraphCache(
            drug_index={},
            variant_index={"CYP2D6": {"star4": saved_variant}},
        )
        cache.preload_variants(["CYP2D6_star4"])
        assert cache.cached_variant_count == 1
        cache.get_variant("CYP2D6_star4")
        assert cache.stats()["geno_hits"] == 1


class TestInferenceMode:
    def test_metadata_preserved(self, saved_drug: Path) -> None:
        cache = GraphCache(
            drug_index={"2244": saved_drug},
            variant_index={},
            inference_mode=True,
        )
        g = cache.get_drug("2244")
        assert hasattr(g, "cid")
        assert hasattr(g, "name")
        assert g.cid == "2244"

    def test_metadata_stripped_in_training_mode(self, saved_drug: Path) -> None:
        cache = GraphCache(
            drug_index={"2244": saved_drug},
            variant_index={},
            inference_mode=False,
        )
        g = cache.get_drug("2244")
        assert not hasattr(g, "cid")
        assert not hasattr(g, "name")
