"""Tests for PharmagenTwoTower — focused on the per-molecule fusion branches.

The drug tower optionally fuses a per-molecule global descriptor vector
(``global_feats``) and a predicted ADMET profile (``admet_feats``) into its graph
embedding; these tests cover each branch on (dim>0) and off (dim=0).
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.config.models import TrainingParams
from src.data.cache import GraphDims
from src.model.architectures.config import AxisSpec, PharmagenConfig
from src.model.architectures.model import PharmagenTwoTower

PARAMS = TrainingParams(
    embedding_dim=32,
    hidden_dim=16,
    dropout_rate=0.1,
    n_layers=2,
    heads=4,
)
SWITCHES = {"use_polypharmacy": False, "use_cross_attention": False}
GLOBAL_DIM = 1038
ADMET_DIM = 41
GENO_FUNCTION_DIM = 6
TARGETS = {"phenotype_category": 5}
AXES = {
    name: AxisSpec(name=name, dim=dim, kind="multiclass")
    for name, dim in TARGETS.items()
}


def _drug_graph(*, with_global: bool, with_admet: bool = False) -> Data:
    g = Data(
        x=torch.randn(4, 61),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        edge_attr=torch.randn(3, 18),
    )
    if with_global:
        g.global_feats = torch.randn(1, GLOBAL_DIM)
    if with_admet:
        g.admet_feats = torch.randn(1, ADMET_DIM)
    return g


def _geno_graph() -> Data:
    return Data(
        x=torch.randn(3, 9),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 3),
    )


def _model(global_dim: int, admet_dim: int = 0):
    cfg = PharmagenConfig.from_dims(
        dims=GraphDims(
            drug_edges=18,
            drug_global=global_dim,
            drug_admet=admet_dim,
            geno_edges=3,
            geno_function=0,
        ),
        drug_dim=61,
        geno_dim=9,
        axes=AXES,
        params=PARAMS,
        switches=SWITCHES,
    )
    return PharmagenTwoTower(cfg)


def _geno_graph_v2(*, with_function: bool) -> Data:
    g = Data(
        x=torch.randn(3, 30),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 2),
    )
    if with_function:
        g.geno_function = torch.randn(1, GENO_FUNCTION_DIM)
    return g


def _geno_model(geno_function_dim: int):
    cfg = PharmagenConfig.from_dims(
        dims=GraphDims(
            drug_edges=18,
            drug_global=0,
            drug_admet=0,
            geno_edges=2,
            geno_function=geno_function_dim,
        ),
        drug_dim=61,
        geno_dim=30,
        axes=AXES,
        params=PARAMS,
        switches=SWITCHES,
    )
    return PharmagenTwoTower(cfg)


class TestGenoFunctionBranch:
    def test_forward_fuses_geno_function(self) -> None:
        model = _geno_model(GENO_FUNCTION_DIM)
        assert hasattr(model, "geno_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(3)])
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=True) for _ in range(3)]
        )
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_geno_function_branch_disabled(self) -> None:
        model = _geno_model(0)
        assert not hasattr(model, "geno_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=False) for _ in range(2)]
        )
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_geno_function_raises(self) -> None:
        model = _geno_model(GENO_FUNCTION_DIM)
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        # geno graphs lack ``geno_function`` → the fusion branch must complain.
        geno = Batch.from_data_list(
            [_geno_graph_v2(with_function=False) for _ in range(2)]
        )
        with pytest.raises(ValueError, match="geno_function"):
            model(drug, geno)


class TestGlobalBranch:
    def test_forward_with_global_descriptors(self) -> None:
        model = _model(GLOBAL_DIM)
        assert hasattr(model, "drug_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=True) for _ in range(3)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(3)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_global_branch_disabled(self) -> None:
        model = _model(0)
        assert not hasattr(model, "drug_global_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_global_feats_raises(self) -> None:
        model = _model(GLOBAL_DIM)
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        with pytest.raises(ValueError, match="missing 'global_feats'"):
            model(drug, geno)


class TestAdmetBranch:
    def test_forward_with_admet_profile(self) -> None:
        model = _model(0, ADMET_DIM)
        assert hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list(
            [_drug_graph(with_global=False, with_admet=True) for _ in range(3)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(3)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (3, 5)

    def test_admet_branch_disabled(self) -> None:
        model = _model(0, 0)
        assert not hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list([_drug_graph(with_global=False) for _ in range(2)])
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)

    def test_missing_admet_feats_raises(self) -> None:
        model = _model(0, ADMET_DIM)
        drug = Batch.from_data_list(
            [_drug_graph(with_global=False, with_admet=False) for _ in range(2)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        with pytest.raises(ValueError, match="missing 'admet_feats'"):
            model(drug, geno)

    def test_global_and_admet_fuse_together(self) -> None:
        model = _model(GLOBAL_DIM, ADMET_DIM)
        assert hasattr(model, "drug_global_mlp")
        assert hasattr(model, "drug_admet_mlp")
        drug = Batch.from_data_list(
            [_drug_graph(with_global=True, with_admet=True) for _ in range(2)]
        )
        geno = Batch.from_data_list([_geno_graph() for _ in range(2)])
        out = model(drug, geno)
        assert out["phenotype_category"].shape == (2, 5)


# --------------------------------------------------------------------------- #
# HGVS genotype tower: gene-identity embedding + LayerNorm
# --------------------------------------------------------------------------- #


def _hgvs_model(gene_vocab_size: int) -> PharmagenTwoTower:
    cfg = PharmagenConfig.from_dims(
        dims=GraphDims(drug_global=0, drug_admet=0, geno_function=0),
        drug_dim=61,
        geno_dim=45,
        axes=AXES,
        params=PARAMS,
        switches=SWITCHES,
        gene_vocab_size=gene_vocab_size,
    )
    return PharmagenTwoTower(cfg)


def _hgvs_graph(gene_token: int) -> Data:
    """Gene hub + two variants (the shape GenotypeGraphBuilder produces)."""
    g = Data(
        x=torch.randn(3, 45),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_attr=torch.randn(6, 3),
    )
    g.gene_token = torch.tensor([gene_token, 0, 0])
    return g


class TestHgvsGenotypeTower:
    def test_gene_identity_changes_the_output(self) -> None:
        model = _hgvs_model(gene_vocab_size=5).eval()
        drug = Batch.from_data_list([_drug_graph(with_global=False)])
        graph = _hgvs_graph(2)
        other = graph.clone()
        other.gene_token = torch.tensor([3, 0, 0])
        a = model(drug, Batch.from_data_list([graph]))["phenotype_category"]
        b = model(drug, Batch.from_data_list([other]))["phenotype_category"]
        assert not torch.allclose(a, b)

    def test_variant_nodes_get_no_gene_vector(self) -> None:
        model = _hgvs_model(gene_vocab_size=5)
        assert torch.equal(
            model.gene_embedding.weight[0], torch.zeros(PARAMS.hidden_dim)
        )

    def test_missing_gene_token_raises(self) -> None:
        model = _hgvs_model(gene_vocab_size=5)
        drug = Batch.from_data_list([_drug_graph(with_global=False)])
        graph = _hgvs_graph(2)
        del graph.gene_token
        with pytest.raises(ValueError, match="gene_token"):
            model(drug, Batch.from_data_list([graph]))

    def test_single_node_genotype_is_not_erased(self) -> None:
        """LayerNorm keeps a lone gene hub informative (GraphNorm would zero it)."""
        model = _hgvs_model(gene_vocab_size=5).eval()
        drug = Batch.from_data_list([_drug_graph(with_global=False)])

        def lone(token: int) -> Batch:
            g = Data(
                x=torch.zeros(1, 45),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 3)),
            )
            g.gene_token = torch.tensor([token])
            return Batch.from_data_list([g])

        a = model(drug, lone(2))["phenotype_category"]
        b = model(drug, lone(3))["phenotype_category"]
        assert not torch.allclose(a, b)
