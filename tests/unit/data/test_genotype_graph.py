"""HGVS genotype graph: features, HGVS linking, dosage, graph layout, catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch

from src.data.cache import PackedGraphs
from src.data.genotype import (
    GENO_EDGE_DIM,
    HGVS_NODE_DIM,
    GenotypeGraphBuilder,
    HgvsCatalog,
    VariantInfo,
    build_gene_vocab,
    build_hgvs_catalog,
    gene_vocab_size,
)
from src.data.genotype.features import (
    EDIT_KINDS,
    LEVELS,
    NODE_TYPES,
    REGIONS,
    Dosage,
    parse_hgvs,
    variant_node_vector,
)
from src.data.genotype.graph import PAD_TOKEN, UNK_TOKEN, AlleleCall

if TYPE_CHECKING:
    from pathlib import Path

G_A = "NC_000010.11:g.94781859G>A"  # CYP2C19*2 defining SNV
G_T = "NC_000010.11:g.94781859G>T"  # other alternative at the same site
G_FAR = "NC_000010.11:g.94842866A>G"

_OFFSETS = {
    "edit": len(NODE_TYPES),
    "level": len(NODE_TYPES) + len(EDIT_KINDS),
    "region": len(NODE_TYPES) + len(EDIT_KINDS) + len(LEVELS),
    "change": len(NODE_TYPES) + len(EDIT_KINDS) + len(LEVELS) + len(REGIONS),
}


def _vec(hgvs: str, **kwargs) -> list[float]:
    variant = parse_hgvs(hgvs)
    assert variant is not None
    return variant_node_vector(variant, variant.changes[0], **kwargs)


def _block(vec: list[float], name: str, options: tuple[str, ...]) -> str:
    start = _OFFSETS[name]
    return options[vec[start : start + len(options)].index(1.0)]


@pytest.fixture
def catalog() -> HgvsCatalog:
    return HgvsCatalog(
        variants={
            G_A: VariantInfo("CYP2C19", ("missense_variant",), None),
            G_FAR: VariantInfo("CYP2C19"),
        },
        labels={"rs4244285": (G_A, G_T), "CYP2C19*2": (G_A, G_FAR)},
        annotations={"42": (G_A,)},
    )


@pytest.fixture
def builder(catalog: HgvsCatalog) -> GenotypeGraphBuilder:
    return GenotypeGraphBuilder(catalog, build_gene_vocab(["CYP2C19"] * 2 + ["XRCC1"]))


# ----------------------------------------------------------------- features


class TestFeatures:
    def test_dimension(self) -> None:
        assert len(_vec(G_A)) == HGVS_NODE_DIM == 45

    @pytest.mark.parametrize(
        ("hgvs", "edit", "level", "region"),
        [
            ("NM_000769.4:c.681G>A", "substitution", "transcript", "cds"),
            ("NM_000769.4:c.332-23A>G", "substitution", "transcript", "intron"),
            ("NM_000769.4:c.-806C>T", "substitution", "transcript", "utr5"),
            ("NM_000769.4:c.*30dup", "duplication", "transcript", "utr3"),
            ("NC_000010.11:g.100_101insA", "insertion", "genomic", "unknown"),
            ("NC_000010.11:g.100_110inv", "inversion", "genomic", "unknown"),
            ("NP_000760.1:p.(Trp212Ter)", "substitution", "protein", "unknown"),
        ],
    )
    def test_reads_the_notation(self, hgvs, edit, level, region) -> None:
        vec = _vec(hgvs)
        assert _block(vec, "edit", EDIT_KINDS) == edit
        assert _block(vec, "level", LEVELS) == level
        assert _block(vec, "region", REGIONS) == region

    def test_transition_vs_transversion(self) -> None:
        c = _OFFSETS["change"]
        assert _vec(G_A)[c : c + 2] == [1.0, 0.0]  # G>A purine→purine
        assert _vec(G_T)[c : c + 2] == [0.0, 1.0]

    def test_frameshift_like_indel(self) -> None:
        c = _OFFSETS["change"] + 3
        assert _vec("NM_000769.4:c.100del")[c] == 1.0
        assert _vec("NM_000769.4:c.100_102del")[c] == 0.0

    def test_repeat_length_tracks_copy_number(self) -> None:
        c = _OFFSETS["change"] + 2
        short = _vec("NC_000002.12:g.85554455_85554456T[5]GTT[1]")
        long = _vec("NC_000002.12:g.85554455_85554456T[7]GTT[1]")
        assert _block(short, "edit", EDIT_KINDS) == "repeat"
        assert short[c] < long[c]

    def test_dosage_block(self) -> None:
        assert _vec(G_A, dosage=Dosage(2))[-2:] == [1.0, 1.0]
        assert _vec(G_A)[-2:] == [0.0, 0.0]

    def test_unparseable_is_none(self) -> None:
        assert parse_hgvs("rs4244285") is None
        assert parse_hgvs("NC_1:g.nonsense") is None


# -------------------------------------------------------------- allele calls


@pytest.mark.parametrize(
    ("text", "bases", "stars"),
    [
        ("CT", ("C", "T"), ()),
        ("C/T", ("C", "T"), ()),
        ("A", ("A",), ()),
        ("*1/*4", (), ("*1", "*4")),
    ],
)
def test_allele_call_parse(text, bases, stars) -> None:
    call = AlleleCall.parse(text)
    assert call is not None
    assert (call.bases, call.stars) == (bases, stars)


@pytest.mark.parametrize("text", [None, "", "CC + CT", "del/del"])
def test_allele_call_ambiguous(text) -> None:
    assert AlleleCall.parse(text) is None


# ----------------------------------------------------------------- linking


class TestLinking:
    def test_rsid_narrowed_by_observed_alt(self, builder) -> None:
        g = builder.build("CYP2C19", "rs4244285", "AG")
        assert g.num_nodes == 2  # gene + the observed alternative only
        assert g.x[1, -1] == 1.0  # dosage known
        assert g.x[1, -2] == 0.5  # one copy of A

    def test_rsid_without_call_keeps_all_alternatives(self, builder) -> None:
        g = builder.build("CYP2C19", "rs4244285")
        assert g.num_nodes == 3  # gene + both alternatives

    def test_annotation_link_wins_over_rsid(self, builder) -> None:
        builder.build("CYP2C19", "rs4244285", None, 42)
        assert builder.stats["variants_from_annotation"] == 1
        assert builder.stats["variants_from_rsid"] == 0

    def test_explicit_hgvs_token(self, builder) -> None:
        g = builder.build("CYP2C19", f"{G_FAR}|REF_SEQ")
        assert g.num_nodes == 2
        assert builder.stats["variants_from_hgvs"] == 1

    def test_star_diplotype_expands_with_copies(self, builder) -> None:
        g = builder.build("CYP2C19", "REF_SEQ", "*2/*2")
        assert g.num_nodes == 3  # gene + the two *2 variants
        assert g.x[1:, -2:].tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_unlinked_genotype_is_reference(self, builder) -> None:
        g = builder.build("XRCC1", "Gene_Deletion")
        assert g.num_nodes == 1
        assert g.edge_index.shape == (2, 0)


# ------------------------------------------------------------------- graph


class TestGraph:
    def test_layout_and_attrs(self, builder) -> None:
        g = builder.build("CYP2C19", "REF_SEQ", "*2/*2")
        assert g.x.shape == (3, HGVS_NODE_DIM)
        assert g.x[0, 0] == 1.0  # gene hub first
        assert g.gene_token.tolist()[1:] == [PAD_TOKEN, PAD_TOKEN]
        assert g.edge_attr.shape[1] == GENO_EDGE_DIM
        # 2 membership pairs + 1 adjacency pair, all bidirectional.
        assert g.edge_index.shape[1] == 6
        adjacency = g.edge_attr[g.edge_attr[:, 1] == 1.0]
        assert len(adjacency) == 2
        assert 0.0 < adjacency[0, 2].item() < 1.0  # log genomic distance
        assert g.geno_function.shape == (1, 6)

    def test_multi_gene_row_has_one_hub_per_gene(self, builder) -> None:
        g = builder.build("CYP2C19|XRCC1", G_A)
        assert g.x[:2, 0].tolist() == [1.0, 1.0]
        tokens = g.gene_token.tolist()
        assert tokens[0] != UNK_TOKEN  # CYP2C19 in vocab
        assert tokens[1] == UNK_TOKEN  # XRCC1 seen once → UNK
        # G_A belongs to CYP2C19 only → linked to hub 0 alone.
        assert sorted(g.edge_index[0].tolist()) == [0, 2]


def test_gene_vocab() -> None:
    vocab = build_gene_vocab(["A", "A|B", "C", "C"], min_count=2)
    assert vocab == {"A": 2, "C": 3}
    assert gene_vocab_size(vocab) == 4
    assert gene_vocab_size({}) == UNK_TOKEN + 1


# ------------------------------------------------------------------- catalog


def test_catalog_round_trip(catalog, tmp_path: Path) -> None:
    path = tmp_path / "catalog.pt"
    catalog.save(path)
    loaded = HgvsCatalog.load(path)
    assert loaded.labels == catalog.labels
    assert loaded.info(G_A).gene == "CYP2C19"
    assert loaded.variants_of("CYP2C19") == [G_A, G_FAR]


def test_build_catalog_from_kg_exports(tmp_path: Path) -> None:
    kg = tmp_path / "kg"
    kg.mkdir()
    pl.DataFrame(
        {
            "label": ["rs4244285", "rs4244285", "CYP2C19*2", "CYP2C19*1"],
            "kind": ["rsid", "rsid", "star", "star"],
            "hgvs_g": [G_A, G_T, G_A, None],
        }
    ).write_parquet(kg / "allele_variants.parquet")
    pl.DataFrame({"annotation_id": ["42"], "hgvs_g": [G_A]}).write_parquet(
        kg / "annotation_variants.parquet"
    )
    catalog = build_hgvs_catalog(
        snp_table=tmp_path / "missing.tsv",
        pharmvar_dir=tmp_path / "missing",
        kg_dir=kg,
        star_alleles_tsv=tmp_path / "missing.tsv",
    )
    assert catalog.labels["rs4244285"] == (G_A, G_T)
    assert catalog.labels["CYP2C19*2"] == (G_A,)
    assert "CYP2C19*1" not in catalog.labels
    assert catalog.annotations == {"42": (G_A,)}
    assert catalog.info(G_A).gene == "CYP2C19"  # from the star label


# ------------------------------------------------------------------- packing


def test_packed_graphs_round_trip(builder) -> None:
    graphs = [
        builder.build("CYP2C19", "REF_SEQ", "*2/*2"),
        builder.build("XRCC1", "Gene_Deletion"),
        builder.build("CYP2C19", "rs4244285"),
    ]
    packed = PackedGraphs(
        graphs, node_keys=("x", "gene_token"), graph_keys=("geno_function",)
    )
    assert len(packed) == 3
    assert len(packed.tensors) == 4  # few storages, whatever the row count
    for original, restored in zip(graphs, (packed[i] for i in range(3)), strict=True):
        assert torch.equal(original.x, restored.x)
        assert torch.equal(original.edge_index, restored.edge_index)
        assert torch.equal(original.edge_attr, restored.edge_attr)
        assert torch.equal(original.gene_token, restored.gene_token)
        assert torch.equal(original.geno_function, restored.geno_function)
