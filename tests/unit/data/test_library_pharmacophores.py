"""Tests for src.data.library.pharmacophore.

The schema dimensions are pinned explicitly (11/13/258): changing them silently
would invalidate any graph library or model built on the pharmacophore graphs.
The atom-order contract with the drug graph (``atom2feat`` indexes into it) is
checked against :func:`smiles_to_graph` on the same SMILES.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest
import torch

from src.data.library.builder import LibraryBuilder
from src.data.library.config import LibraryBuildConfig
from src.data.library.drugs import (
    DrugFailureCategory,
    DrugGraphBuilder,
    smiles_to_graph,
)
from src.data.library.manifest import BuildManifest
from src.data.library.pharmacophore import (
    _FAMILIES,
    _PAIR_INDEX,
    _TOPO_UPPER,
    PHARM_EDGE_DIM,
    PHARM_GLOBAL_DIM,
    PHARM_NODE_DIM,
    PharmacophoreGraphBuilder,
    _bin,
    _log_count,
    _microspecies,
    _prepare_mol,
    smiles_to_pharmacophore_graph,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torch_geometric.data import Data as PyGData

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
BENZENE = "c1ccccc1"
BUTANE = "CCCC"
AMMONIA = "N"  # exactly one feature (Donor) -> no pairs
METHANE = "C"  # no pharmacophore feature at all
SODIUM_ACETATE = "CC(=O)[O-].[Na+]"  # acetate + Na+ counterion

_N_FAM = len(_FAMILIES)
_N_TOPO = len(_TOPO_UPPER) + 1
_AROMATIC_COL_DRUG = 48  # is_aromatic column of the drug-graph node features


def _graph(smiles: str, *, strip_salts: bool = False) -> PyGData:
    g = smiles_to_pharmacophore_graph(smiles, strip_salts=strip_salts)
    assert g is not None
    return g


def _family(graph: PyGData, node: int) -> str:
    """Family name of a node, read off its one-hot (the 9th slot is 'other')."""
    idx = int(graph.x[node, : _N_FAM + 1].argmax())
    return _FAMILIES[idx] if idx < _N_FAM else "other"


def _distance(graph: PyGData, edge: int) -> int:
    """Topological distance of an edge, recovered from its scaled column."""
    return round(float(graph.edge_attr[edge, -1]) * 30.0)


def _write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["cid\tsmiles\tcmpd_name_cleaned"]
    lines += [f"{cid}\t{smiles}\t{name}" for cid, smiles, name in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Schema + helpers
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_dimensions_are_frozen(self) -> None:
        assert PHARM_NODE_DIM == 11
        assert PHARM_EDGE_DIM == 13
        assert PHARM_GLOBAL_DIM == 258

    def test_global_dim_layout(self) -> None:
        # 6 family counts + 21 unordered family pairs x the 12 edge distance bins.
        assert _FAMILIES == (
            "Donor",
            "Acceptor",
            "PosIonizable",
            "NegIonizable",
            "Aromatic",
            "Hydrophobe",
        )
        assert len(_PAIR_INDEX) == _N_FAM * (_N_FAM + 1) // 2 == 21
        assert PHARM_GLOBAL_DIM == _N_FAM + len(_PAIR_INDEX) * _N_TOPO

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, 0), (1, 1), (6, 6), (7, 7), (8, 7), (9, 8), (17, 10), (18, 11), (30, 11)],
    )
    def test_topological_bins(self, value: float, expected: int) -> None:
        assert _bin(value, _TOPO_UPPER) == expected

    def test_log_count_is_unbounded_and_strictly_monotonic(self) -> None:
        assert _log_count(0) == 0.0
        assert _log_count(4) == pytest.approx(math.log1p(4))
        # No cap: large counts stay distinguishable from each other.
        values = [_log_count(c) for c in (0, 1, 10, 50, 100, 1000)]
        assert values == sorted(set(values))


# --------------------------------------------------------------------------- #
# smiles_to_pharmacophore_graph
# --------------------------------------------------------------------------- #


class TestSmilesToPharmacophoreGraph:
    def test_aspirin_shapes(self) -> None:
        g = _graph(ASPIRIN)
        k = g.num_nodes
        assert k == 6
        assert g.x.shape == (k, PHARM_NODE_DIM)
        # Complete graph over the features, both directions.
        assert g.edge_index.shape == (2, k * (k - 1))
        assert g.edge_attr.shape == (k * (k - 1), PHARM_EDGE_DIM)
        assert g.global_feats.shape == (1, PHARM_GLOBAL_DIM)
        assert g.num_drug_atoms == 13
        assert g.atom2feat.shape[1] == 2

    def test_dtypes(self) -> None:
        g = _graph(ASPIRIN)
        assert g.x.dtype == torch.float
        assert g.edge_index.dtype == torch.long
        assert g.edge_attr.dtype == torch.float
        assert g.global_feats.dtype == torch.float
        assert g.atom2feat.dtype == torch.long

    def test_aspirin_families(self) -> None:
        # pH 7.4: the acid is a carboxylate (one anion group, both O acceptors,
        # no OH donor); the ring is one Aromatic feature.
        g = _graph(ASPIRIN)
        families = sorted(_family(g, i) for i in range(g.num_nodes))
        assert families == sorted(["Acceptor"] * 4 + ["Aromatic", "NegIonizable"])

    @pytest.mark.parametrize("bad", ["not_smiles!!!", "", "   ", None])
    def test_invalid_returns_none(self, bad: str | None) -> None:
        assert smiles_to_pharmacophore_graph(bad) is None  # type: ignore[arg-type]

    def test_no_feature_gives_empty_graph(self) -> None:
        g = _graph(METHANE)
        assert g.num_nodes == 0
        assert g.x.shape == (0, PHARM_NODE_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_attr.shape == (0, PHARM_EDGE_DIM)

    def test_empty_graph_has_the_same_attributes(self) -> None:
        # A featureless drug must batch with the others: zero global vector,
        # empty atom2feat, and its atom count.
        g = _graph(METHANE)
        assert torch.equal(g.global_feats, torch.zeros((1, PHARM_GLOBAL_DIM)))
        assert g.atom2feat.shape == (0, 2)
        assert g.atom2feat.dtype == torch.long
        assert g.num_drug_atoms == 1

    def test_single_feature_has_no_edges_but_globals(self) -> None:
        g = _graph(AMMONIA)
        assert g.num_nodes == 1
        assert _family(g, 0) == "Donor"
        assert g.edge_index.shape == (2, 0)
        assert g.edge_attr.shape == (0, PHARM_EDGE_DIM)
        assert g.global_feats.shape == (1, PHARM_GLOBAL_DIM)
        # Only the Donor count is set; no pair can exist.
        donor = _FAMILIES.index("Donor")
        assert float(g.global_feats[0, donor]) == pytest.approx(_log_count(1))
        assert float(g.global_feats.sum()) == pytest.approx(_log_count(1))

    def test_deterministic(self) -> None:
        a, b = _graph(CAFFEINE), _graph(CAFFEINE)
        for attr in ("x", "edge_index", "edge_attr", "global_feats", "atom2feat"):
            assert torch.equal(getattr(a, attr), getattr(b, attr)), attr


class TestNodeFeatures:
    def test_family_one_hot_is_exclusive(self) -> None:
        g = _graph(CAFFEINE)
        one_hot = g.x[:, : _N_FAM + 1]
        assert torch.equal(one_hot.sum(dim=1), torch.ones(g.num_nodes))

    def test_size_matches_atom2feat(self) -> None:
        g = _graph(ASPIRIN)
        sizes = torch.bincount(g.atom2feat[:, 1], minlength=g.num_nodes)
        expected = sizes.clamp(max=6).float() / 6.0
        assert torch.allclose(g.x[:, _N_FAM + 1], expected)

    def test_benzene_ring_is_aromatic_and_cyclic(self) -> None:
        g = _graph(BENZENE)
        assert [_family(g, i) for i in range(g.num_nodes)] == ["Aromatic"]
        aromatic_frac, ring_frac = g.x[:, _N_FAM + 2], g.x[:, _N_FAM + 3]
        assert torch.allclose(aromatic_frac, torch.ones(g.num_nodes))
        assert torch.allclose(ring_frac, torch.ones(g.num_nodes))

    def test_acyclic_features_have_zero_ring_fraction(self) -> None:
        g = _graph(BUTANE)
        assert torch.equal(g.x[:, _N_FAM + 2], torch.zeros(g.num_nodes))
        assert torch.equal(g.x[:, _N_FAM + 3], torch.zeros(g.num_nodes))

    def test_charge_is_bounded(self) -> None:
        charge = _graph(CAFFEINE).x[:, -1]
        assert torch.isfinite(charge).all()
        assert (charge.abs() <= 1.0).all()


def _families(smiles: str) -> list[str]:
    g = _graph(smiles)
    return sorted(_family(g, i) for i in range(g.num_nodes))


class TestRedundantFeatures:
    """One chemical group must not be counted as several overlapping features."""

    def test_fused_rings_keep_one_aromatic_per_ring(self) -> None:
        assert _families("c1ccc2ccccc2c1") == ["Aromatic", "Aromatic"]

    def test_lumped_groups_are_one_hydrophobe_each(self) -> None:
        # t-butyl + cyclohexyl: two lumped groups; their single-atom
        # Hydrophobe features are inside them.
        assert _families("CC(C)(C)C1CCCCC1") == ["Hydrophobe"] * 2

    def test_bonded_chain_hydrophobes_merge(self) -> None:
        assert _families(BUTANE) == ["Hydrophobe"]
        assert _families("CCCCCCO").count("Hydrophobe") == 1

    @pytest.mark.parametrize("smiles", [ASPIRIN, CAFFEINE, "OC(=O)CCS", "CC(=O)NO"])
    def test_only_the_six_families(self, smiles: str) -> None:
        # No LumpedHydrophobe / ZnBinder / 'other' node is ever emitted.
        assert set(_families(smiles)) <= set(_FAMILIES)


class TestMicrospecies:
    """Features describe the dominant pH 7.4 protonation state."""

    def test_acid_written_any_way_gives_one_graph(self) -> None:
        ref = _graph("CC(=O)O")
        for smiles in ("CC(=O)[O-]", "CC(=O)[O-].[Na+]"):
            other = _graph(smiles, strip_salts=True)
            assert torch.equal(other.x, ref.x), smiles
            assert torch.equal(other.global_feats, ref.global_feats), smiles

    def test_carboxylate_group_has_no_donor(self) -> None:
        g = _graph("CC(=O)O")
        assert _families("CC(=O)O") == ["Acceptor", "Acceptor", "NegIonizable"]
        neg = next(i for i in range(g.num_nodes) if _family(g, i) == "NegIonizable")
        atoms = sorted(a for a, f in g.atom2feat.tolist() if f == neg)
        assert atoms == [1, 2, 3]  # C, =O, O-

    def test_aliphatic_amine_is_a_protonated_donor(self) -> None:
        assert _families("CCN(CC)CC").count("PosIonizable") == 1
        assert "Donor" in _families("CCN(CC)CC")  # N+-H

    def test_aniline_and_neutral_tertiary_n_are_not_donors(self) -> None:
        families = _families("CN(C)c1ccccc1")
        assert "Donor" not in families
        assert "PosIonizable" not in families

    def test_imidazole_is_neutral_and_its_bare_n_no_donor(self) -> None:
        families = _families("c1c[nH]cn1")
        assert "PosIonizable" not in families
        assert families.count("Donor") == 1  # only the N-H

    def test_piperazine_protonates_once(self) -> None:
        assert _families("CN1CCNCC1").count("PosIonizable") == 1

    def test_guanidines(self) -> None:
        assert _families("CN(C)C(=N)N=C(N)N").count("PosIonizable") == 1  # metformin
        assert "PosIonizable" not in _families("CN=C(NC#N)NC")  # cyanoguanidine

    @pytest.mark.parametrize(
        ("smiles", "anion_atoms"),
        [
            ("Cc1nnn[nH]1", 1),  # tetrazole: the N-
            ("CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O", 2),  # warfarin enolate
        ],
    )
    def test_other_acids_are_anions(self, smiles: str, anion_atoms: int) -> None:
        g = _graph(smiles)
        neg = [i for i in range(g.num_nodes) if _family(g, i) == "NegIonizable"]
        assert len(neg) == 1
        sizes = torch.bincount(g.atom2feat[:, 1], minlength=g.num_nodes)
        assert int(sizes[neg[0]]) == anion_atoms

    def test_nitro_and_n_oxide_are_not_ions(self) -> None:
        for smiles in ("O=[N+]([O-])c1ccccc1", "C[N+](C)([O-])C"):
            families = _families(smiles)
            assert "PosIonizable" not in families, smiles
            assert "NegIonizable" not in families, smiles

    def test_atom_order_is_unchanged(self) -> None:
        for smiles in (ASPIRIN, CAFFEINE, "CN1CCNCC1", "Cc1nnn[nH]1"):
            mol = _prepare_mol(smiles, strip_salts=False)
            assert mol is not None
            micro = _microspecies(mol)
            assert [a.GetAtomicNum() for a in micro.GetAtoms()] == [
                a.GetAtomicNum() for a in mol.GetAtoms()
            ]


class TestEdgeFeatures:
    def test_bins_are_one_hot_and_consistent_with_distance(self) -> None:
        g = _graph(ASPIRIN)
        bins = g.edge_attr[:, :_N_TOPO]
        assert torch.equal(bins.sum(dim=1), torch.ones(g.edge_attr.shape[0]))
        for e in range(g.edge_attr.shape[0]):
            assert int(bins[e].argmax()) == _bin(_distance(g, e), _TOPO_UPPER)

    def test_symmetric(self) -> None:
        g = _graph(CAFFEINE)
        attrs = {
            (i, j): g.edge_attr[e] for e, (i, j) in enumerate(g.edge_index.t().tolist())
        }
        for (i, j), attr in attrs.items():
            assert torch.equal(attr, attrs[(j, i)])

    def test_no_self_loops(self) -> None:
        g = _graph(ASPIRIN)
        assert (g.edge_index[0] != g.edge_index[1]).all()

    def test_overlapping_features_have_zero_distance(self) -> None:
        # Methanol: the hydroxyl O is both Donor and Acceptor.
        g = _graph("CO")
        assert sorted(_family(g, i) for i in range(g.num_nodes)) == [
            "Acceptor",
            "Donor",
        ]
        assert g.edge_attr.shape[0] == 2
        assert (g.edge_attr[:, 0] == 1.0).all()  # bin "d <= 0"
        assert (g.edge_attr[:, -1] == 0.0).all()

    def test_distance_uses_closest_atom_pair(self) -> None:
        # beta-alanine zwitterion: N+ to the carboxylate group (C,O,O) is 3 bonds
        # via its carbon, 4 via either oxygen; the closest pair counts.
        g = _graph("NCCC(=O)O")
        pos = next(i for i in range(g.num_nodes) if _family(g, i) == "PosIonizable")
        neg = next(i for i in range(g.num_nodes) if _family(g, i) == "NegIonizable")
        edges = g.edge_index.t().tolist()
        assert _distance(g, edges.index([pos, neg])) == 3

    def test_disconnected_fragments_are_capped(self) -> None:
        # Without salt stripping RDKit puts 1e8 between fragments; the edge must be
        # clamped to the cap (scaled 1.0) and land in the overflow bin.
        g = _graph("N.N", strip_salts=False)
        assert g.num_nodes == 2
        assert (g.edge_attr[:, -1] == 1.0).all()
        assert (g.edge_attr[:, _N_TOPO - 1] == 1.0).all()


class TestGlobalFeatures:
    def test_family_counts(self) -> None:
        g = _graph(ASPIRIN)
        counts = [0] * _N_FAM
        for i in range(g.num_nodes):
            counts[_FAMILIES.index(_family(g, i))] += 1
        expected = torch.tensor([_log_count(c) for c in counts])
        assert torch.allclose(g.global_feats[0, :_N_FAM], expected)

    def test_pair_histogram_matches_edges(self) -> None:
        g = _graph(CAFFEINE)
        hist = [0] * (len(_PAIR_INDEX) * _N_TOPO)
        for e, (i, j) in enumerate(g.edge_index.t().tolist()):
            if i > j:  # each unordered pair once
                continue
            fi = _FAMILIES.index(_family(g, i))
            fj = _FAMILIES.index(_family(g, j))
            pair = _PAIR_INDEX[(min(fi, fj), max(fi, fj))]
            hist[pair * _N_TOPO + _bin(_distance(g, e), _TOPO_UPPER)] += 1
        assert sum(hist) == g.num_nodes * (g.num_nodes - 1) // 2
        expected = torch.tensor([_log_count(c) for c in hist])
        assert torch.allclose(g.global_feats[0, _N_FAM:], expected)

    def test_range(self) -> None:
        g = _graph(CAFFEINE)
        assert torch.isfinite(g.global_feats).all()
        assert (g.global_feats >= 0).all()

    def test_pair_distances_4_and_6_are_told_apart(self) -> None:
        # The former coarse histogram put both in one "4-6" bin.
        donor = _FAMILIES.index("Donor")
        slot = _N_FAM + _PAIR_INDEX[(donor, donor)] * _N_TOPO
        near = _graph("OCCCO").global_feats[0, slot : slot + _N_TOPO]
        far = _graph("OCCCCCO").global_feats[0, slot : slot + _N_TOPO]
        assert int(near.argmax()) == _bin(4, _TOPO_UPPER)
        assert int(far.argmax()) == _bin(6, _TOPO_UPPER)
        assert not torch.equal(near, far)

    def test_similar_drugs_closer_than_dissimilar(self) -> None:
        aspirin = _graph(ASPIRIN)
        salicylic = _graph("Oc1ccccc1C(=O)O")
        butane = _graph(BUTANE)
        cos = torch.nn.functional.cosine_similarity
        near = float(cos(aspirin.global_feats, salicylic.global_feats))
        far = float(cos(aspirin.global_feats, butane.global_feats))
        assert near > far


# --------------------------------------------------------------------------- #
# Contract with the drug graph
# --------------------------------------------------------------------------- #


class TestAtomOrderContract:
    @pytest.mark.parametrize("smiles", [ASPIRIN, CAFFEINE, SODIUM_ACETATE])
    @pytest.mark.parametrize("strip", [False, True])
    def test_num_atoms_matches_drug_graph(self, smiles: str, strip: bool) -> None:
        pharm = _graph(smiles, strip_salts=strip)
        drug = smiles_to_graph(smiles, strip_salts=strip)
        assert drug is not None
        assert pharm.num_drug_atoms == drug.num_nodes
        assert int(pharm.atom2feat[:, 0].max()) < drug.num_nodes
        assert int(pharm.atom2feat[:, 1].max()) < pharm.num_nodes

    def test_every_feature_has_atoms(self) -> None:
        g = _graph(CAFFEINE)
        assert set(g.atom2feat[:, 1].tolist()) == set(range(g.num_nodes))

    def test_aromatic_feature_atoms_are_aromatic_in_drug_graph(self) -> None:
        pharm = _graph(ASPIRIN)
        drug = smiles_to_graph(ASPIRIN)
        assert drug is not None
        aromatic = {
            i for i in range(pharm.num_nodes) if _family(pharm, i) == "Aromatic"
        }
        assert aromatic
        for atom, feat in pharm.atom2feat.tolist():
            if feat in aromatic:
                assert drug.x[atom, _AROMATIC_COL_DRUG] == 1.0

    def test_salt_stripping_drops_counterion(self) -> None:
        stripped = _graph(SODIUM_ACETATE, strip_salts=True)
        kept = _graph(SODIUM_ACETATE, strip_salts=False)
        bare = _graph("CC(=O)[O-]")
        assert stripped.num_drug_atoms == bare.num_drug_atoms == 4
        assert kept.num_drug_atoms == 5
        assert torch.equal(stripped.x[:, : _N_FAM + 1], bare.x[:, : _N_FAM + 1])

    def test_salt_stripped_fragment_keeps_ring_features(self) -> None:
        # _largest_fragment skips sanitisation; ring perception must be restored
        # or the aromatic/ring SMARTS silently miss.
        salt = _graph(f"{BENZENE}.[Na+]", strip_salts=True)
        bare = _graph(BENZENE)
        assert salt.num_nodes == bare.num_nodes
        assert torch.allclose(salt.x, bare.x)


# --------------------------------------------------------------------------- #
# PharmacophoreGraphBuilder
# --------------------------------------------------------------------------- #


class TestPharmacophoreGraphBuilder:
    def test_builds_from_tsv(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin"), ("2", CAFFEINE, "caffeine")])
        out = tmp_path / "pharmacophores"
        manifest = BuildManifest()
        builder = PharmacophoreGraphBuilder(out, failures_log=tmp_path / "fail.log")

        built, skipped, failed = builder.build(tsv, manifest=manifest)

        assert (built, skipped, failed) == (2, 0, 0)
        assert {p.name for p in out.glob("*.pt")} == {"1_aspirin.pt", "2_caffeine.pt"}
        assert sorted(manifest.completed_pharm_cids) == [1, 2]
        assert manifest.completed_drug_cids == []  # pharm never marks drug CIDs
        assert (manifest.stats.pharm_built, manifest.stats.pharm_failed) == (2, 0)

    def test_saved_graph_metadata(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("2244", ASPIRIN, "aspirin")])
        out = tmp_path / "pharmacophores"
        PharmacophoreGraphBuilder(out).build(tsv, manifest=BuildManifest())

        g = torch.load(out / "2244_aspirin.pt", weights_only=False)
        assert g.cid == 2244
        assert g.name == "aspirin"
        assert g.smiles == ASPIRIN
        reference = _graph(ASPIRIN, strip_salts=True)
        assert torch.equal(g.x, reference.x)
        assert torch.equal(g.edge_attr, reference.edge_attr)
        assert torch.equal(g.global_feats, reference.global_feats)

    def test_file_stem_matches_drug_graph(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("5", ASPIRIN, "weird:name/x")])
        DrugGraphBuilder(tmp_path / "drugs").build(tsv, manifest=BuildManifest())
        PharmacophoreGraphBuilder(tmp_path / "pharm").build(
            tsv, manifest=BuildManifest()
        )
        drug_names = {p.name for p in (tmp_path / "drugs").glob("*.pt")}
        pharm_names = {p.name for p in (tmp_path / "pharm").glob("*.pt")}
        assert drug_names == pharm_names == {"5_weird_name_x.pt"}

    def test_missing_name_falls_back_to_cid(self, tmp_path: Path) -> None:
        js = tmp_path / "drugs.json"
        js.write_text(json.dumps({" 7": ASPIRIN}), encoding="utf-8")
        out = tmp_path / "pharm"
        built, _, failed = PharmacophoreGraphBuilder(out).build(
            js, manifest=BuildManifest()
        )
        assert (built, failed) == (1, 0)
        assert (out / "7_cid7.pt").exists()

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin")])
        out = tmp_path / "pharm"
        PharmacophoreGraphBuilder(out).build(tsv, manifest=BuildManifest())

        manifest = BuildManifest()
        result = PharmacophoreGraphBuilder(out).build(tsv, manifest=manifest)

        assert result == (0, 1, 0)
        assert manifest.completed_pharm_cids == [1]  # skipped still counts as done
        assert manifest.stats.pharm_skipped == 1

    def test_force_rebuilds(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin")])
        out = tmp_path / "pharm"
        PharmacophoreGraphBuilder(out).build(tsv, manifest=BuildManifest())
        result = PharmacophoreGraphBuilder(out, force=True).build(
            tsv, manifest=BuildManifest()
        )
        assert result == (1, 0, 0)

    def test_failure_counter_by_nature(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(
            tsv,
            [
                ("1", ASPIRIN, "ok"),
                ("xyz", ASPIRIN, "bad_cid"),  # non_integer_cid
                ("3", "", "no_smiles"),  # missing_smiles
                ("4", "Q##notReal", "bad_smiles"),  # invalid_smiles
                ("5", METHANE, "methane"),  # empty_graph
            ],
        )
        log = tmp_path / "logs" / "library" / "pharm_fail.log"
        manifest = BuildManifest()
        builder = PharmacophoreGraphBuilder(tmp_path / "pharm", failures_log=log)

        built, skipped, failed = builder.build(tsv, manifest=manifest)

        assert (built, skipped, failed) == (1, 0, 4)
        for category in (
            DrugFailureCategory.NON_INTEGER_CID,
            DrugFailureCategory.MISSING_SMILES,
            DrugFailureCategory.INVALID_SMILES,
            DrugFailureCategory.EMPTY_GRAPH,
        ):
            assert builder.failure_counts[category] == 1, category
        assert not (tmp_path / "pharm" / "5_methane.pt").exists()

        # A non-integer CID has no key to record; the rest land as "pharm:<cid>".
        assert set(manifest.failed) == {"pharm:3", "pharm:4", "pharm:5"}
        assert manifest.stats.pharm_failed == 4

        contents = log.read_text(encoding="utf-8")
        assert "Build run started" in contents
        for nature in ("non_integer_cid", "missing_smiles", "invalid_smiles"):
            assert f"nature={nature}" in contents
        assert "nature=empty_graph" in contents
        assert "Build run finished" in contents

    def test_save_error_is_categorised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin")])

        def _boom(*_args: object, **_kwargs: object) -> None:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr("src.data.library.pharmacophore.torch.save", _boom)
        builder = PharmacophoreGraphBuilder(tmp_path / "pharm")
        manifest = BuildManifest()

        assert builder.build(tsv, manifest=manifest) == (0, 0, 1)
        assert builder.failure_counts[DrugFailureCategory.SAVE_ERROR] == 1
        assert "disk full" in manifest.failed["pharm:1"]

    def test_salt_stripping_flag(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", SODIUM_ACETATE, "na_acetate")])
        stripped_dir, kept_dir = tmp_path / "stripped", tmp_path / "kept"
        PharmacophoreGraphBuilder(stripped_dir).build(tsv, manifest=BuildManifest())
        PharmacophoreGraphBuilder(kept_dir, strip_salts=False).build(
            tsv, manifest=BuildManifest()
        )
        stripped = torch.load(stripped_dir / "1_na_acetate.pt", weights_only=False)
        kept = torch.load(kept_dir / "1_na_acetate.pt", weights_only=False)
        assert stripped.num_drug_atoms == 4
        assert kept.num_drug_atoms == 5

    def test_missing_catalog_raises(self, tmp_path: Path) -> None:
        builder = PharmacophoreGraphBuilder(tmp_path / "pharm")
        with pytest.raises(FileNotFoundError, match="Drug catalog not found"):
            builder.build(tmp_path / "nope.tsv", manifest=BuildManifest())


# --------------------------------------------------------------------------- #
# Wiring: manifest, config, orchestrator, CLI
# --------------------------------------------------------------------------- #


class TestManifestPharm:
    def test_mark_done_is_idempotent(self) -> None:
        m = BuildManifest()
        m.mark_pharm_done(2244)
        m.mark_pharm_done(2244)
        assert m.completed_pharm_cids == [2244]

    def test_failed_key_does_not_clash_with_drug(self) -> None:
        m = BuildManifest()
        m.mark_drug_failed(1, "drug reason")
        m.mark_pharm_failed(1, "pharm reason")
        assert m.failed == {"drug:1": "drug reason", "pharm:1": "pharm reason"}

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        m = BuildManifest()
        m.mark_pharm_done(7)
        m.stats.pharm_built = 1
        m.save(path)
        loaded = BuildManifest.load_or_empty(path)
        assert loaded.completed_pharm_cids == [7]
        assert loaded.stats.pharm_built == 1

    def test_manifest_without_pharm_fields_loads(self, tmp_path: Path) -> None:
        # Manifests written before the pharmacophore pipeline existed.
        path = tmp_path / "manifest.json"
        path.write_text('{"completed_drug_cids": [1], "stats": {}}', encoding="utf-8")
        loaded = BuildManifest.load_or_empty(path)
        assert loaded.completed_drug_cids == [1]
        assert loaded.completed_pharm_cids == []
        assert loaded.stats.pharm_built == 0


def _config(tmp_path: Path, drugs_tsv: Path, **overrides: object) -> LibraryBuildConfig:
    base: dict[str, object] = {
        "variants_tsv": tmp_path / "snp.tsv",
        "drugs_tsv": drugs_tsv,
        "pgx_dir": tmp_path / "pgx",
        "kg_dir": tmp_path / "kg",
        "library_root": tmp_path / "library",
        "admet_cache": tmp_path / "admet.parquet",
        "star_alleles_tsv": tmp_path / "star.tsv",
        "skip_drugs": True,
        "skip_genes": True,
        "skip_admet": True,
    }
    return LibraryBuildConfig.model_validate(base | overrides)


class TestLibraryBuilderWiring:
    def test_config_paths(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, tmp_path / "drugs.tsv")
        assert cfg.pharm_out == tmp_path / "library" / "pharmacophores"
        assert cfg.skip_pharm is False
        cfg.ensure_outputs()
        assert cfg.pharm_out.is_dir()

    def test_from_settings_wires_pharm_options(self) -> None:
        cfg = LibraryBuildConfig.from_settings(skip_pharm=True)
        assert cfg.skip_pharm is True
        assert cfg.log_pharm_failures_path is not None
        assert cfg.log_pharm_failures_path.name.startswith("pharm_build_failures_")

    def test_run_builds_pharmacophores(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin"), ("2", METHANE, "methane")])
        cfg = _config(tmp_path, tsv, log_pharm_failures_path=tmp_path / "pf.log")

        summary = LibraryBuilder(cfg).run()

        assert summary.pharm_built == 1
        assert summary.pharm_skipped == 0
        assert summary.pharm_failed == 1
        assert (cfg.pharm_out / "1_aspirin.pt").exists()
        manifest = BuildManifest.load_or_empty(cfg.manifest_path)
        assert manifest.completed_pharm_cids == [1]
        assert "pharm:2" in manifest.failed

    def test_run_skip_pharm(self, tmp_path: Path) -> None:
        tsv = tmp_path / "drugs.tsv"
        _write_tsv(tsv, [("1", ASPIRIN, "aspirin")])
        cfg = _config(tmp_path, tsv, skip_pharm=True)

        summary = LibraryBuilder(cfg).run()

        assert (summary.pharm_built, summary.pharm_failed) == (0, 0)
        assert list(cfg.pharm_out.glob("*.pt")) == []

    def test_cli_flag(self) -> None:
        from src.data.library.__main__ import _build_parser

        parser = _build_parser()
        assert parser.parse_args(["--skip-pharm"]).skip_pharm is True
        assert parser.parse_args([]).skip_pharm is False
