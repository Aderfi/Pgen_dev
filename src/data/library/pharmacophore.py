"""Pharmacophore graph builder — SMILES → PyG ``Data`` feature graph.

Companion of :mod:`src.data.library.drugs`. Each drug yields a second graph whose
nodes are *pharmacophore features* (not atoms) and whose edges carry the
topological distance between features. It is keyed by the same ``cid`` and file
name as the drug graph, so any drug↔gene table joins on ``cid`` unchanged.

Atom order is guaranteed identical to the drug graph (same parse, same
``_largest_fragment`` salt stripping), so ``atom2feat`` indexes into it.

Features are perceived on the molecule's pH 7.4 microspecies (rule-based,
``_microspecies``; atom order unchanged) and normalised to six non-overlapping
families (``_normalise_features``), so a group is one feature whatever way its
SMILES was written.

    Node features (11), one node per pharmacophore feature:
        family one-hot[Donor,Acceptor,PosIonizable,NegIonizable,
                       Aromatic,Hydrophobe,other]                            (7)
        n_atoms in the feature / 6 (clamped)                                 (1)
        aromatic fraction, ring fraction of member atoms                     (2)
        mean Gasteiger charge of member atoms                                (1)
    Edge features (13), complete graph over the features:
        topological-distance bin one-hot (closest atom pair, 11 bins+overflow)(12)
        distance / 30 (clamped)                                              (1)
    Global features (258), attached as ``global_feats`` [1, 258]:
        6 family counts + 21 family-pair × 12 distance-bin counts (the edge
        bins above), as log1p(count), unbounded. Every graph carries it, an
        empty one as zeros (all-zero family counts = no feature).
        Not standardised here: a consumer must z-score it with statistics fitted
        on its training drugs only (clip, drop near-empty columns), since most
        drugs share donor/acceptor/aromatic counts and the raw vectors of
        unrelated drugs are close.
    Extra attributes:
        atom2feat  [M, 2]  (atom index in the drug graph, feature node index)
        num_drug_atoms     atoms of the (salt-stripped) molecule
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch_geometric.data import Data as PyGData
from tqdm.auto import tqdm

from src.data.library.drugs import (
    DrugFailureCategory,
    _build_file_logger,
    _gasteiger_charge,
    _largest_fragment,
    _one_hot,
    load_drug_records,
    safe_filename,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.data.library.drugs import DrugRow
    from src.data.library.manifest import BuildManifest

logger = logging.getLogger(__name__)

_FAILURE_LOGGER_NAME = "Pharmagen.library.pharm_failures"

# Six families (the CATS / ErG / vendor vocabulary), perceived with RDKit's
# BaseFeatures.fdef and normalised by ``_normalise_features``. Excluded volume
# (MOE "Excl", LigandScout "XVOL") is deliberately absent: it marks space taken
# by the *receptor* (from a protein structure or a 3D ligand alignment) and
# cannot be perceived from one drug's SMILES, so it would be an always-empty
# family here.
_FAMILIES: tuple[str, ...] = (
    "Donor",
    "Acceptor",
    "PosIonizable",
    "NegIonizable",
    "Aromatic",
    "Hydrophobe",
)
_FAMILY_INDEX = {f: i for i, f in enumerate(_FAMILIES)}
_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)

# Topological-distance bins (upper bound, inclusive); one extra overflow slot.
_TOPO_UPPER: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 17)
_DIST_CAP = 30.0
_N_TOPO_BINS = len(_TOPO_UPPER) + 1  # 11 bins + overflow
_PAIR_INDEX = {
    pair: i
    for i, pair in enumerate(combinations_with_replacement(range(len(_FAMILIES)), 2))
}

PHARM_NODE_DIM: int = len(_FAMILIES) + 1 + 4  # 11
PHARM_EDGE_DIM: int = _N_TOPO_BINS + 1  # 13
# Same 12 distance bins as the edges: the former 4 coarse bins (<=3, 4-6, 7-10,
# >=11) put most pairs of a drug in one bin and separated drug classes worse.
PHARM_GLOBAL_DIM: int = len(_FAMILIES) + len(_PAIR_INDEX) * _N_TOPO_BINS  # 258


def _bin(value: float, uppers: tuple[int, ...]) -> int:
    for i, upper in enumerate(uppers):
        if value <= upper:
            return i
    return len(uppers)


def _log_count(count: int) -> float:
    """log1p, unbounded: no cap, so large counts still differ from each other."""
    return math.log1p(count)


def _prepare_mol(smiles: str, *, strip_salts: bool) -> Chem.Mol | None:
    """Parse exactly like ``smiles_to_graph`` so atom order matches the drug graph."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_salts:
        mol = _largest_fragment(mol)
        # The fragment is built with sanitizeFrags=False, so it has no ring info
        # and the feature SMARTS fail on it. Sanitising never reorders atoms.
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            Chem.FastFindRings(mol)
    AllChem.ComputeGasteigerCharges(mol)  # same side effect as the drug graph
    return mol


# --------------------------------------------------------------------------- #
# Microspecies at pH 7.4
# --------------------------------------------------------------------------- #

# Rule-based protonation at physiological pH (approximate pKa classes, like the
# defaults of Phase/Epik-style tools). Applied in place on a neutralised copy,
# so heavy-atom indices never change and ``atom2feat`` keeps indexing the drug
# graph. RDKit's canonical tautomer is deliberately NOT used: it is a canonical
# representative, not the dominant tautomer (it turns oxindoles into
# hydroxyindoles and deconjugates steroid enones), and would invent donors.
_ACID_SMARTS: tuple[tuple[str, str], ...] = (
    # (name, SMARTS whose FIRST atom is the one deprotonated)
    ("carboxylic acid", "[OX2H1]-[CX3]=O"),
    ("sulfonic/sulfinic acid", "[OX2H1]-[SX4,SX3]=O"),
    ("phosphoric/phosphonic acid", "[OX2H1]-[PX4]=O"),
    ("vinylogous acid", "[OX2H1]-[#6]=[#6]-[#6]=O"),  # meloxicam, ascorbic acid
    ("4-hydroxycoumarin/2-pyrone", "[OX2H1]-c1:c:c(=O):o:c:c1"),  # warfarin
    ("tetrazole", "[nX3H1;r5;$(n1nnnc1),$(n1nncn1),$(n1ncnn1),$(n1cnnn1)]"),
    ("acyl sulfonamide", "[NX3H1](-[CX3]=O)-[SX4](=O)=O"),  # incl. sulfonylureas
    ("N-heteroaryl sulfonamide", "[NX3H1](-[SX4](=O)=O)-c:n"),  # sulfa drugs
)
_BASE_SMARTS: tuple[tuple[str, str], ...] = (
    # Amidines / guanidines: protonate the imine N. Acyl-, sulfonyl-, cyano-
    # (cimetidine) and N-aryl imine variants drop the pKa well below 7.4.
    (
        "amidine/guanidine",
        "[NX2;+0;!$(N-[!#6;!#1]);!$(N-C=[O,S]);!$(N-C#N);!$(N-a)]"
        "=[CX3;!$(C-N-C#N);!$(C-N-C=[O,S]);!$(C-N-S(=O)=O)]"
        "-[NX3;+0;!$(N-[!#6;!#1])]",
    ),
    # Aliphatic amines: every neighbour an sp3 carbon (no amide, aniline,
    # enamine, sulfonamide, hydrazine, N-oxide), not next to a CF2/CF3 group.
    (
        "aliphatic amine",
        "[NX3;+0;$(N-[#6]);!$(N~[!#6;!#1]);!$(N-[#6;!X4]);"
        "!$(N-[CX4](F)F);!$(N-[CX4]-[CX4](F)(F)F)]",
    ),
)
_ACID_PATTERNS = tuple((n, Chem.MolFromSmarts(s)) for n, s in _ACID_SMARTS)
_BASE_PATTERNS = tuple((n, Chem.MolFromSmarts(s)) for n, s in _BASE_SMARTS)
_ANIONIC_O = Chem.MolFromSmarts("[O-;!$([O-]-[N+]=O)]")
_NEAR_CHARGE = 3  # a second base within 3 bonds of a cation stays neutral
_UNCHARGER = rdMolStandardize.Uncharger()


def _set_charge(atom: Chem.Atom, charge: int, h_delta: int) -> None:
    hs = atom.GetTotalNumHs()
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(max(0, hs + h_delta))
    atom.SetFormalCharge(charge)


def _microspecies(mol: Chem.Mol) -> Chem.Mol:
    """Copy of ``mol`` in its dominant protonation state at pH 7.4.

    Neutralise first (so a SMILES written as a salt or zwitterion gives the same
    result as the neutral one), then deprotonate acids with pKa < ~6.5 and
    protonate bases with pKa > ~8.5. Of bases within ``_NEAR_CHARGE`` bonds of
    an existing cation (piperazine, ethylenediamine) only the first is
    protonated: the second pKa drops by several units. Atom order is unchanged.
    """
    neutral = _UNCHARGER.uncharge(Chem.Mol(mol))
    if neutral.GetNumAtoms() != mol.GetNumAtoms():  # never expected; stay safe
        neutral = Chem.Mol(mol)
    rw = Chem.RWMol(neutral)

    acids: set[int] = set()
    centres: set[int] = set()  # one deprotonation per acid centre (P(=O)(OH)2)
    for _, patt in _ACID_PATTERNS:
        for match in rw.GetSubstructMatches(patt):
            atom_idx = match[0]
            centre = match[1] if len(match) > 1 else atom_idx
            if atom_idx in acids or centre in centres:
                continue
            acids.add(atom_idx)
            centres.add(centre)

    candidates: list[tuple[int, int, int]] = []  # (rule rank, heavy degree, idx)
    for rank, (_, patt) in enumerate(_BASE_PATTERNS):
        for match in rw.GetSubstructMatches(patt):
            atom = rw.GetAtomWithIdx(match[0])
            candidates.append((rank, atom.GetDegree(), match[0]))

    for idx in acids:
        _set_charge(rw.GetAtomWithIdx(idx), -1, -1)

    dist = Chem.GetDistanceMatrix(rw)
    cations = [a.GetIdx() for a in rw.GetAtoms() if a.GetFormalCharge() > 0]
    for _, _, idx in sorted(set(candidates)):
        if idx in cations or any(dist[idx, c] <= _NEAR_CHARGE for c in cations):
            continue
        _set_charge(rw.GetAtomWithIdx(idx), +1, +1)
        cations.append(idx)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except ValueError, RuntimeError:
        logger.debug("pH 7.4 microspecies failed to sanitise; using neutral form")
        out = neutral
    AllChem.ComputeGasteigerCharges(out)
    return out


# --------------------------------------------------------------------------- #
# Feature perception
# --------------------------------------------------------------------------- #


def _perceive(mol: Chem.Mol) -> list[tuple[str, tuple[int, ...]]]:
    """Pharmacophore features of a pH 7.4 microspecies, normalised and sorted.

    ``mol`` must already be the microspecies (:func:`_microspecies`). See
    :func:`_normalise_features` for the rules. The sort makes node order
    deterministic across RDKit runs.
    """
    raw: set[tuple[str, tuple[int, ...]]] = set()
    for feat in _FACTORY.GetFeaturesForMol(mol):
        raw.add((feat.GetFamily(), tuple(sorted(feat.GetAtomIds()))))
    kept = set(_normalise_features(mol, raw))
    return sorted(kept, key=lambda f: (f[1][0], _FAMILY_INDEX.get(f[0], 99), f[1]))


def _normalise_features(
    mol: Chem.Mol, raw: set[tuple[str, tuple[int, ...]]]
) -> list[tuple[str, tuple[int, ...]]]:
    """Map BaseFeatures.fdef output onto six non-overlapping families.

    BaseFeatures emits overlapping patterns and hedges over protonation and
    tautomers; kept, one chemical group counts as several features and any two
    drugs sharing it look alike. On the pH 7.4 microspecies:

    * ``Donor`` only on atoms that carry a hydrogen (drops the neutral tertiary
      amine and the H-free aromatic N of the tautomer hedge);
    * ``Acceptor`` as perceived, plus anionic oxygens; ``Aromatic`` one per ring;
    * ``PosIonizable`` / ``NegIonizable`` one per charged atom of the
      microspecies (nitro / N-oxide zwitterions excluded); an oxyanion group
      also takes its acid centre and that centre's terminal O/S (carboxylate =
      C,O,O); an anionic N (sulfonamide, tetrazole) is the single atom.
      Neutral imidazole/guanidine/amine patterns no longer count as charged;
    * ``Hydrophobe``: a non-aromatic ``LumpedHydrophobe`` group (t-butyl,
      cyclohexyl) is one feature; the remaining single-atom hydrophobes outside
      aromatic rings and lumped groups are merged into bonded chains;
    * ``ZnBinder`` is dropped (its atoms are already donors/acceptors/anions).
    """
    by_family: dict[str, list[tuple[int, ...]]] = {}
    for fam, ids in raw:
        by_family.setdefault(fam, []).append(ids)

    out: list[tuple[str, tuple[int, ...]]] = []
    out += [
        ("Donor", ids)
        for ids in by_family.get("Donor", [])
        if all(mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 for i in ids)
    ]
    out += [("Acceptor", ids) for ids in by_family.get("Acceptor", [])]
    out += _anionic_acceptors(mol)
    aromatic = by_family.get("Aromatic", [])
    out += [("Aromatic", ids) for ids in aromatic]
    out += _ionizable_features(mol)

    ring_sets = [frozenset(ids) for ids in aromatic]
    lumped = [
        frozenset(ids)
        for ids in by_family.get("LumpedHydrophobe", [])
        if not any(frozenset(ids) <= r for r in ring_sets)
    ]
    out += [("Hydrophobe", tuple(sorted(g))) for g in lumped]
    covered = ring_sets + lumped
    singles = {
        ids[0]
        for ids in by_family.get("Hydrophobe", [])
        if not any(ids[0] in g for g in covered)
    }
    out += [("Hydrophobe", chain) for chain in _bonded_components(mol, singles)]
    return out


def _ionizable_features(mol: Chem.Mol) -> list[tuple[str, tuple[int, ...]]]:
    out: list[tuple[str, tuple[int, ...]]] = []
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if not charge:
            continue
        nbrs = list(atom.GetNeighbors())
        if any(n.GetFormalCharge() * charge < 0 for n in nbrs):
            continue  # nitro, N-oxide: zwitterionic notation, not an ion
        if charge > 0:
            out.append(("PosIonizable", (atom.GetIdx(),)))
            continue
        group = {atom.GetIdx()}
        if atom.GetSymbol() in {"O", "S"}:  # oxyacid: take the centre and its =O
            for centre in nbrs:
                if centre.GetSymbol() in {"C", "S", "P"}:
                    group.add(centre.GetIdx())
                    group |= {
                        t.GetIdx()
                        for t in centre.GetNeighbors()
                        if t.GetSymbol() in {"O", "S"} and t.GetDegree() == 1
                    }
                    break
        out.append(("NegIonizable", tuple(sorted(group))))
    return out


def _anionic_acceptors(mol: Chem.Mol) -> list[tuple[str, tuple[int, ...]]]:
    """Acceptors on anionic oxygens (carboxylate, phenolate, N-oxide), not nitro.

    BaseFeatures.fdef means to cover them, but that line is spelled ``Atomtype``
    and RDKit does not apply it, so without this a carboxylate loses an acceptor
    once it is ionised.
    """
    return [("Acceptor", (match[0],)) for match in mol.GetSubstructMatches(_ANIONIC_O)]


def _bonded_components(mol: Chem.Mol, atoms: set[int]) -> list[tuple[int, ...]]:
    """Connected components of ``atoms`` through bonds between them."""
    left, comps = set(atoms), []
    while left:
        stack, comp = [left.pop()], set()
        while stack:
            i = stack.pop()
            comp.add(i)
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() in left:
                    left.discard(n.GetIdx())
                    stack.append(n.GetIdx())
        comps.append(tuple(sorted(comp)))
    return comps


def _node_features(
    mol: Chem.Mol, family: str, atom_ids: tuple[int, ...]
) -> list[float]:
    atoms = [mol.GetAtomWithIdx(i) for i in atom_ids]
    n = len(atoms)
    feats = _one_hot(family, list(_FAMILIES), other=True)
    feats.append(min(n, 6) / 6.0)
    feats.append(sum(a.GetIsAromatic() for a in atoms) / n)
    feats.append(sum(a.IsInRing() for a in atoms) / n)
    feats.append(sum(_gasteiger_charge(a) for a in atoms) / n)
    return feats


def pharmacophore_graph(mol: Chem.Mol) -> PyGData:
    """Build the pharmacophore graph of an already-prepared molecule.

    Features, charges and distances are read off the pH 7.4 microspecies of
    ``mol``; its atom indices are those of ``mol`` (and of the drug graph).
    """
    num_atoms = mol.GetNumAtoms()
    mol = _microspecies(mol)
    features = _perceive(mol)
    k = len(features)
    if k == 0:
        # Same attributes as a non-empty graph so batches never miss a key.
        data = PyGData(
            x=torch.empty((0, PHARM_NODE_DIM), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, PHARM_EDGE_DIM), dtype=torch.float),
        )
        data.global_feats = torch.zeros((1, PHARM_GLOBAL_DIM), dtype=torch.float)
        data.atom2feat = torch.empty((0, 2), dtype=torch.long)
        data.num_drug_atoms = num_atoms
        return data

    x = torch.tensor(
        [_node_features(mol, fam, ids) for fam, ids in features], dtype=torch.float
    )

    dist = Chem.GetDistanceMatrix(mol)  # 1e8 between disconnected fragments
    fam_counts = [0] * len(_FAMILIES)
    pair_hist = [0] * (len(_PAIR_INDEX) * _N_TOPO_BINS)
    for fam, _ in features:
        if fam in _FAMILY_INDEX:
            fam_counts[_FAMILY_INDEX[fam]] += 1

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            d = min(
                float(dist[np.ix_(features[i][1], features[j][1])].min()), _DIST_CAP
            )
            dbin = _bin(d, _TOPO_UPPER)
            bins = [0.0] * _N_TOPO_BINS
            bins[dbin] = 1.0
            attr = [*bins, d / _DIST_CAP]
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [attr, attr]
            fi, fj = (
                _FAMILY_INDEX.get(features[i][0]),
                _FAMILY_INDEX.get(features[j][0]),
            )
            if fi is not None and fj is not None:
                pair = _PAIR_INDEX[(min(fi, fj), max(fi, fj))]
                pair_hist[pair * _N_TOPO_BINS + dbin] += 1

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:  # a single feature has no pairs
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, PHARM_EDGE_DIM), dtype=torch.float)

    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.global_feats = torch.tensor(
        [[_log_count(c) for c in (*fam_counts, *pair_hist)]], dtype=torch.float
    )
    data.atom2feat = torch.tensor(
        [[a, f] for f, (_, ids) in enumerate(features) for a in ids], dtype=torch.long
    )
    data.num_drug_atoms = num_atoms
    return data


def smiles_to_pharmacophore_graph(
    smiles: str, *, strip_salts: bool = False
) -> PyGData | None:
    """SMILES → pharmacophore graph, or ``None`` if RDKit can't parse it.

    A molecule with no detectable feature returns a graph with zero nodes (the
    builder reports it as ``EMPTY_GRAPH``, like the drug builder does).
    """
    mol = _prepare_mol(smiles, strip_salts=strip_salts)
    return None if mol is None else pharmacophore_graph(mol)


class PharmacophoreGraphBuilder:
    """Iterate the drug catalog and save one pharmacophore graph per drug.

    Same contract as :class:`~src.data.library.drugs.DrugGraphBuilder`: resumable
    (skips existing ``.pt`` unless ``force``), failures go to a dedicated file
    logger and to the manifest. Output files share the drug graph's
    ``<cid>_<name>.pt`` stem.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        force: bool = False,
        failures_log: Path | None = None,
        strip_salts: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.force = force
        self.strip_salts = strip_salts
        self.failure_counts: Counter[DrugFailureCategory] = Counter()
        self._failure_logger = _build_file_logger(_FAILURE_LOGGER_NAME, failures_log)

    def build(
        self, drugs_path: Path, *, manifest: BuildManifest
    ) -> tuple[int, int, int]:
        """Build all pharmacophore graphs. Returns (built, skipped, failed)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not drugs_path.exists():
            msg = f"Drug catalog not found: {drugs_path}"
            raise FileNotFoundError(msg)

        rows = load_drug_records(drugs_path)
        logger.info("Pharmacophore builder: %d rows in %s", len(rows), drugs_path.name)
        self._failure_logger.info(
            "Build run started | source=%s | rows=%d", drugs_path.name, len(rows)
        )
        self.failure_counts = Counter()
        tally: Counter[str] = Counter()
        for row in tqdm(rows, total=len(rows), desc="Pharmacophores"):
            tally[self._build_one(row, manifest)] += 1

        built, skipped, failed = tally["built"], tally["skipped"], tally["failed"]
        manifest.stats.pharm_built = built
        manifest.stats.pharm_skipped = skipped
        manifest.stats.pharm_failed = failed
        breakdown = ", ".join(
            f"{c.value}={self.failure_counts[c]}" for c in DrugFailureCategory
        )
        logger.info(
            "Pharmacophore builder done: %d built, %d skipped, %d failed.",
            built,
            skipped,
            failed,
        )
        self._failure_logger.info(
            "Build run finished | built=%d skipped=%d failed=%d | %s",
            built,
            skipped,
            failed,
            breakdown,
        )
        if failed:
            logger.warning("Pharmacophore failure breakdown: %s", breakdown)
        return built, skipped, failed

    def _build_one(  # noqa PLR0911
        self, row: DrugRow, manifest: BuildManifest
    ) -> Literal["built", "skipped", "failed"]:
        cid_raw = row["cid"]
        try:
            cid = int(str(cid_raw).strip())
        except TypeError, ValueError:
            return self._fail(
                manifest,
                -1,
                str(cid_raw),
                DrugFailureCategory.NON_INTEGER_CID,
                f"non-integer CID: {cid_raw!r}",
            )

        raw_name = row.get("name")
        name = str(raw_name).strip() if raw_name else f"cid{cid}"
        out_path = self.output_dir / f"{cid}_{safe_filename(name)}.pt"
        if out_path.exists() and not self.force:
            manifest.mark_pharm_done(cid)
            return "skipped"

        smiles_raw = row.get("smiles")
        smiles = str(smiles_raw).strip() if smiles_raw is not None else ""
        if not smiles:
            return self._fail(
                manifest, cid, name, DrugFailureCategory.MISSING_SMILES, "empty SMILES"
            )

        graph = smiles_to_pharmacophore_graph(smiles, strip_salts=self.strip_salts)
        if graph is None:
            return self._fail(
                manifest,
                cid,
                name,
                DrugFailureCategory.INVALID_SMILES,
                f"RDKit could not parse: {smiles}",
            )
        if graph.num_nodes == 0:
            return self._fail(
                manifest,
                cid,
                name,
                DrugFailureCategory.EMPTY_GRAPH,
                f"no pharmacophore features: {smiles}",
            )

        graph.cid = cid
        graph.name = name
        graph.smiles = smiles
        try:
            torch.save(graph, out_path)
        except OSError as e:
            return self._fail(
                manifest, cid, name, DrugFailureCategory.SAVE_ERROR, f"save error: {e}"
            )
        manifest.mark_pharm_done(cid)
        return "built"

    def _fail(
        self,
        manifest: BuildManifest,
        cid: int,
        name: str,
        category: DrugFailureCategory,
        reason: str,
    ) -> Literal["failed"]:
        self.failure_counts[category] += 1
        if cid >= 0:
            manifest.mark_pharm_failed(cid, reason)
        self._failure_logger.warning(
            "pharm build failed | cid=%s | name=%s | nature=%s | reason=%s",
            cid,
            name,
            category.value,
            reason,
        )
        return "failed"
