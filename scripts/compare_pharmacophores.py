"""Compare the pharmacophore graphs of two drugs and print a side-by-side report.

Edit the constants below (a drug name or a PubChem CID each) and run::

    uv run python -m scripts.compare_pharmacophores

Each drug is resolved against the drug catalog (``data/drugs_cid.tsv``): a CID
directly, a name by exact match on ``cmpd_name``, then ``cmpd_name_cleaned``, then
the PubChem synonyms. The pharmacophore graph is read from
``data/library/pharmacophores/<cid>_<name>.pt`` when the library has it, and
built on the fly from the catalog SMILES otherwise (same code path as the
library build, so the result is identical).

The report covers, for each drug, the feature nodes (family, member atoms, node
features) and the feature-feature topological distance matrix; then the
comparison: family composition, the family-pair x distance profile, a
feature-to-feature matching and a set of similarity scores (the model-facing
``global_feats`` vector, multiset Tanimoto on raw counts, RDKit's Gobbi 2D
pharmacophore fingerprint and, for contrast, plain ECFP4 structure similarity).

This is an inspection utility, not part of the library pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # also runnable as `python scripts/...py`
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.data.library import pharmacophore as _pharm_module  # noqa: E402
from src.data.library.config import LibraryBuildConfig  # noqa: E402
from src.data.library.pharmacophore import (  # noqa: E402
    _DIST_CAP,
    _FAMILIES,
    PHARM_GLOBAL_DIM,
    _prepare_mol,
    pharmacophore_graph,
)

# --------------------------------------------------------------------------- #
# Inputs — a drug name (case-insensitive) or a PubChem CID (int or digit string)
# --------------------------------------------------------------------------- #
DRUG_A: str | int = "clopidogrel"
DRUG_B: str | int = "aspirin"

STRIP_SALTS = True  # same default as the library build
PREFER_LIBRARY = True  # read the built .pt when present; False = always rebuild
MAX_MATRIX_FEATURES = 24  # skip the distance matrix beyond this many features

# Defaults to the project paths; override to point at another catalog/library.
DRUGS_CATALOG: Path | None = None
PHARM_DIR: Path | None = None

# --------------------------------------------------------------------------- #

_FAMILY_ABBR = {
    "Donor": "Don",
    "Acceptor": "Acc",
    "PosIonizable": "Pos",
    "NegIonizable": "Neg",
    "Aromatic": "Aro",
    "Hydrophobe": "Hyd",
    "LumpedHydrophobe": "LHy",
    "ZnBinder": "Zn",
    "other": "oth",
}
_N_FAM = len(_FAMILIES)
# Coarse distance bins for the readable pair table / pair Tanimoto only (the
# library vector uses the 12 edge bins).
_PAIR_UPPER: tuple[int, ...] = (3, 6, 10)
_PAIR_BIN_LABELS = ("<=3", "4-6", "7-10", ">=11")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_WIDTH = 100
_BASELINE_MAX_ATOMS = 70  # heavy atoms; larger molecules are left out of the baseline


# --------------------------------------------------------------------------- #
# Resolution: name / CID -> catalog row
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CatalogEntry:
    cid: int
    name: str
    clean_name: str
    smiles: str


@dataclass
class Resolved:
    entry: CatalogEntry
    matched_on: str
    alternatives: list[CatalogEntry] = field(default_factory=list)


def load_catalog(path: Path) -> pl.DataFrame:
    """Read the drug catalog with every column the resolver needs."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema_length=0,  # everything as strings
        ignore_errors=True,
    )
    wanted = ["cid", "cmpd_name", "cmpdsynonym", "smiles", "cmpd_name_cleaned"]
    for col in wanted:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    return (
        df.select(wanted)
        .with_columns(pl.col("cid").str.strip_chars().cast(pl.Int64, strict=False))
        .filter(pl.col("cid").is_not_null())
    )


def _entry(row: dict[str, object]) -> CatalogEntry:
    clean = str(row.get("cmpd_name_cleaned") or "").strip()
    name = str(row.get("cmpd_name") or "").strip() or clean
    return CatalogEntry(
        cid=int(row["cid"]),  # type: ignore[arg-type]
        name=name or f"cid{row['cid']}",
        clean_name=clean or name or f"cid{row['cid']}",
        smiles=str(row.get("smiles") or "").strip(),
    )


def resolve(query: str | int, catalog: pl.DataFrame) -> Resolved:
    """Resolve a drug name or CID to one catalog entry.

    An exact ``cmpd_name`` match wins outright (the user named that record).
    Otherwise cleaned-name and synonym hits are pooled and ranked so the free
    drug (single-fragment SMILES) beats its salts, then cleaned name beats
    synonym, then the lowest CID.
    """
    text = str(query).strip()
    if text.isdigit():
        rows = catalog.filter(pl.col("cid") == int(text)).to_dicts()
        if not rows:
            msg = f"CID {text} is not in the drug catalog"
            raise LookupError(msg)
        return Resolved(_entry(rows[0]), matched_on="cid")

    needle = text.casefold()
    exact = catalog.filter(
        (pl.col("cmpd_name").str.to_lowercase() == needle).fill_null(value=False)
    ).to_dicts()
    if exact:
        entries = sorted((_entry(r) for r in exact), key=lambda e: e.cid)
        return Resolved(entries[0], matched_on="cmpd_name", alternatives=entries[1:])

    synonyms = pl.col("cmpdsynonym").fill_null("").str.to_lowercase().str.split("|")
    lookups = (
        ("cmpd_name_cleaned", pl.col("cmpd_name_cleaned").str.to_lowercase() == needle),
        ("synonym", synonyms.list.contains(needle)),
    )
    pooled: dict[int, tuple[CatalogEntry, int, str]] = {}
    for level, (label, expr) in enumerate(lookups):
        for row in catalog.filter(expr.fill_null(value=False)).to_dicts():
            entry = _entry(row)
            pooled.setdefault(entry.cid, (entry, level, label))
    if not pooled:
        msg = f"No drug named {text!r} in the catalog (tried name, cleaned name, synonyms)"
        raise LookupError(msg)
    ranked = sorted(pooled.values(), key=lambda t: ("." in t[0].smiles, t[1], t[0].cid))
    best, _, label = ranked[0]
    return Resolved(best, matched_on=label, alternatives=[t[0] for t in ranked[1:]])


# --------------------------------------------------------------------------- #
# Graph loading + decoding
# --------------------------------------------------------------------------- #


@dataclass
class PharmProfile:
    """A pharmacophore graph decoded into readable pieces."""

    entry: CatalogEntry
    source: str
    mol: Chem.Mol
    graph: PyGData
    families: list[str]
    atoms: list[list[int]]
    distances: torch.Tensor  # [K, K] topological distance (capped at _DIST_CAP)

    @property
    def k(self) -> int:
        return len(self.families)

    @cached_property
    def gobbi_fp(self) -> DataStructs.SparseBitVect:
        return Generate.Gen2DFingerprint(self.mol, Gobbi_Pharm2D.factory)

    @cached_property
    def morgan_fp(self) -> DataStructs.ExplicitBitVect:
        return _MORGAN.GetFingerprint(self.mol)

    def family_counts(self) -> Counter[str]:
        return Counter(self.families)

    def pair_profile(self) -> Counter[tuple[str, str, str]]:
        """(family_a, family_b, distance bin) -> count, over unordered pairs."""
        prof: Counter[tuple[str, str, str]] = Counter()
        for i in range(self.k):
            for j in range(i + 1, self.k):
                fa, fb = sorted((self.families[i], self.families[j]), key=_fam_order)
                prof[fa, fb, _pair_bin(float(self.distances[i, j]))] += 1
        return prof

    def atom_label(self, idx: int) -> str:
        return f"{self.mol.GetAtomWithIdx(idx).GetSymbol()}{idx}"


def _fam_order(family: str) -> int:
    return _FAMILIES.index(family) if family in _FAMILIES else _N_FAM


def _pair_bin(d: float) -> str:
    for upper, label in zip(_PAIR_UPPER, _PAIR_BIN_LABELS, strict=False):
        if d <= upper:
            return label
    return _PAIR_BIN_LABELS[-1]


def _find_library_graph(pharm_dir: Path, cid: int) -> Path | None:
    hits = sorted(pharm_dir.glob(f"{cid}_*.pt"))
    return hits[0] if hits else None


def build_profile(
    entry: CatalogEntry,
    *,
    pharm_dir: Path | None,
    strip_salts: bool = STRIP_SALTS,
    prefer_library: bool = PREFER_LIBRARY,
) -> PharmProfile:
    """Load (or build) the pharmacophore graph of ``entry`` and decode it."""
    mol = _prepare_mol(entry.smiles, strip_salts=strip_salts)
    if mol is None:
        msg = f"RDKit cannot parse the SMILES of CID {entry.cid}: {entry.smiles!r}"
        raise ValueError(msg)

    graph = pharmacophore_graph(mol)
    source = "built on the fly"
    path = _find_library_graph(pharm_dir, entry.cid) if pharm_dir else None
    if prefer_library and path is not None:
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if _same_graph(stored, graph):
            graph, source = stored, f"library ({path.name})"
        else:
            # Built by older code or another salt setting: show what the current
            # code produces instead of a stale graph.
            source = (
                f"built on the fly (library {path.name} is stale; "
                "rebuild with `python -m src.data.library --force`)"
            )

    k = graph.num_nodes
    families = []
    for row in graph.x[:, : _N_FAM + 1]:
        idx = int(row.argmax())
        families.append(_FAMILIES[idx] if idx < _N_FAM else "other")

    atoms: list[list[int]] = [[] for _ in range(k)]
    if k:
        for atom, feat in graph.atom2feat.tolist():
            atoms[feat].append(atom)

    distances = torch.zeros((k, k))
    for e, (i, j) in enumerate(graph.edge_index.t().tolist()):
        distances[i, j] = round(float(graph.edge_attr[e, -1]) * _DIST_CAP)

    return PharmProfile(entry, source, mol, graph, families, atoms, distances)


def _same_graph(a: PyGData, b: PyGData) -> bool:
    if a.num_nodes != b.num_nodes or a.x.shape != b.x.shape:
        return False
    if not a.num_nodes:
        return True
    return torch.equal(a.x, b.x) and torch.equal(a.atom2feat, b.atom2feat)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #


def multiset_tanimoto(a: Counter, b: Counter) -> float:
    """Sum of minima over sum of maxima (1.0 = identical counts)."""
    keys = set(a) | set(b)
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den else 1.0


def global_cosine(a: PharmProfile, b: PharmProfile) -> float:
    """Cosine of the library ``global_feats`` vectors (0.0 if either is empty).

    The vectors are non-negative log-scaled counts, so the cosine of two
    unrelated drug-like molecules is already well above 0: read it against the
    random-pair percentile, not as an absolute [0, 1] similarity.
    """
    ga, gb = _global(a), _global(b)
    if not ga.any() or not gb.any():
        return 0.0
    return float(torch.nn.functional.cosine_similarity(ga, gb).item())


def _global(p: PharmProfile) -> torch.Tensor:
    g = getattr(p.graph, "global_feats", None)
    if g is None:  # featureless molecule
        return torch.zeros((1, PHARM_GLOBAL_DIM))
    return g.float()


def gobbi_tanimoto(a: PharmProfile, b: PharmProfile) -> float:
    return DataStructs.TanimotoSimilarity(a.gobbi_fp, b.gobbi_fp)


def morgan_tanimoto(a: PharmProfile, b: PharmProfile) -> float:
    return DataStructs.TanimotoSimilarity(a.morgan_fp, b.morgan_fp)


@dataclass(frozen=True)
class Match:
    a: int
    b: int
    node: float  # similarity of the node features
    context: float  # similarity of the distance profile to the other features

    @property
    def score(self) -> float:
        """Node x context similarity.

        Multiplied, not averaged: two features of the same family almost always
        have node similarity ~1 (same size, same aromatic/ring fraction), so an
        average would put a ~0.5 floor under every match. The product is high
        only when the feature *and* its environment agree.
        """
        return self.node * self.context


def node_similarity(a: PharmProfile, i: int, b: PharmProfile, j: int) -> float:
    """1 - mean absolute difference of (size, aromatic, ring, charge), in [0, 1].

    Charge lives in [-1, 1], so its difference is halved to stay on [0, 1].
    """
    xa, xb = a.graph.x[i, _N_FAM + 1 :], b.graph.x[j, _N_FAM + 1 :]
    diff = (xa - xb).abs()
    diff[-1] = diff[-1] / 2.0
    return 1.0 - float(diff.mean())


def match_features(a: PharmProfile, b: PharmProfile) -> list[Match]:
    """Greedy one-to-one matching of features of the same family.

    Score = node-feature similarity (size, aromatic/ring fraction, charge) x
    how alike the two features' environments are (multiset Tanimoto over
    (other family, distance bin)). Highest-scoring pairs first.
    """
    if not a.k or not b.k:
        return []
    ctx_a, ctx_b = _context(a), _context(b)

    candidates: list[Match] = []
    for i in range(a.k):
        for j in range(b.k):
            if a.families[i] != b.families[j]:
                continue
            candidates.append(
                Match(
                    i,
                    j,
                    node=node_similarity(a, i, b, j),
                    context=multiset_tanimoto(ctx_a[i], ctx_b[j]),
                )
            )

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[Match] = []
    for m in sorted(candidates, key=lambda m: -m.score):
        if m.a in used_a or m.b in used_b:
            continue
        matches.append(m)
        used_a.add(m.a)
        used_b.add(m.b)
    return sorted(matches, key=lambda m: m.a)


def _context(p: PharmProfile) -> list[Counter[tuple[str, str]]]:
    """Per feature: (other family, distance bin) counts — its local environment."""
    ctx: list[Counter[tuple[str, str]]] = [Counter() for _ in range(p.k)]
    for i in range(p.k):
        for j in range(p.k):
            if i != j:
                ctx[i][p.families[j], _pair_bin(float(p.distances[i, j]))] += 1
    return ctx


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #


def _rule(title: str = "", char: str = "=") -> None:
    if title:
        print(f"\n{char * 3} {title} {char * max(3, _WIDTH - len(title) - 5)}")
    else:
        print(char * _WIDTH)


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_identity(tag: str, resolved: Resolved, p: PharmProfile) -> None:
    e = resolved.entry
    _rule(f"Drug {tag}: {e.name} (CID {e.cid})")
    print(f"  matched on      : {resolved.matched_on}")
    print(f"  cleaned name    : {e.clean_name}")
    print(f"  SMILES          : {e.smiles}")
    print(f"  graph source    : {p.source}")
    print(
        f"  heavy atoms     : {p.mol.GetNumAtoms()} (after salt stripping={STRIP_SALTS})"
    )
    print(f"  features (nodes): {p.k}")
    print(f"  edges (directed): {p.graph.edge_index.shape[1]}")
    print(f"  atoms covered   : {len({a for ids in p.atoms for a in ids})}")
    if resolved.alternatives:
        alts = ", ".join(f"{x.cid} ({x.name})" for x in resolved.alternatives[:6])
        more = len(resolved.alternatives) - 6
        print(f"  other matches   : {alts}{f' (+{more} more)' if more > 0 else ''}")


def print_features(tag: str, p: PharmProfile) -> None:
    _rule(f"{tag} - pharmacophore features", "-")
    if not p.k:
        print("  (no pharmacophore feature detected)")
        return
    print(
        f"  {'#':>3}  {'family':<17} {'n':>2}  {'arom':>5} {'ring':>5} {'charge':>7}  atoms"
    )
    for i, (fam, ids) in enumerate(zip(p.families, p.atoms, strict=True)):
        x = p.graph.x[i]
        atoms = " ".join(p.atom_label(a) for a in ids)
        print(
            f"  {i:>3}  {fam:<17} {len(ids):>2}  {float(x[_N_FAM + 2]):>5.2f} "
            f"{float(x[_N_FAM + 3]):>5.2f} {float(x[_N_FAM + 4]):>+7.3f}  {_short(atoms, 40)}"
        )


def print_distance_matrix(tag: str, p: PharmProfile) -> None:
    _rule(f"{tag} - topological distance between features (bonds)", "-")
    if p.k < 2:
        print("  (fewer than two features: no pairs)")
        return
    if p.k > MAX_MATRIX_FEATURES:
        print(
            f"  ({p.k} features > MAX_MATRIX_FEATURES={MAX_MATRIX_FEATURES}; skipped)"
        )
        return
    labels = [f"{i}{_FAMILY_ABBR.get(f, f[:3])}" for i, f in enumerate(p.families)]
    w = max(len(s) for s in labels) + 1
    print("  " + " " * w + "".join(f"{s:>{w}}" for s in labels))
    for i, lab in enumerate(labels):
        cells = []
        for j in range(p.k):
            d = int(p.distances[i, j])
            cells.append("." if i == j else (">30" if d >= _DIST_CAP else str(d)))
        print(f"  {lab:<{w}}" + "".join(f"{c:>{w}}" for c in cells))


def print_family_comparison(a: PharmProfile, b: PharmProfile) -> None:
    _rule("Family composition")
    ca, cb = a.family_counts(), b.family_counts()
    print(f"  {'family':<17} {'A':>4} {'B':>4} {'B-A':>5}  {'shared':>6}")
    for fam in [*_FAMILIES, "other"]:
        if not ca[fam] and not cb[fam]:
            continue
        print(
            f"  {fam:<17} {ca[fam]:>4} {cb[fam]:>4} {cb[fam] - ca[fam]:>+5}  "
            f"{min(ca[fam], cb[fam]):>6}"
        )
    print(
        f"  {'total':<17} {a.k:>4} {b.k:>4} {b.k - a.k:>+5}  {sum((ca & cb).values()):>6}"
    )
    only_a = sorted(set(ca) - set(cb), key=_fam_order)
    only_b = sorted(set(cb) - set(ca), key=_fam_order)
    print(
        f"\n  families shared : {', '.join(sorted(set(ca) & set(cb), key=_fam_order)) or '-'}"
    )
    print(f"  only in A       : {', '.join(only_a) or '-'}")
    print(f"  only in B       : {', '.join(only_b) or '-'}")


def print_pair_profile(a: PharmProfile, b: PharmProfile) -> None:
    _rule("Family-pair x distance profile (unordered feature pairs)")
    pa, pb = a.pair_profile(), b.pair_profile()
    keys = sorted(
        set(pa) | set(pb),
        key=lambda k: (
            _fam_order(k[0]),
            _fam_order(k[1]),
            _PAIR_BIN_LABELS.index(k[2]),
        ),
    )
    if not keys:
        print("  (no feature pairs in either drug)")
        return
    print(f"  {'pair':<36} {'dist':>5} {'A':>4} {'B':>4}  note")
    for fa, fb, dbin in keys:
        na, nb = pa[fa, fb, dbin], pb[fa, fb, dbin]
        note = (
            "same" if na == nb else ("only A" if not nb else "only B" if not na else "")
        )
        print(f"  {fa + ' - ' + fb:<36} {dbin:>5} {na:>4} {nb:>4}  {note}")
    same = sum(1 for k in keys if pa[k] == pb[k])
    print(
        f"\n  pair/bin cells: {len(keys)} | identical: {same} "
        f"| only A: {sum(1 for k in keys if not pb[k])} "
        f"| only B: {sum(1 for k in keys if not pa[k])}"
    )


def print_matching(a: PharmProfile, b: PharmProfile, matches: list[Match]) -> None:
    _rule("Feature matching (same family, greedy by node x context similarity)")
    if not matches:
        print("  (no feature can be matched)")
    for m in matches:
        atoms_a = " ".join(a.atom_label(x) for x in a.atoms[m.a])
        atoms_b = " ".join(b.atom_label(x) for x in b.atoms[m.b])
        print(
            f"  A#{m.a:<3} {a.families[m.a]:<17} [{_short(atoms_a, 22):<22}]  <->  "
            f"B#{m.b:<3} [{_short(atoms_b, 22):<22}]  "
            f"score={m.score:.2f} (node={m.node:.2f} ctx={m.context:.2f})"
        )
    un_a = [i for i in range(a.k) if i not in {m.a for m in matches}]
    un_b = [j for j in range(b.k) if j not in {m.b for m in matches}]
    print(
        f"\n  unmatched in A: {', '.join(f'#{i} {a.families[i]}' for i in un_a) or '-'}"
    )
    print(
        f"  unmatched in B: {', '.join(f'#{j} {b.families[j]}' for j in un_b) or '-'}"
    )


@dataclass(frozen=True)
class Scores:
    family_tanimoto: float
    pair_tanimoto: float
    match_coverage: float
    match_similarity: float
    global_cosine: float
    gobbi_tanimoto: float
    morgan_tanimoto: float


_SCORE_LABELS: dict[str, str] = {
    "family_tanimoto": "family composition (multiset Tanimoto)",
    "pair_tanimoto": "pair x distance profile (multiset Tanimoto)",
    "match_coverage": "feature matching coverage",
    "match_similarity": "feature matching similarity",
    "global_cosine": "global_feats cosine (library vector)",
    "gobbi_tanimoto": "RDKit Gobbi 2D pharmacophore FP Tanimoto",
    "morgan_tanimoto": "ECFP4 structural Tanimoto (contrast)",
}


def compute_scores(a: PharmProfile, b: PharmProfile, matches: list[Match]) -> Scores:
    """All similarity scores of a pair; every one is in [0, 1], 1 = identical.

    The matching scores are normalised by the larger feature count, so features
    left unmatched in either drug pull them down.
    """
    longest = max(a.k, b.k)
    return Scores(
        family_tanimoto=multiset_tanimoto(a.family_counts(), b.family_counts()),
        pair_tanimoto=multiset_tanimoto(a.pair_profile(), b.pair_profile()),
        match_coverage=len(matches) / longest if longest else 1.0,
        match_similarity=sum(m.score for m in matches) / longest if longest else 1.0,
        global_cosine=global_cosine(a, b),
        gobbi_tanimoto=gobbi_tanimoto(a, b),
        morgan_tanimoto=morgan_tanimoto(a, b),
    )


# --------------------------------------------------------------------------- #
# Random-pair baseline
# --------------------------------------------------------------------------- #


@dataclass
class Baseline:
    """Each score's distribution over random catalog pairs (sorted ascending)."""

    n_molecules: int
    n_pairs: int
    values: dict[str, list[float]]

    def percentile(self, metric: str, value: float) -> float:
        """Fraction of random pairs scoring <= ``value``."""
        dist = self.values[metric]
        return bisect_right(dist, value) / len(dist)

    def quantile(self, metric: str, q: float) -> float:
        dist = self.values[metric]
        return dist[min(len(dist) - 1, int(q * len(dist)))]


def random_pair_baseline(
    catalog: pl.DataFrame,
    *,
    n_molecules: int,
    n_pairs: int,
    seed: int = 0,
) -> Baseline:
    """Score ``n_pairs`` random pairs drawn from ``n_molecules`` catalog drugs.

    Graphs are built with the current code (not read from the library), so the
    baseline always matches the scores it is compared with.
    """
    rng = random.Random(seed)
    rows = [
        r
        for r in catalog.to_dicts()
        if r.get("smiles") and "." not in str(r["smiles"])  # free drugs only
    ]
    profiles: list[PharmProfile] = []
    for row in rng.sample(rows, min(n_molecules, len(rows))):
        try:
            profile = build_profile(_entry(row), pharm_dir=None)
        except ValueError:
            continue
        # Large peptides/macrolides make Gobbi FP and matching very slow and are
        # not typical small-molecule drugs.
        if profile.k and profile.mol.GetNumAtoms() <= _BASELINE_MAX_ATOMS:
            profiles.append(profile)
    if len(profiles) < 2:  # noqa: PLR2004
        msg = "Not enough parsable catalog drugs for a random-pair baseline"
        raise ValueError(msg)

    values: dict[str, list[float]] = {f.name: [] for f in fields(Scores)}
    for _ in range(n_pairs):
        a, b = rng.sample(profiles, 2)
        scores = compute_scores(a, b, match_features(a, b))
        for name, dist in values.items():
            dist.append(getattr(scores, name))
    return Baseline(
        n_molecules=len(profiles),
        n_pairs=n_pairs,
        values={k: sorted(v) for k, v in values.items()},
    )


def cached_baseline(
    catalog_path: Path,
    catalog: pl.DataFrame,
    *,
    n_molecules: int,
    n_pairs: int,
    seed: int = 0,
) -> Baseline:
    """:func:`random_pair_baseline`, cached under ``reports/pharmacophores/``.

    The cache key covers the catalog file, the sampling parameters and the
    source of this script and of the pharmacophore builder, so any change to
    how graphs or scores are computed recomputes it.
    """
    key = hashlib.sha256()
    stat = catalog_path.stat()
    key.update(f"{catalog_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    key.update(f"{n_molecules}|{n_pairs}|{seed}|{_BASELINE_MAX_ATOMS}".encode())
    for src in (Path(__file__), Path(_pharm_module.__file__)):
        key.update(src.read_bytes())
    cache = (
        get_settings().paths.reports
        / "pharmacophores"
        / f"compare_baseline_{key.hexdigest()[:16]}.json"
    )
    if cache.is_file():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return Baseline(data["n_molecules"], data["n_pairs"], data["values"])

    print(
        f"Computing the random-pair baseline ({n_pairs} pairs, cached in {cache})...",
        file=sys.stderr,
    )
    baseline = random_pair_baseline(
        catalog, n_molecules=n_molecules, n_pairs=n_pairs, seed=seed
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "n_molecules": baseline.n_molecules,
                "n_pairs": baseline.n_pairs,
                "values": baseline.values,
            }
        ),
        encoding="utf-8",
    )
    return baseline


# --------------------------------------------------------------------------- #
# Summary + verdict
# --------------------------------------------------------------------------- #

_HIGH_PCTL = 0.95  # above 95% of random pairs: clearly similar
_MODERATE_PCTL = 0.80  # above 80%: somewhat similar
# Fallback without a baseline (rough literature-style cut-offs).
_GOBBI_SIMILAR = 0.20
_MORGAN_SIMILAR = 0.35


def _level(baseline: Baseline | None, metric: str, value: float) -> str:
    """'high' / 'moderate' / 'typical' (of unrelated drugs) for one score."""
    if baseline is None:
        cut = _GOBBI_SIMILAR if metric == "gobbi_tanimoto" else _MORGAN_SIMILAR
        return "high" if value >= cut else "typical"
    pct = baseline.percentile(metric, value)
    if pct >= _HIGH_PCTL:
        return "high"
    if pct >= _MODERATE_PCTL:
        return "moderate"
    return "typical"


def verdict(s: Scores, baseline: Baseline | None) -> str:
    """One-line reading of the two reference fingerprints (Gobbi + ECFP4)."""
    pharm = _level(baseline, "gobbi_tanimoto", s.gobbi_tanimoto)
    struct = _level(baseline, "morgan_tanimoto", s.morgan_tanimoto)
    pharm_sim, struct_sim = pharm != "typical", struct != "typical"
    if pharm_sim and struct_sim:
        return f"Similar both pharmacophorically ({pharm}) and structurally ({struct})."
    if pharm_sim:
        return (
            f"Pharmacophorically similar ({pharm}) but not structurally: "
            "possible scaffold hop."
        )
    if struct_sim:
        return (
            f"Structurally similar ({struct}) but not pharmacophorically: shared "
            "scaffold, different interaction pattern."
        )
    reference = (
        "the range of unrelated drug pairs"
        if baseline
        else f"below Gobbi {_GOBBI_SIMILAR} / ECFP4 {_MORGAN_SIMILAR}"
    )
    return (
        "NOT similar: pharmacophoric and structural similarity are both in "
        f"{reference}."
    )


def print_scores(s: Scores, baseline: Baseline | None) -> None:
    _rule("Similarity summary")
    if baseline is None:
        print(f"  {'score':<46} {'value':>7}")
    else:
        print(
            f"  {'score':<46} {'value':>7}  {'pctl':>5}  {'random p50':>10} {'p95':>6}"
        )
    for f in fields(Scores):
        value = getattr(s, f.name)
        line = f"  {_SCORE_LABELS[f.name]:<46} {value:>7.3f}"
        if baseline is not None:
            line += (
                f"  {baseline.percentile(f.name, value):>5.0%}  "
                f"{baseline.quantile(f.name, 0.5):>10.3f} "
                f"{baseline.quantile(f.name, 0.95):>6.3f}"
            )
        print(line)

    print()
    if baseline is not None:
        print(
            f"  pctl = share of {baseline.n_pairs} random catalog pairs "
            f"({baseline.n_molecules} drugs) scoring <= this pair. Unrelated drugs\n"
            "  already share common families (donor/acceptor/aromatic), so a raw "
            "value only means\n  something next to the random-pair median."
        )
    else:
        print("  (no random-pair baseline: raw values are not calibrated)")
    print(f"\n  -> {verdict(s, baseline)}")


def compare(
    query_a: str | int,
    query_b: str | int,
    *,
    baseline_molecules: int = 400,
    baseline_pairs: int = 3000,
) -> Scores:
    """Resolve both drugs, print the full report and return the scores.

    ``baseline_pairs=0`` skips the random-pair baseline.
    """
    defaults = LibraryBuildConfig.from_settings()
    catalog_path = DRUGS_CATALOG or defaults.drugs_tsv
    pharm_dir = PHARM_DIR or defaults.pharm_out
    if not pharm_dir.is_dir():
        pharm_dir = None

    catalog = load_catalog(catalog_path)
    ra, rb = resolve(query_a, catalog), resolve(query_b, catalog)
    pa = build_profile(ra.entry, pharm_dir=pharm_dir)
    pb = build_profile(rb.entry, pharm_dir=pharm_dir)

    _rule()
    print(f"PHARMACOPHORE COMPARISON   A = {ra.entry.name}   vs   B = {rb.entry.name}")
    print(f"catalog: {catalog_path}")
    print(f"library: {pharm_dir or '(not built)'}")
    _rule()
    if ra.entry.cid == rb.entry.cid:
        print("  note: both queries resolve to the same CID.")

    for tag, r, p in (("A", ra, pa), ("B", rb, pb)):
        print_identity(tag, r, p)
        print_features(tag, p)
        print_distance_matrix(tag, p)

    matches = match_features(pa, pb)
    print_family_comparison(pa, pb)
    print_pair_profile(pa, pb)
    print_matching(pa, pb, matches)
    scores = compute_scores(pa, pb, matches)
    baseline = (
        cached_baseline(
            catalog_path,
            catalog,
            n_molecules=baseline_molecules,
            n_pairs=baseline_pairs,
        )
        if baseline_pairs > 0
        else None
    )
    print_scores(scores, baseline)
    _rule()
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the pharmacophore graphs of two drugs."
    )
    parser.add_argument(
        "--drug_a", "-a", type=str, default=DRUG_A, help="Drug A (name or PubChem CID)"
    )
    parser.add_argument(
        "--drug_b", "-b", type=str, default=DRUG_B, help="Drug B (name or PubChem CID)"
    )
    parser.add_argument(
        "--baseline-pairs",
        type=int,
        default=3000,
        help="Random catalog pairs used to calibrate the scores (0 = skip).",
    )
    parser.add_argument(
        "--baseline-molecules",
        type=int,
        default=400,
        help="Catalog drugs sampled for the random-pair baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        compare(
            args.drug_a,
            args.drug_b,
            baseline_molecules=args.baseline_molecules,
            baseline_pairs=args.baseline_pairs,
        )
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
