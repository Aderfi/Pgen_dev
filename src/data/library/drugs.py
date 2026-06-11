"""Drug graph builder — SMILES → PyG ``Data`` molecular graph.

Output schema (must stay in sync with ``models.toml`` and the trained model).
Every categorical field carries an explicit "other" / full vocabulary and every
continuous field is normalised, so out-of-vocabulary values are never silently
encoded as an all-zeros vector (see :class:`FeatureSaturation`).

    Node features (61):
        element one-hot[C,N,O,S,P,F,Cl,Br,I,B,Se,Si,other]   (13)
        degree one-hot[0..6]                                  (7)
        total-valence one-hot[0..6]                           (7)
        formal-charge scalar + one-hot[-1,0,+1,other]         (1+4)
        hybridization one-hot[SP,SP2,SP3,SP3D,SP3D2,S,other]  (7)
        total-Hs one-hot[0..4]                                (5)
        chirality one-hot[unspec,CW,CCW,other]                (4)
        is_aromatic, is_in_ring                               (2)
        ring-size membership[3,4,5,6,7]                        (5)
        Gasteiger partial charge (scalar, clamped)            (1)
        electronegativity (Pauling, normalised)               (1)
        is_H_donor, is_H_acceptor                             (2)
        mass (normalised), num_radical_electrons              (2)
    Edge features (18):
        bond-type one-hot[SINGLE,DOUBLE,TRIPLE,AROMATIC,other](5)
        is_conjugated, is_in_ring                             (2)
        ring-size membership[3,4,5,6,7]                        (5)
        stereo one-hot[NONE,Z,E,CIS,TRANS,other]              (6)

Input formats (auto-detected by file extension):
    * ``.tsv`` / ``.csv`` — tabular catalog with ``cid, smiles,
      cmpd_name_cleaned`` columns (e.g. ``data/drugs_cid.tsv``).
    * ``.json`` — flat ``{cid: smiles}`` mapping, no name column
      (e.g. ``BACKUPS/cid_smiles_dict.json``).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from torch_geometric.data import Data as PyGData
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.data.library.manifest import BuildManifest

logger = logging.getLogger(__name__)


_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')

# Schema dimensions exposed for sanity checks — see DOCSTRING above. Must match
# ``drug_node_features`` / ``drug_attrs_features`` in src/config/data/models.toml.
DRUG_NODE_DIM: int = 61
DRUG_EDGE_DIM: int = 18

# --- Categorical vocabularies (all use an explicit "other" bucket) --- #
_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Se", "Si"]
_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.S,
]
_CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
_BOND_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]

# --- Bounded numeric bins (overflow IS recorded as saturation) --- #
_DEGREE_BINS = [0, 1, 2, 3, 4, 5, 6]
_VALENCE_BINS = [0, 1, 2, 3, 4, 5, 6]
_CHARGE_BINS = [-1, 0, 1]
_HS_BINS = [0, 1, 2, 3, 4]
_RING_SIZES = [3, 4, 5, 6, 7]

# Pauling electronegativity by atomic number (normalised by max ≈ 3.98 = F).
_PAULING_EN: dict[int, float] = {
    1: 2.20,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    34: 2.55,
    35: 2.96,
    53: 2.66,
}
_DONOR_ELEMENTS = {"N", "O", "S"}
_ACCEPTOR_ELEMENTS = {"N", "O"}


class DrugFailureCategory(StrEnum):
    """The nature of a per-drug build failure, used for the failure counter."""

    NON_INTEGER_CID = "non_integer_cid"
    MISSING_SMILES = "missing_smiles"
    INVALID_SMILES = "invalid_smiles"
    EMPTY_GRAPH = "empty_graph"
    SAVE_ERROR = "save_error"


# Dedicated loggers. Each owns a FileHandler pointing at logs/library/ and does
# not propagate, so its report stays a clean, self-contained audit trail
# separate from the main application log.
_FAILURE_LOGGER_NAME = "Pharmagen.library.drug_failures"
_SATURATION_LOGGER_NAME = "Pharmagen.library.drug_feature_saturation"

# Numeric bins that can still overflow to an all-zeros vector (categorical
# fields all use an explicit "other" bucket, so only these are tracked). A
# non-zero count here flags a genuinely out-of-range value to widen a bin for.
_SATURATING_FEATURES: tuple[str, ...] = (
    "degree",
    "valence",
    "total_hs",
)


class FeatureSaturation:
    """Accumulator for one-hot saturation events (all-zeros encodings).

    Categorical fields use explicit "other" buckets, but the bounded numeric bins
    (degree, valence, total_Hs) can still overflow to an all-zeros vector — losing
    the value silently. This tallies *where* that happens so the loss is
    observable: ``by_feature`` counts events per field (e.g. ``degree``),
    ``by_value`` per concrete out-of-range value (e.g. ``degree=7``).
    """

    def __init__(self) -> None:
        self.by_feature: Counter[str] = Counter()
        self.by_value: Counter[str] = Counter()

    def record(self, feature: str, value: Any) -> None:
        self.by_feature[feature] += 1
        self.by_value[f"{feature}={value}"] += 1

    @property
    def total(self) -> int:
        return sum(self.by_feature.values())


def safe_filename(name: str, max_length: int = 100) -> str:
    """Strip filesystem-unsafe characters and cap length.

    Used to build ``<cid>_<safe_name>.pt`` paths. Keeps drug names readable
    while ensuring they round-trip through Linux/Windows filesystems.
    """
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("_", str(name).strip()).replace(" ", "_")
    return cleaned[:max_length]


def _one_hot(
    value: Any,
    choices: list[Any],
    *,
    other: bool = False,
    sink: FeatureSaturation | None = None,
    feature: str | None = None,
) -> list[float]:
    """One-hot encode ``value`` over ``choices``.

    With ``other=True`` an extra trailing slot catches any unknown value (so the
    encoding never saturates). Otherwise, an unknown value yields an all-zeros
    vector and — when a ``sink``/``feature`` is given — is recorded as a
    saturation event.
    """
    encoding = [0.0] * (len(choices) + (1 if other else 0))
    if value in choices:
        encoding[choices.index(value)] = 1.0
    elif other:
        encoding[-1] = 1.0
    elif sink is not None and feature is not None:
        # All-zeros: the value is out of range and its information is lost.
        sink.record(feature, value)
    return encoding


def _gasteiger_charge(atom: rdchem.Atom) -> float:
    """Gasteiger partial charge, clamped to [-1, 1] (NaN/inf → 0)."""
    try:
        q = atom.GetDoubleProp("_GasteigerCharge")
    except KeyError:
        return 0.0
    if math.isnan(q) or math.isinf(q):
        return 0.0
    return max(-1.0, min(1.0, q))


def _atom_features(
    atom: rdchem.Atom, saturation: FeatureSaturation | None
) -> list[float]:
    """Encode one atom into the 61-dim node feature vector."""
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    feats: list[float] = []
    feats += _one_hot(symbol, _ELEMENTS, other=True)
    feats += _one_hot(atom.GetDegree(), _DEGREE_BINS, sink=saturation, feature="degree")
    feats += _one_hot(
        atom.GetTotalValence(), _VALENCE_BINS, sink=saturation, feature="valence"
    )
    feats.append(max(-1.0, min(1.0, charge / 2.0)))
    feats += _one_hot(charge, _CHARGE_BINS, other=True)
    feats += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS, other=True)
    feats += _one_hot(
        atom.GetTotalNumHs(), _HS_BINS, sink=saturation, feature="total_hs"
    )
    feats += _one_hot(atom.GetChiralTag(), _CHIRAL_TAGS, other=True)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats += [1.0 if atom.IsInRingSize(n) else 0.0 for n in _RING_SIZES]
    feats.append(_gasteiger_charge(atom))
    feats.append(_PAULING_EN.get(atom.GetAtomicNum(), 0.0) / 3.98)
    feats.append(1.0 if symbol in _DONOR_ELEMENTS and atom.GetTotalNumHs() > 0 else 0.0)
    feats.append(1.0 if symbol in _ACCEPTOR_ELEMENTS and charge <= 0 else 0.0)
    feats.append(atom.GetMass() / 120.0)
    feats.append(float(atom.GetNumRadicalElectrons()))
    return feats


def _bond_features(bond: rdchem.Bond) -> list[float]:
    """Encode one bond into the 18-dim edge feature vector."""
    feats: list[float] = []
    feats += _one_hot(bond.GetBondType(), _BOND_TYPES, other=True)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    feats += [1.0 if bond.IsInRingSize(n) else 0.0 for n in _RING_SIZES]
    feats += _one_hot(bond.GetStereo(), _BOND_STEREO, other=True)
    return feats


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the fragment with the most atoms (drops salts / counterions).

    SMILES separated by ``.`` are disconnected components; for a drug salt the
    pharmacophore is the largest fragment and the rest (Na+, Cl-, …) is inert
    counterion noise that would otherwise pollute the graph and its pooling.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def smiles_to_graph(
    smiles: str,
    *,
    saturation: FeatureSaturation | None = None,
    strip_salts: bool = False,
) -> PyGData | None:
    """Convert a SMILES string into a PyG molecular graph.

    Returns ``None`` if RDKit can't parse the SMILES. When a
    :class:`FeatureSaturation` is passed, every one-hot that falls outside its
    frozen bins (hypervalent atoms, exotic bond types, …) is recorded there; the
    returned graph is byte-identical either way, so this never affects training.

    With ``strip_salts=True`` a multi-fragment molecule is reduced to its largest
    fragment before encoding, dropping salt counterions.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_salts:
        mol = _largest_fragment(mol)
    # Populates each atom's ``_GasteigerCharge`` property (read in _atom_features).
    AllChem.ComputeGasteigerCharges(mol)

    # Nodes (atoms) — 61 features each.
    atom_features = [_atom_features(atom, saturation) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges (bonds) — 18 features each, bidirectional.
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feats = _bond_features(bond)
        edge_indices += [[start, end], [end, start]]
        edge_attrs += [bond_feats, bond_feats]

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, DRUG_EDGE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


# --------------------------------------------------------------------------- #
# Catalog readers — normalise every supported format to ``{cid, smiles, name}``
# --------------------------------------------------------------------------- #

# A normalised raw row before validation: ``cid``/``smiles``/``name`` may still
# be ``None`` or unparseable — the build loop categorises those failures.
DrugRow = dict[str, Any]


def _read_drugs_tabular(path: Path, *, separator: str) -> list[DrugRow]:
    """Read a tabular drug catalog (``cid, smiles, cmpd_name_cleaned``)."""
    df = pl.read_csv(
        path,
        separator=separator,
        columns=["cid", "smiles", "cmpd_name_cleaned"],
        ignore_errors=True,
    )
    return [
        {"cid": r["cid"], "smiles": r["smiles"], "name": r["cmpd_name_cleaned"]}
        for r in df.iter_rows(named=True)
    ]


def _read_drugs_json(path: Path) -> list[DrugRow]:
    """Read a flat ``{cid: smiles}`` JSON mapping (no name column)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        msg = f"Expected a JSON object mapping cid->smiles in {path}, got {type(data)}"
        raise ValueError(msg)
    # Keys may carry stray whitespace (e.g. '" 1"'); leave parsing to the loop.
    return [
        {"cid": cid, "smiles": smiles, "name": None} for cid, smiles in data.items()
    ]


def load_drug_records(path: Path) -> list[DrugRow]:
    """Load a drug catalog, dispatching on file extension.

    Returns a list of ``{cid, smiles, name}`` dicts. The values are *raw* —
    validation and failure categorisation happen in :meth:`DrugGraphBuilder.build`.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_drugs_json(path)
    if suffix == ".csv":
        return _read_drugs_tabular(path, separator=",")
    if suffix in (".tsv", ".txt"):
        return _read_drugs_tabular(path, separator="\t")
    msg = f"Unsupported drug catalog format '{suffix}' for {path} (use .tsv/.csv/.json)"
    raise ValueError(msg)


def _build_file_logger(logger_name: str, log_path: Path | None) -> logging.Logger:
    """Return a dedicated, non-propagating logger wiring a fresh FileHandler.

    Idempotent per logger: existing handlers are closed first so repeated builds
    do not duplicate output. With ``log_path=None`` the logger has no file
    handler and (since it never propagates) silently drops its records — the
    in-memory counters and main-log summaries still report them.
    """
    file_logger = logging.getLogger(logger_name)
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    for handler in list(file_logger.handlers):
        file_logger.removeHandler(handler)
        handler.close()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        file_logger.addHandler(file_handler)
    return file_logger


class DrugGraphBuilder:
    """Iterate the drug catalog, render each SMILES to a graph, save to disk.

    Resumable: by default skips drugs whose ``.pt`` file is already present.
    Failures are reported through the standard :mod:`logging` machinery: each
    failure is emitted on a dedicated ``Pharmagen.library.drug_failures`` logger
    whose :class:`~logging.FileHandler` writes to ``logs/library/`` (when a
    ``failures_log`` path is given), and a per-:class:`DrugFailureCategory`
    counter is logged as a summary at the end. The manifest still records every
    failing CID so the next run can retry it.

    Separately, one-hot *saturation* (out-of-vocabulary values encoded as
    all-zeros — hypervalent atoms, exotic bonds) is tallied in
    :attr:`saturation` and reported on a second dedicated logger
    (``saturation_log``). The graph output is unchanged; only observability is
    added.

    When ``strip_salts`` is on (default), multi-fragment drug SMILES are reduced
    to their largest fragment before encoding (dropping salt counterions); the
    number reduced is tracked in :attr:`salts_stripped`.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        force: bool = False,
        failures_log: Path | None = None,
        saturation_log: Path | None = None,
        strip_salts: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.force = force
        self.failures_log = failures_log
        self.saturation_log = saturation_log
        self.strip_salts = strip_salts
        # Tally of failures by nature; populated by ``build``.
        self.failure_counts: Counter[DrugFailureCategory] = Counter()
        # One-hot saturation tally + count of drugs with >=1 saturated feature.
        self.saturation = FeatureSaturation()
        self.drugs_with_saturation = 0
        # Count of multi-fragment drugs reduced to their largest fragment.
        self.salts_stripped = 0
        self._failure_logger = _build_file_logger(_FAILURE_LOGGER_NAME, failures_log)
        self._saturation_logger = _build_file_logger(
            _SATURATION_LOGGER_NAME, saturation_log
        )

    def build(
        self,
        drugs_path: Path,
        *,
        manifest: BuildManifest,
    ) -> tuple[int, int, int]:
        """Build all drug graphs. Returns (built, skipped, failed)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not drugs_path.exists():
            msg = f"Drug catalog not found: {drugs_path}"
            raise FileNotFoundError(msg)

        rows = load_drug_records(drugs_path)
        logger.info("Drug builder: %d rows in %s", len(rows), drugs_path.name)
        self._failure_logger.info(
            "Build run started | source=%s | rows=%d", drugs_path.name, len(rows)
        )
        self._saturation_logger.info(
            "Build run started | source=%s | rows=%d", drugs_path.name, len(rows)
        )

        self.failure_counts = Counter()
        self.saturation = FeatureSaturation()
        self.drugs_with_saturation = 0
        self.salts_stripped = 0
        tally: Counter[str] = Counter()
        for row in tqdm(rows, total=len(rows), desc="Drugs"):
            tally[self._build_one(row, manifest)] += 1

        built, skipped, failed = tally["built"], tally["skipped"], tally["failed"]
        manifest.stats.drugs_built = built
        manifest.stats.drugs_skipped = skipped
        manifest.stats.drugs_failed = failed

        self._log_summary(built, skipped, failed)
        self._log_saturation_summary()
        return built, skipped, failed

    def _build_one(
        self,
        row: DrugRow,
        manifest: BuildManifest,
    ) -> Literal["built", "skipped", "failed"]:
        """Build a single drug graph. Logs failures; returns the outcome."""
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
            manifest.mark_drug_done(cid)
            return "skipped"

        smiles_raw = row.get("smiles")
        smiles = str(smiles_raw).strip() if smiles_raw is not None else ""
        if not smiles:
            return self._fail(
                manifest, cid, name, DrugFailureCategory.MISSING_SMILES, "empty SMILES"
            )
        return self._render_and_save(cid, name, smiles, out_path, manifest)

    def _render_and_save(
        self,
        cid: int,
        name: str,
        smiles: str,
        out_path: Path,
        manifest: BuildManifest,
    ) -> Literal["built", "failed"]:
        """Render a SMILES to a graph and persist it. Logs failures."""
        had_salt = self.strip_salts and "." in smiles
        before = self.saturation.total
        graph = smiles_to_graph(
            smiles, saturation=self.saturation, strip_salts=self.strip_salts
        )
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
                f"parsed to zero atoms: {smiles}",
            )

        if had_salt:
            self.salts_stripped += 1

        events = self.saturation.total - before
        if events:
            self.drugs_with_saturation += 1
            self._saturation_logger.warning(
                "one-hot saturation | cid=%s | name=%s | events=%d | smiles=%s",
                cid,
                name,
                events,
                smiles,
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

        manifest.mark_drug_done(cid)
        return "built"

    def _fail(
        self,
        manifest: BuildManifest,
        cid: int,
        name: str,
        category: DrugFailureCategory,
        reason: str,
    ) -> Literal["failed"]:
        """Tally a failure, record it on the manifest, and log it."""
        self.failure_counts[category] += 1
        if cid >= 0:
            manifest.mark_drug_failed(cid, reason)
        self._failure_logger.warning(
            "drug build failed | cid=%s | name=%s | nature=%s | reason=%s",
            cid,
            name,
            category.value,
            reason,
        )
        return "failed"

    def _log_summary(self, built: int, skipped: int, failed: int) -> None:
        logger.info(
            "Drug builder done: %d built, %d skipped, %d failed.",
            built,
            skipped,
            failed,
        )
        breakdown = ", ".join(
            f"{category.value}={self.failure_counts[category]}"
            for category in DrugFailureCategory
        )
        self._failure_logger.info(
            "Build run finished | built=%d skipped=%d failed=%d | %s",
            built,
            skipped,
            failed,
            breakdown,
        )
        if failed:
            logger.warning("Drug failure breakdown: %s", breakdown)
        if self.strip_salts and self.salts_stripped:
            logger.info(
                "Salt stripping: reduced %d multi-fragment drugs to their largest "
                "fragment.",
                self.salts_stripped,
            )

    def _log_saturation_summary(self) -> None:
        """Log the one-hot saturation tally (silent information loss).

        Always writes a per-feature line to the dedicated saturation log; when
        any saturation occurred, also surfaces a breakdown on the main logger so
        the loss is visible without opening the audit file.
        """
        total = self.saturation.total
        by_feature = ", ".join(
            f"{feature}={self.saturation.by_feature[feature]}"
            for feature in _SATURATING_FEATURES
        )
        self._saturation_logger.info(
            "Build run finished | total_events=%d | drugs_affected=%d | %s",
            total,
            self.drugs_with_saturation,
            by_feature,
        )
        for value, count in self.saturation.by_value.most_common():
            self._saturation_logger.info("  out-of-vocab | %s | count=%d", value, count)

        if total:
            logger.warning(
                "One-hot saturation: %d events across %d drugs (%s). "
                "Hypervalent atoms / exotic bonds are encoded as all-zeros — "
                "see the saturation log for the per-value breakdown.",
                total,
                self.drugs_with_saturation,
                by_feature,
            )
