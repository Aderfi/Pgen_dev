"""Drug graph builder — SMILES → PyG ``Data`` molecular graph.

Output schema (frozen — must stay in sync with the trained model):
    Node features  (25): atomic_num/100, degree (raw),
                         one-hot degree[0..4],
                         one-hot formal_charge[-2..2],
                         one-hot hybridization[SP, SP2, SP3],
                         one-hot total_Hs[0..4],
                         one-hot chiral_tag[unspec, CW, CCW],
                         is_aromatic, mass*0.01
    Edge features  (7):  one-hot bond_type[SINGLE, DOUBLE, TRIPLE, AROMATIC],
                         is_conjugated, in_ring, has_stereo
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data as PyGData
from tqdm.auto import tqdm

from src.data.library.manifest import BuildManifest

logger = logging.getLogger(__name__)


_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')

# Schema dimensions exposed for sanity checks — see DOCSTRING above.
DRUG_NODE_DIM: int = 25
DRUG_EDGE_DIM: int = 7

_DEGREE_BINS = [0, 1, 2, 3, 4]
_CHARGE_BINS = [-2, -1, 0, 1, 2]
_HS_BINS = [0, 1, 2, 3, 4]
_HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
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


def safe_filename(name: str, max_length: int = 100) -> str:
    """Strip filesystem-unsafe characters and cap length.

    Used to build ``<cid>_<safe_name>.pt`` paths. Keeps drug names readable
    while ensuring they round-trip through Linux/Windows filesystems.
    """
    cleaned = _ILLEGAL_FILENAME_CHARS.sub("_", str(name).strip()).replace(" ", "_")
    return cleaned[:max_length]


def _one_hot(value: Any, choices: list[Any]) -> list[float]:
    encoding = [0.0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1.0
    return encoding


def smiles_to_graph(smiles: str) -> PyGData | None:
    """Convert a SMILES string into a PyG molecular graph.

    Returns ``None`` if RDKit can't parse the SMILES.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Nodes (atoms) — 25 features each.
    atom_features: list[list[float]] = []
    for atom in mol.GetAtoms():
        feats: list[float] = [atom.GetAtomicNum() / 100.0, float(atom.GetDegree())]
        feats += _one_hot(atom.GetDegree(), _DEGREE_BINS)
        feats += _one_hot(atom.GetFormalCharge(), _CHARGE_BINS)
        feats += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        feats += _one_hot(atom.GetTotalNumHs(), _HS_BINS)
        feats += _one_hot(atom.GetChiralTag(), _CHIRAL_TAGS)
        feats.append(1.0 if atom.GetIsAromatic() else 0.0)
        feats.append(atom.GetMass() * 0.01)
        atom_features.append(feats)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges (bonds) — 7 features each, bidirectional.
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feats: list[float] = _one_hot(bond.GetBondType(), _BOND_TYPES)
        bond_feats += [
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0,
            1.0 if bond.GetStereo() != rdchem.BondStereo.STEREONONE else 0.0,
        ]
        edge_indices += [[start, end], [end, start]]
        edge_attrs += [bond_feats, bond_feats]

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, DRUG_EDGE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _read_drugs_tsv(path: Path) -> pl.DataFrame:
    """Read the drug catalog TSV with the columns the builder expects."""
    return pl.read_csv(
        path,
        separator="\t",
        columns=["cid", "smiles", "cmpd_name_cleaned"],
        ignore_errors=True,
    )


class DrugGraphBuilder:
    """Iterate the drug catalog, render each SMILES to a graph, save to disk.

    Resumable: by default skips drugs whose ``.pt`` file is already present.
    Failures are recorded both in the manifest (for the next run to retry)
    and a free-text log file (for human review).
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        force: bool = False,
        failures_log: Path | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.force = force
        self.failures_log = failures_log

    def build(
        self,
        drugs_tsv: Path,
        *,
        manifest: BuildManifest,
    ) -> tuple[int, int, int]:
        """Build all drug graphs. Returns (built, skipped, failed)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not drugs_tsv.exists():
            msg = f"Drugs TSV not found: {drugs_tsv}"
            raise FileNotFoundError(msg)

        df = _read_drugs_tsv(drugs_tsv)
        logger.info("Drug builder: %d rows in %s", len(df), drugs_tsv.name)

        built = skipped = failed = 0
        failures: list[tuple[int, str, str]] = []

        rows: Iterable[dict[str, Any]] = df.iter_rows(named=True)
        for row in tqdm(rows, total=len(df), desc="Drugs"):
            cid_raw = row["cid"]
            try:
                cid = int(cid_raw)
            except (TypeError, ValueError):
                failed += 1
                failures.append((-1, str(cid_raw), "non-integer CID"))
                continue

            raw_name = str(row["cmpd_name_cleaned"]).strip()
            out_path = self.output_dir / f"{cid}_{safe_filename(raw_name)}.pt"

            if out_path.exists() and not self.force:
                skipped += 1
                manifest.mark_drug_done(cid)
                continue

            smiles = str(row["smiles"]).strip()
            graph = smiles_to_graph(smiles)
            if graph is None:
                failed += 1
                failures.append((cid, raw_name, "Invalid SMILES"))
                manifest.mark_drug_failed(cid, "Invalid SMILES")
                continue

            graph.cid = cid
            graph.name = raw_name
            graph.smiles = smiles
            try:
                torch.save(graph, out_path)
            except OSError as e:
                failed += 1
                failures.append((cid, raw_name, f"save error: {e}"))
                manifest.mark_drug_failed(cid, str(e))
                continue

            built += 1
            manifest.mark_drug_done(cid)

        manifest.stats.drugs_built = built
        manifest.stats.drugs_skipped = skipped
        manifest.stats.drugs_failed = failed

        if failures and self.failures_log is not None:
            self._append_failures(failures)

        logger.info(
            "Drug builder done: %d built, %d skipped, %d failed.",
            built,
            skipped,
            failed,
        )
        return built, skipped, failed

    def _append_failures(self, failures: list[tuple[int, str, str]]) -> None:
        assert self.failures_log is not None
        self.failures_log.parent.mkdir(parents=True, exist_ok=True)
        with self.failures_log.open("a", encoding="utf-8") as f:
            f.write("CID\tName\tReason\n")
            for cid, name, reason in failures:
                f.write(f"{cid}\t{name}\t{reason}\n")
        logger.warning(
            "Logged %d drug failures to %s", len(failures), self.failures_log
        )
