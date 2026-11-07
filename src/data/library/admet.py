"""Predicted ADMET / enzyme-interaction profile for the drug tower (step c).

Why this exists
---------------
The molecular graph + ``global_feats`` (QSAR physchem + ECFP) encode *structure*.
The pharmacogenetic phenotype, however, depends on how the drug *interacts with
the variant gene product* — metabolism (which CYP isoform), transport, toxicity.
This module predicts that pharmacokinetic / enzyme-interaction profile from SMILES
with **ADMET-AI** (Chemprop D-MPNN models trained on the Therapeutics Data Commons
benchmark) and attaches it to each drug graph as a separate ``admet_feats`` vector,
kept *decoupled* from ``global_feats`` so structure and predicted PK stay distinct.

The 41-dim profile (``DRUG_ADMET_DIM``)
---------------------------------------
Curated from ADMET-AI's ~40 endpoints, **excluding** the physicochemical
descriptors (MolWt, LogP, TPSA, HBD, HBA, QED, stereo centres, Lipinski) that the
QSAR block of ``global_feats`` already carries, and the deterministic structural
alerts (PAINS/BRENK/NIH). What remains is the *predicted* ADMET signal:

    Absorption (8)   HIA, oral bioavailability, solubility, lipophilicity,
                     hydration free energy, Caco-2, PAMPA, P-gp substrate
    Distribution (3) BBB penetration, plasma-protein binding, volume of distribution
    Metabolism (8)   CYP1A2/2C19/2C9/2D6/3A4 inhibition (Veith) +
                     CYP2C9/2D6/3A4 substrate (CarbonMangels) — the PGx-causal core
    Excretion (3)    hepatocyte clearance, microsome clearance, half-life
    Toxicity (19)    hERG, AMES, DILI, ClinTox, carcinogenicity, skin reaction,
                     LD50, and the 12-assay Tox21 nuclear-receptor / stress panel

Representation
--------------
* **Classification endpoints** keep their raw probability in ``[0, 1]`` — we store
  the model's calibrated uncertainty, never a binarised class.
* **Regression endpoints** (clearance, half-life, solubility, …) have open-ended
  units, so we store ADMET-AI's *DrugBank-approved percentile* divided by 100 — a
  bounded, distribution-aware normalisation (rank against approved drugs).

Reproducibility
---------------
Predictions run on each library compound's ``working_smiles`` — the
standardized parent Modephere builds the atom graph from, salts already
removed — and are persisted to a Parquet cache keyed by ``inchikey``. A build
predicts only the compounds the cache lacks (the D-MPNN ensemble load dominates
the cost, so they go in one batch). See :func:`update_admet_table`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
import torch
from rdkit import Chem

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

# --- Curated endpoint selection (order defines the admet_feats layout) --- #
_ABSORPTION = (
    "HIA_Hou",
    "Bioavailability_Ma",
    "Solubility_AqSolDB",
    "Lipophilicity_AstraZeneca",
    "HydrationFreeEnergy_FreeSolv",
    "Caco2_Wang",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
)
_DISTRIBUTION = (
    "BBB_Martins",
    "PPBR_AZ",
    "VDss_Lombardo",
)
_METABOLISM = (
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
)
_EXCRETION = (
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
    "Half_Life_Obach",
)
_TOXICITY = (
    "hERG",
    "AMES",
    "DILI",
    "ClinTox",
    "Carcinogens_Lagunin",
    "Skin_Reaction",
    "LD50_Zhu",
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)

# The full ordered profile. ``admet_feats[i]`` is ADMET_ENDPOINTS[i].
ADMET_ENDPOINTS: tuple[str, ...] = (
    *_ABSORPTION,
    *_DISTRIBUTION,
    *_METABOLISM,
    *_EXCRETION,
    *_TOXICITY,
)

# Regression endpoints (open-ended units) — stored as the DrugBank-approved
# percentile / 100 instead of the raw value. Everything else is a probability.
_REGRESSION_ENDPOINTS: frozenset[str] = frozenset(
    {
        "Solubility_AqSolDB",
        "Lipophilicity_AstraZeneca",
        "HydrationFreeEnergy_FreeSolv",
        "Caco2_Wang",
        "PPBR_AZ",
        "VDss_Lombardo",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ",
        "Half_Life_Obach",
        "LD50_Zhu",
    }
)

#: Length of the per-molecule ADMET vector attached as ``admet_feats``.
DRUG_ADMET_DIM: int = len(ADMET_ENDPOINTS)

_PERCENTILE_SUFFIX = "_drugbank_approved_percentile"
_KEY = "inchikey"
ADMET_COLUMNS: tuple[str, ...] = (_KEY, *ADMET_ENDPOINTS)


def _empty_table() -> pl.DataFrame:
    return pl.DataFrame(
        {_KEY: pl.Series(_KEY, [], dtype=pl.String)}
        | {e: pl.Series(e, [], dtype=pl.Float64) for e in ADMET_ENDPOINTS}
    )


def _select_columns(raw: pl.DataFrame) -> pl.DataFrame:
    """Project ADMET-AI's wide output down to the curated, normalised profile.

    Classification endpoints keep their probability; regression endpoints are
    replaced by their DrugBank percentile / 100. Output columns are exactly
    :data:`ADMET_ENDPOINTS`, in order.
    """
    exprs: list[pl.Expr] = []
    for endpoint in ADMET_ENDPOINTS:
        if endpoint in _REGRESSION_ENDPOINTS:
            source = f"{endpoint}{_PERCENTILE_SUFFIX}"
            if source not in raw.columns:
                msg = f"ADMET-AI output missing expected column {source!r}"
                raise KeyError(msg)
            exprs.append((pl.col(source) / 100.0).alias(endpoint))
        else:
            if endpoint not in raw.columns:
                msg = f"ADMET-AI output missing expected column {endpoint!r}"
                raise KeyError(msg)
            exprs.append(pl.col(endpoint).alias(endpoint))
    return raw.select(exprs)


def compute_admet_table(records: Iterable[tuple[str, str]]) -> pl.DataFrame:
    """Predict the ADMET profile of ``(inchikey, smiles)`` records in one batch.

    Returns ``inchikey`` plus the :data:`ADMET_ENDPOINTS` columns; SMILES RDKit
    cannot parse are dropped (and logged) rather than poisoning the table.
    """
    keys: list[str] = []
    smiles: list[str] = []
    dropped = 0
    for key, raw in dict(records).items():
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            dropped += 1
            continue
        keys.append(key)
        smiles.append(Chem.MolToSmiles(mol))
    if dropped:
        logger.warning(
            "ADMET: dropped %d unparseable SMILES before prediction", dropped
        )
    if not smiles:
        return _empty_table()

    # Imported lazily: heavy (torch/lightning) and only needed at build time.
    from admet_ai import ADMETModel

    # Tensor-core throughput on the descriptor MLP — opt in to the faster matmul.
    torch.set_float32_matmul_precision("high")
    logger.info("ADMET: loading ADMET-AI model ensemble...")
    model = ADMETModel(num_workers=0)
    logger.info("ADMET: predicting %d molecules (batched)...", len(smiles))
    raw_out = model.predict(smiles=smiles)
    if not isinstance(raw_out, pl.DataFrame):
        # ADMET-AI returns a pandas DataFrame (or dict for a single molecule).
        import pandas as pd

        frame = pd.DataFrame([raw_out]) if isinstance(raw_out, dict) else raw_out
        raw_out = pl.from_pandas(frame)
    profile = _select_columns(raw_out)
    return profile.with_columns(pl.Series(_KEY, keys)).select(ADMET_COLUMNS)


def read_admet_table(path: Path) -> pl.DataFrame | None:
    """The cached table, or None when absent or written with other endpoints."""
    if not path.exists():
        return None
    cached = pl.read_parquet(path)
    if tuple(cached.columns) != ADMET_COLUMNS:
        logger.warning(
            "ADMET: cache %s has other columns (%d) than the current endpoint set; "
            "it is recomputed",
            path,
            len(cached.columns),
        )
        return None
    return cached


def update_admet_table(
    records: Iterable[tuple[str, str]], path: Path, *, force: bool = False
) -> pl.DataFrame:
    """Profiles of ``(inchikey, smiles)`` records, predicting only what ``path`` lacks.

    ``force`` recomputes every record. Rows of the cache whose compound is no
    longer in ``records`` are kept (the cache only grows).
    """
    wanted = dict(records)
    cached = None if force else read_admet_table(path)
    have = set() if cached is None else set(cached[_KEY].to_list())
    missing = [(k, smi) for k, smi in wanted.items() if k not in have]
    if not missing and cached is not None:
        logger.info("ADMET: cache %s covers all %d compounds", path, len(wanted))
        return cached
    logger.info(
        "ADMET: %d of %d compounds need a profile (cache=%s, force=%s)",
        len(missing),
        len(wanted),
        path,
        force,
    )
    fresh = compute_admet_table(missing)
    table = fresh if cached is None else pl.concat([cached, fresh])
    table = table.unique(_KEY, keep="last").sort(_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    table.write_parquet(tmp)
    tmp.replace(path)
    logger.info("ADMET: wrote %s (%d compounds)", path, table.height)
    return table


class AdmetProvider:
    """Lookup from InChIKey to its normalised :data:`DRUG_ADMET_DIM` vector.

    Compounds absent from the table (an unparseable SMILES, an ADMET-free build)
    yield a zero vector — a valid "no predicted profile" input — and are tallied
    in :attr:`misses` so the gap is observable.
    """

    def __init__(self, table: pl.DataFrame) -> None:
        self._vectors: dict[str, torch.Tensor] = {}
        feats = torch.from_numpy(table.select(ADMET_ENDPOINTS).to_numpy()).float()
        for key, row in zip(table.get_column(_KEY).to_list(), feats, strict=True):
            self._vectors[str(key)] = row.unsqueeze(0).contiguous()
        self._zero = torch.zeros((1, DRUG_ADMET_DIM), dtype=torch.float)
        self.misses = 0

    @classmethod
    def null(cls) -> AdmetProvider:
        """No entries: every lookup is a zero vector (``--skip-admet`` builds)."""
        return cls(_empty_table())

    @classmethod
    def load(cls, path: Path) -> AdmetProvider:
        """The provider over the cached table at ``path`` (empty, with a warning, if none)."""
        table = read_admet_table(path)
        if table is None:
            logger.warning(
                "ADMET table %s not found or stale: drugs get a zero admet_feats; "
                "build it with 'python -m src.data.library'",
                path,
            )
            return cls.null()
        return cls(table)

    def vector_for(self, inchikey: str) -> torch.Tensor:
        """The ``[1, DRUG_ADMET_DIM]`` vector of ``inchikey`` (zeros if unknown)."""
        vector = self._vectors.get(inchikey)
        if vector is None:
            self.misses += 1
            return self._zero.clone()
        return vector.clone()

    def __contains__(self, inchikey: str) -> bool:
        return inchikey in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)


def zero_admet_vector() -> torch.Tensor:
    """A ``[1, DRUG_ADMET_DIM]`` zero vector (the no-information ADMET input)."""
    return torch.zeros((1, DRUG_ADMET_DIM), dtype=torch.float)


__all__ = [
    "ADMET_COLUMNS",
    "ADMET_ENDPOINTS",
    "DRUG_ADMET_DIM",
    "AdmetProvider",
    "compute_admet_table",
    "read_admet_table",
    "update_admet_table",
    "zero_admet_vector",
]
