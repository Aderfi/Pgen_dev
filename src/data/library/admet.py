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
Predictions are computed once over the whole catalog (the D-MPNN ensemble load
dominates the cost) and persisted to a Parquet cache keyed by ``cid``; a second
build reuses the cache. See :func:`load_or_build_admet_table`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl
import torch
from rdkit import Chem

from src.data.library.drugs import _largest_fragment

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
_CID_COL = "cid"


def records_from_rows(rows: Iterable[dict]) -> list[tuple[int, str]]:
    """Extract clean ``(cid, smiles)`` pairs from raw drug-catalog rows.

    Mirrors the cid/SMILES validation in ``DrugGraphBuilder._build_one`` (integer
    CID, non-empty SMILES); rows failing either are skipped here and surface as
    proper build failures downstream.
    """
    records: list[tuple[int, str]] = []
    for row in rows:
        cid_raw = row.get("cid")
        smiles_raw = row.get("smiles")
        try:
            cid = int(str(cid_raw).strip())
        except TypeError, ValueError:
            continue
        smiles = str(smiles_raw).strip() if smiles_raw is not None else ""
        if smiles:
            records.append((cid, smiles))
    return records


def _canonical_smiles(smiles: str, *, strip_salts: bool) -> str | None:
    """Parse, optionally strip salts, and canonicalise a SMILES.

    ADMET-AI is run on the same moiety the graph encodes — the largest fragment
    when ``strip_salts`` is on — so the predicted profile matches the structure.
    Returns ``None`` if RDKit cannot parse the input.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if strip_salts:
        mol = _largest_fragment(mol)
    return Chem.MolToSmiles(mol)


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


def compute_admet_table(
    records: Iterable[tuple[int, str]],
    *,
    strip_salts: bool = True,
) -> pl.DataFrame:
    """Predict the ADMET profile for ``(cid, smiles)`` records.

    Loads the ADMET-AI D-MPNN ensemble once and predicts over every unique,
    parseable structure in a single batched pass (the model load dominates the
    cost, so per-molecule prediction would be pathologically slow). Returns a
    DataFrame with a ``cid`` column plus the :data:`ADMET_ENDPOINTS` columns;
    unparseable SMILES are dropped (and logged) rather than poisoning the table.
    """
    # Imported lazily: heavy (torch/lightning) and only needed at build time.
    from admet_ai import ADMETModel

    # Tensor-core throughput on the descriptor MLP — opt in to the faster matmul.
    torch.set_float32_matmul_precision("high")

    cids: list[int] = []
    smiles: list[str] = []
    dropped = 0
    for cid, raw_smiles in records:
        canonical = _canonical_smiles(raw_smiles, strip_salts=strip_salts)
        if canonical is None:
            dropped += 1
            continue
        cids.append(cid)
        smiles.append(canonical)

    if dropped:
        logger.warning(
            "ADMET: dropped %d unparseable SMILES before prediction", dropped
        )
    if not smiles:
        logger.warning("ADMET: no parseable SMILES; returning empty table")
        return pl.DataFrame({_CID_COL: []}).with_columns(
            [pl.lit(0.0).alias(e) for e in ADMET_ENDPOINTS]
        )

    logger.info("ADMET: loading ADMET-AI model ensemble...")
    model = ADMETModel(num_workers=0)
    logger.info("ADMET: predicting %d molecules (batched)...", len(smiles))
    raw = model.predict(smiles=smiles)
    if not isinstance(raw, pl.DataFrame):
        # ADMET-AI returns a pandas DataFrame (or dict for a single molecule).
        import pandas as pd

        raw = pl.from_pandas(pd.DataFrame([raw]) if isinstance(raw, dict) else raw)

    profile = _select_columns(raw)
    table = profile.with_columns(pl.Series(_CID_COL, cids)).select(
        [_CID_COL, *ADMET_ENDPOINTS]
    )
    # Collapse duplicate CIDs (same drug listed twice) to a single row.
    table = table.unique(subset=[_CID_COL], keep="first")
    logger.info("ADMET: built profile table for %d unique CIDs", table.height)
    return table


def load_or_build_admet_table(
    records: Iterable[tuple[int, str]],
    cache_path: Path,
    *,
    strip_salts: bool = True,
    force: bool = False,
) -> pl.DataFrame:
    """Return the ADMET profile table, reusing the Parquet cache when valid.

    The cache at ``cache_path`` is reused unless ``force`` is set or its schema
    no longer matches the current :data:`ADMET_ENDPOINTS` (e.g. the endpoint set
    changed) — in which case it is recomputed and rewritten.
    """
    expected = {_CID_COL, *ADMET_ENDPOINTS}
    if cache_path.exists() and not force:
        cached = pl.read_parquet(cache_path)
        if set(cached.columns) == expected:
            logger.info("ADMET: reusing cache %s (%d rows)", cache_path, cached.height)
            return cached
        logger.warning(
            "ADMET: cache %s schema mismatch (%d cols) — recomputing",
            cache_path,
            len(cached.columns),
        )

    table = compute_admet_table(records, strip_salts=strip_salts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(cache_path)
    logger.info("ADMET: wrote cache %s (%d rows)", cache_path, table.height)
    return table


class AdmetProvider:
    """Lookup from ``cid`` to its normalised :data:`DRUG_ADMET_DIM` vector.

    Missing CIDs (drug absent from the table, e.g. an unparseable SMILES) yield a
    zero vector — a valid "no predicted profile" input — and are tallied in
    :attr:`misses` so the gap is observable at build time.
    """

    def __init__(self, table: pl.DataFrame) -> None:
        self._vectors: dict[int, torch.Tensor] = {}
        feats = torch.from_numpy(table.select(ADMET_ENDPOINTS).to_numpy()).float()
        for cid, row in zip(table.get_column(_CID_COL).to_list(), feats, strict=True):
            self._vectors[int(cid)] = row.unsqueeze(0).contiguous()
        self._zero = torch.zeros((1, DRUG_ADMET_DIM), dtype=torch.float)
        self.misses = 0

    @classmethod
    def null(cls) -> AdmetProvider:
        """A provider with no entries — every lookup returns a zero vector.

        Used for ADMET-free builds (``--skip-admet``): the graph schema stays
        complete (``admet_feats`` present) but carries no predicted signal.
        """
        empty = pl.DataFrame(
            {_CID_COL: pl.Series(_CID_COL, [], dtype=pl.Int64)},
        ).with_columns([pl.lit(0.0).alias(e) for e in ADMET_ENDPOINTS])
        return cls(empty)

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[int, str]],
        cache_path: Path,
        *,
        strip_salts: bool = True,
        force: bool = False,
    ) -> AdmetProvider:
        """Build a provider, computing/loading the table via the Parquet cache."""
        table = load_or_build_admet_table(
            records, cache_path, strip_salts=strip_salts, force=force
        )
        return cls(table)

    def vector_for(self, cid: int) -> torch.Tensor:
        """Return the ``[1, DRUG_ADMET_DIM]`` ADMET vector for ``cid`` (or zeros)."""
        vector = self._vectors.get(cid)
        if vector is None:
            self.misses += 1
            return self._zero.clone()
        return vector.clone()

    def __contains__(self, cid: int) -> bool:
        return cid in self._vectors


def zero_admet_vector() -> torch.Tensor:
    """A ``[1, DRUG_ADMET_DIM]`` zero vector (the no-information ADMET input)."""
    return torch.zeros((1, DRUG_ADMET_DIM), dtype=torch.float)


__all__ = [
    "ADMET_ENDPOINTS",
    "DRUG_ADMET_DIM",
    "AdmetProvider",
    "compute_admet_table",
    "load_or_build_admet_table",
    "records_from_rows",
    "zero_admet_vector",
]
