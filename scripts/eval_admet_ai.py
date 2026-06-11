"""Evaluate ADMET-AI on a sample of the real Pharmagen SMILES.

Step (c) of the drug-tower enrichment: before integrating ADMET features into
the library build, measure on real data:

  * coverage   — how many SMILES yield valid (non-NaN) predictions,
  * endpoints  — what the output vector looks like (esp. the 5 CYP columns),
  * quality    — basic sanity of the probability/value distributions,
  * throughput — wall-clock per molecule, to extrapolate to ~109k.

This is an exploration utility, not part of the library pipeline. It only reads
``data/dicts/cid_smiles_dict.json`` and prints a report through ``logging``.

Usage::

    uv run python -m scripts.eval_admet_ai --sample 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import setup_logging

logger = logging.getLogger("pharmagen.eval_admet")

SMILES_JSON = Path("data/dicts/cid_smiles_dict.json")
CYP_HINT = "CYP"


def load_sample(path: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Return ``n`` random ``(cid, smiles)`` pairs from the SMILES dictionary."""
    raw: dict[str, str] = json.loads(path.read_text())
    pairs = [
        (cid.strip(), smiles.strip())
        for cid, smiles in raw.items()
        if smiles and smiles.strip()
    ]
    logger.info("Loaded %d non-empty SMILES from %s", len(pairs), path)
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))
    logger.info("Sampled %d SMILES (seed=%d)", len(sample), seed)
    return sample


def run_predictions(smiles: list[str]) -> pd.DataFrame:
    """Run ADMET-AI over ``smiles`` and return the prediction DataFrame."""
    from admet_ai import ADMETModel

    logger.info("Loading ADMETModel (this loads the D-MPNN ensembles)...")
    t0 = time.perf_counter()
    model = ADMETModel()
    logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    preds = model.predict(smiles=smiles)
    elapsed = time.perf_counter() - t0
    if isinstance(preds, dict):  # single-molecule path returns a dict
        preds = pd.DataFrame([preds])
    logger.info(
        "Predicted %d molecules in %.1fs (%.3fs/mol, ~%.1f min for 109k)",
        len(smiles),
        elapsed,
        elapsed / max(len(smiles), 1),
        elapsed / max(len(smiles), 1) * 109_477 / 60,
    )
    return preds


def report(preds: pd.DataFrame, cids: list[str]) -> None:
    """Log coverage, endpoint inventory, CYP columns and distribution sanity."""
    n_rows, n_cols = preds.shape
    logger.info("Prediction matrix: %d rows x %d columns", n_rows, n_cols)

    numeric = preds.select_dtypes(include=[np.number])
    logger.info("Numeric endpoint columns: %d", numeric.shape[1])

    # Coverage: rows fully populated vs. rows with any NaN.
    nan_per_row = numeric.isna().sum(axis=1)
    fully = int((nan_per_row == 0).sum())
    any_nan = int((nan_per_row > 0).sum())
    logger.info(
        "Coverage: %d/%d rows fully predicted, %d rows with >=1 NaN",
        fully,
        n_rows,
        any_nan,
    )
    nan_cols = numeric.columns[numeric.isna().any()].tolist()
    if nan_cols:
        logger.warning("Columns containing NaN: %s", nan_cols)

    # CYP endpoints — the PGx-relevant slice.
    cyp_cols = [c for c in preds.columns if CYP_HINT in c.upper()]
    logger.info("CYP-related columns (%d): %s", len(cyp_cols), cyp_cols)

    # Distribution sanity for a handful of representative endpoints.
    describe = numeric.describe().T[["mean", "std", "min", "max"]]
    logger.info(
        "Endpoint distribution summary (first 15):\n%s", describe.head(15).to_string()
    )
    if cyp_cols:
        cyp_numeric = [c for c in cyp_cols if c in numeric.columns]
        logger.info(
            "CYP endpoint distribution:\n%s",
            numeric[cyp_numeric]
            .describe()
            .T[["mean", "std", "min", "max"]]
            .to_string(),
        )

    # Spot-check a couple of molecules end-to-end.
    preview = preds.copy()
    preview.insert(0, "cid", cids[: len(preview)])
    logger.info(
        "Sample rows (cid + first CYP cols):\n%s",
        preview[["cid", *cyp_cols[:5]]].head(5).to_string(index=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ADMET-AI on real SMILES")
    parser.add_argument(
        "--sample", type=int, default=200, help="number of SMILES to test"
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional CSV path to dump the prediction matrix",
    )
    args = parser.parse_args()

    setup_logging(name="eval_admet", console_level=logging.INFO)

    sample = load_sample(SMILES_JSON, args.sample, args.seed)
    cids = [cid for cid, _ in sample]
    smiles = [smi for _, smi in sample]

    preds = run_predictions(smiles)
    report(preds, cids)

    if args.save is not None:
        out = preds.copy()
        out.insert(0, "cid", cids[: len(out)])
        out.insert(1, "smiles", smiles[: len(out)])
        out.to_csv(args.save, index=False)
        logger.info("Wrote prediction matrix to %s", args.save)


if __name__ == "__main__":
    main()
