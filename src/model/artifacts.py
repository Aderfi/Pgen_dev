"""The training-artifact bundle: what inference needs to rebuild a trained model.

Written by ``train_pipeline`` next to the checkpoints, read by
``PGenPredictor``. Schema v3::

    {
        "encoders": {target: LabelEncoder | MultiLabelBinarizer},
        "drug_dim": int, "geno_dim": int,   # inferred from real graphs
        "dims": GraphDims as dict,          # edge / auxiliary widths
        "axis_specs": {axis: AxisSpec as dict},
        "label_table": {"tuples": [[int, ...]], "labels": [str]},
        "switches": {"use_polypharmacy": bool, "use_cross_attention": bool},
        "gene_vocab": {gene_symbol: embedding index},   # HGVS genotype tower
        "split": {"scenario", "fold", "split_sha256"} | None,  # scenario-fold runs
        "schema_version": 5,
    }

Older bundles still load (v3/v2/v1 and the legacy plain ``{target: encoder}``
dict), but models trained before the HGVS genotype tower (schema < 4) no longer
match the architecture and must be retrained.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import joblib

from src.config import get_settings
from src.core import EncoderError

if TYPE_CHECKING:
    from pathlib import Path

    from src.data.cache import GraphDims
    from src.model.architectures.config import AxisSpec

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5
_KNOWN_VERSIONS = (1, 2, 3, 4, 5)


def artifacts_path(model_name: str) -> Path:
    """``paths.encoders / encoders_{model_name}.pkl`` (``model_name`` may be a run name)."""
    return get_settings().paths.encoders / f"encoders_{model_name}.pkl"


def save_training_artifacts(
    model_name: str,
    *,
    encoders: dict[str, Any],
    drug_dim: int,
    geno_dim: int,
    dims: GraphDims,
    axes: dict[str, AxisSpec],
    switches: dict[str, bool],
    gene_vocab: dict[str, int],
    label_table: dict[str, list] | None = None,
    split_info: dict[str, str | int] | None = None,
) -> Path:
    """Persist the training bundle for ``model_name`` and return its path."""
    path = artifacts_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "encoders": encoders,
        "drug_dim": int(drug_dim),
        "geno_dim": int(geno_dim),
        "dims": asdict(dims),
        "axis_specs": {name: spec.model_dump() for name, spec in axes.items()},
        "label_table": label_table or {"tuples": [], "labels": []},
        "switches": switches,
        "gene_vocab": gene_vocab,
        "split": split_info,
        "schema_version": SCHEMA_VERSION,
    }
    joblib.dump(bundle, path)
    logger.info(
        "Persisted training artifacts v%d (encoders=%d, drug_dim=%d, geno_dim=%d, "
        "axes=%d) to %s",
        SCHEMA_VERSION,
        len(encoders),
        drug_dim,
        geno_dim,
        len(axes),
        path,
    )
    return path


def load_training_artifacts(path: Path) -> dict[str, Any]:
    """Load a bundle, normalized to a dict with ``encoders`` + ``schema_version``.

    ``joblib.load`` unpickles a local artifact written by this project's own
    training pipeline — the same trust boundary as the checkpoint I/O.
    """
    if not path.exists():
        raise FileNotFoundError(f"Encoders file not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise EncoderError(
            f"Encoder artifact at {path} is not a dict (got {type(payload).__name__})"
        )
    if isinstance(payload.get("encoders"), dict):
        bundle = dict(payload)
        if bundle.get("schema_version") not in _KNOWN_VERSIONS:
            bundle["schema_version"] = 1
        return bundle
    logger.warning(
        "Loading legacy encoders pickle at %s — drug_dim / geno_dim will fall "
        "back to models.toml defaults; retrain to refresh the bundle.",
        path,
    )
    return {"encoders": payload, "schema_version": 1}


__all__ = [
    "SCHEMA_VERSION",
    "artifacts_path",
    "load_training_artifacts",
    "save_training_artifacts",
]
