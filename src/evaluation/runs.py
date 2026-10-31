"""Training on a scenario fold: run names, split creation, fold frames.

A run trained on fold ``k`` of a scenario saves its checkpoints and artifact
bundle under ``"{model}__{scenario}__fold{k}"`` so folds never overwrite each
other or the plain (unsplit) model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.config import get_settings
from src.core import EvaluationError
from src.data.genotype import HGVS_CATALOG_FILE, load_hgvs_catalog
from src.evaluation import DEFAULT_FOLDS
from src.evaluation.entities import EntityResolver, Scenario, add_row_id, file_sha256
from src.evaluation.splits import (
    fold_frames,
    load_split,
    make_split,
    save_split,
    verify_split,
)

if TYPE_CHECKING:
    import polars as pl

    from src.config.models import ModelConfig
    from src.evaluation.schemas import SplitManifest

logger = logging.getLogger(__name__)

NEO4J_SOURCE = "neo4j"
VARIANT_SCENARIOS = frozenset({Scenario.VARIANT_COLD, Scenario.PAIR})
_ABSENT = "absent"


class FoldRef(BaseModel):
    """One fold of one split scenario."""

    scenario: Scenario
    fold: int = Field(..., ge=0)


def run_name(model_name: str, ref: FoldRef | None) -> str:
    """Checkpoint / artifact name of ``model_name`` trained on ``ref``."""
    if ref is None:
        return model_name
    return f"{model_name}__{ref.scenario}__fold{ref.fold}"


def catalog_sha256() -> str:
    """sha256 of the HGVS catalog the splits resolve variants through."""
    path = get_settings().paths.library / HGVS_CATALOG_FILE
    return file_sha256(path) if path.exists() else _ABSENT


def _file_source(source: Path | str) -> Path:
    if str(source) == NEO4J_SOURCE:
        msg = (
            "splits are bound to a data file: export the KG snapshot "
            "('python -m src.kg export') and split that file instead of --neo4j"
        )
        raise EvaluationError(msg)
    return Path(source)


def load_clean_frame(source: Path, cfg: ModelConfig, **kwargs: int) -> pl.DataFrame:
    """Cleaned rows of ``source`` with their ``row_id`` (the split's row order)."""
    from src.model.engine.base import load_and_clean_data

    return add_row_id(load_and_clean_data(source, cfg, **kwargs))


def create_splits(
    source: Path, cfg: ModelConfig, scenarios: list[Scenario], k: int = DEFAULT_FOLDS
) -> list[Path]:
    """Make and persist the k-fold split of every scenario in ``scenarios``.

    Variant-keyed scenarios need the HGVS catalog: without it an rsID and its
    HGVS form would be two entities and could land in different folds.
    """
    needs_catalog = [s for s in scenarios if s in VARIANT_SCENARIOS]
    if needs_catalog and catalog_sha256() == _ABSENT:
        msg = (
            f"{', '.join(map(str, needs_catalog))} need the HGVS catalog: build it "
            "with 'python -m src.data.library --skip-drugs' before splitting"
        )
        raise EvaluationError(msg)
    source = _file_source(source)
    df = load_clean_frame(source, cfg)
    resolver = EntityResolver(load_hgvs_catalog(), drug_col=cfg.features[0])
    data_sha, cat_sha = file_sha256(source), catalog_sha256()
    out = []
    for scenario in scenarios:
        assignment, manifest = make_split(
            df, scenario, resolver, k=k, data_sha256=data_sha, catalog_sha256=cat_sha
        )
        out.append(save_split(assignment, manifest))
    return out


def load_fold_frames(
    source: Path | str, cfg: ModelConfig, ref: FoldRef, **kwargs: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, SplitManifest]:
    """(train, val, test, manifest) of ``ref`` after verifying the split's hashes."""
    source = _file_source(source)
    assignment, manifest = load_split(ref.scenario)
    if ref.fold >= manifest.k:
        msg = f"fold {ref.fold} out of range for {ref.scenario!s} (k={manifest.k})"
        raise EvaluationError(msg)
    verify_split(
        manifest, data_sha256=file_sha256(source), catalog_sha256=catalog_sha256()
    )
    df = load_clean_frame(source, cfg, **kwargs)
    train, val, test = fold_frames(df, assignment, ref.fold)
    logger.info(
        "Fold %s/%d: train=%d val=%d test=%d",
        run_name("", ref),
        manifest.k,
        len(train),
        len(val),
        len(test),
    )
    return train, val, test, manifest


__all__ = [
    "FoldRef",
    "catalog_sha256",
    "create_splits",
    "load_clean_frame",
    "load_fold_frames",
    "run_name",
]
