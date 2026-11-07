"""Shared bootstrap for the training, tuning, and inference engines.

``train_pipeline`` (standard training), ``PGenTuner`` (Optuna) and
``PGenPredictor`` (inference) all need the same plumbing. It lives here once:

* :func:`prepare_training_data` — load (TSV/CSV/parquet file or Neo4j), clean,
  split, build the paired datasets, infer tower widths and axis specs.
* :func:`build_gnn_model` — ``PharmagenConfig.from_dims`` + ``PharmagenTwoTower``.
* :func:`build_trainer` — loss modules, optimizer, scheduler and ``Trainer``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from src.config import get_axes_config, get_settings
from src.core import ConfigValidator, DataError, DataValidator, ModelError
from src.data.cache import GraphCache, GraphDims
from src.data.cleaning import PharmacogenomicCleaner
from src.data.collator import DoubleTowerCollater
from src.data.datasets import PRELOAD_THRESHOLD, DoubleTowerDataset
from src.data.genotype import (
    GenotypeGraphBuilder,
    build_gene_vocab,
    gene_vocab_size,
    load_hgvs_catalog,
)
from src.data.loaders import TabularLoader
from src.model.architectures.assembly import infer_axis_specs
from src.model.architectures.config import PharmagenConfig
from src.model.architectures.model import PharmagenTwoTower
from src.model.checkpoint import CheckpointManager
from src.model.factories import build_optimizer, build_scheduler
from src.model.losses import CompositionalLabelLoss, MultiTaskLoss
from src.model.training import Trainer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.config.models import ModelConfig, TrainingParams
    from src.evaluation.entities import EntityResolver, Scenario
    from src.evaluation.runs import FoldRef
    from src.model.architectures.config import AxisSpec

logger = logging.getLogger(__name__)

DEFAULT_NUM_WORKERS = 4
MIN_DATASET_SIZE = 100
GENE_COLUMN = "gene"
NEO4J_SOURCE = "neo4j"

#: Where training rows come from: a TSV/CSV/parquet file, or the live KG.
DataSource = Path | Literal["neo4j"]


# --------------------------------------------------------------------------- #
# Device + config-derived values
# --------------------------------------------------------------------------- #


def resolve_device(override: str | None = None) -> torch.device:
    """``torch.device(override)`` if given, else CUDA when available, else CPU."""
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tower_dims(cfg: ModelConfig) -> GraphDims:
    """Graph feature widths for ``cfg`` (``models.toml`` extras over defaults)."""
    return GraphDims.from_extras(cfg.extras)


def model_switches(cfg: ModelConfig) -> dict[str, bool]:
    """Structural ablation switches declared in ``models.toml``."""
    return {
        key: bool(cfg.extras.get(key, False))
        for key in ("use_polypharmacy", "use_cross_attention", "use_pharmacophore")
    }


# --------------------------------------------------------------------------- #
# Data loading + validation
# --------------------------------------------------------------------------- #


def load_frame(source: DataSource, cfg: ModelConfig) -> pl.DataFrame:
    """Raw training rows from a file (TSV/CSV/parquet) or from Neo4j.

    Projected onto ``cfg.cols``; optional columns (``alleles``,
    ``variant_annotation_id``) are kept only when the source has them — the
    required features/targets are validated afterwards.
    """
    if source == NEO4J_SOURCE:
        from src.kg.training import load_training_rows

        df = load_training_rows()
    else:
        df = TabularLoader.load(source)
    return df.select([c for c in cfg.cols if c in df.columns]) if cfg.cols else df


def load_and_clean_data(
    source: DataSource,
    cfg: ModelConfig,
    *,
    min_size: int = MIN_DATASET_SIZE,
) -> pl.DataFrame:
    """Load ``source`` and run ``PharmacogenomicCleaner``, validating the result.

    Raises DataError on anything that would prevent a meaningful training run.
    """
    try:
        df = PharmacogenomicCleaner(
            multi_label_cols=get_settings().multi_label_set,
        ).clean(load_frame(source, cfg), stratify_col=cfg.stratify_col)
    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Failed to load data from {source}: {e}") from e

    if len(df) < min_size:
        raise DataError(f"Dataset too small: {len(df)} samples (minimum: {min_size})")

    try:
        ConfigValidator.validate_data_columns(df.columns, cfg.features, cfg.targets)
    except ValueError as e:
        raise DataError(str(e)) from e

    missing = DataValidator.check_missing_values(
        df, cfg.features + cfg.targets, threshold=0.5
    )
    if any(frac > 0.1 for frac in missing.values()):
        logger.warning("High missing values detected: %s", missing)
    return df


def stratified_split(
    df: pl.DataFrame,
    validation_split: float,
    *,
    seed: int | None = None,
    members: Mapping[str, frozenset[str]] | None = None,
    resolver: EntityResolver | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split ``df`` into (train, val), stratifying on ``_stratify`` if present.

    Class rows (``is_class_row``) go to train only, minus those that would
    leak against val (:func:`src.evaluation.class_rows.leaking_class_rows`;
    ``members`` are the classes' ATC members, ``resolver`` resolves variants).
    """
    from src.evaluation.class_rows import class_row_mask, leaking_class_rows

    seed = seed if seed is not None else get_settings().seed
    is_class = class_row_mask(df)
    rest = df.filter(~is_class) if is_class.any() else df
    stratify = rest["_stratify"] if "_stratify" in rest.columns else None
    train_df, val_df = train_test_split(
        rest, test_size=validation_split, stratify=stratify, random_state=seed
    )
    if is_class.any():
        if resolver is None:
            from src.evaluation.entities import EntityResolver

            resolver = EntityResolver(load_hgvs_catalog(), drug_col="drugs_chembl")
        classes = df.filter(is_class)
        leaks = leaking_class_rows(classes, val_df, resolver, members or {})
        train_df = pl.concat([train_df, classes.filter(~leaks)], how="vertical")
        logger.info(
            "Class rows: %d in train, %d dropped (leak against val)",
            classes.height - int(leaks.sum()),
            int(leaks.sum()),
        )
    return train_df, val_df


def fold_drug_caches(
    cache: GraphCache,
    scenario: Scenario | None,
    holdout: Iterable[pl.DataFrame],
    *,
    drug_col: str,
) -> tuple[GraphCache, GraphCache]:
    """``(train_cache, val_cache)``: in ``drug_cold`` the training view's class
    pools leave out every drug of the held-out rows; otherwise both are ``cache``.
    """
    from src.data.datasets import ENTITY_SEP
    from src.data.library.bridge import is_class_key
    from src.evaluation.entities import Scenario as _Scenario

    if scenario is not _Scenario.DRUG_COLD:
        return cache, cache
    held = {
        key
        for frame in holdout
        for value in frame[drug_col].drop_nulls().to_list()
        for key in str(value).split(ENTITY_SEP)
        if key and not is_class_key(key)
    }
    logger.info("drug_cold: %d held-out drugs removed from class pools", len(held))
    return cache.without_members(held), cache


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def build_two_tower_datasets(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    cfg: ModelConfig,
    dims: GraphDims,
    *,
    genotype_builder: GenotypeGraphBuilder,
    preload_ram: bool | None = None,
    drug_cache: GraphCache | None = None,
    val_drug_cache: GraphCache | None = None,
) -> tuple[DoubleTowerDataset, DoubleTowerDataset]:
    """Paired train/val datasets; val reuses train's fitted target encoders.

    Both splits share ``genotype_builder`` (one HGVS catalog, one gene
    vocabulary fitted on the training split) and, by default, one drug cache
    (``drug_cache``; ``val_drug_cache`` defaults to it).
    """
    if drug_cache is None:
        drug_cache = GraphCache.from_settings(dims=dims)
    val_drug_cache = val_drug_cache or drug_cache
    if preload_ram is None:
        preload_ram = len(train_df) < PRELOAD_THRESHOLD
    common = {
        "drug_col": cfg.features[0],
        "geno_col": cfg.features[1],
        "target_cols": cfg.targets,
        "multilabel_cols": list(get_settings().multi_label_set),
        "gene_col": GENE_COLUMN,
        "genotype_builder": genotype_builder,
        "preload_ram": preload_ram,
        "dims": dims,
    }
    train_dataset = DoubleTowerDataset(df=train_df, drug_cache=drug_cache, **common)
    val_dataset = DoubleTowerDataset(
        df=val_df,
        encoders=train_dataset.target_encoder.encoders,
        drug_cache=val_drug_cache,
        **common,
    )
    return train_dataset, val_dataset


def infer_dimensions(train_dataset: DoubleTowerDataset) -> tuple[int, int]:
    """Probe one sample for the (drug, genotype) node feature widths."""
    try:
        sample = train_dataset[0]
        drug = sample["drug_data"]
        if isinstance(drug, list):  # a row's entities: (kind, molecules)
            drug = drug[0][1][0]
        drug_dim = drug.x.shape[1]
        geno_dim = sample["geno_data"].x.shape[1]
    except Exception as e:
        raise DataError(f"Failed to infer dimensions from dataset: {e}") from e
    if drug_dim <= 0 or geno_dim <= 0:
        raise DataError(
            f"Invalid inferred dimensions: drug={drug_dim}, geno={geno_dim}"
        )
    return drug_dim, geno_dim


@dataclass
class PreparedData:
    """Everything training needs to know about the data, computed once."""

    train: DoubleTowerDataset
    val: DoubleTowerDataset
    dims: GraphDims
    drug_dim: int
    geno_dim: int
    axes: dict[str, AxisSpec]
    gene_vocab: dict[str, int]
    split_sha256: str | None = None


def prepare_training_data(
    cfg: ModelConfig,
    source: DataSource,
    *,
    validation_split: float = 0.2,
    seed: int | None = None,
    preload_ram: bool | None = None,
    min_size: int = MIN_DATASET_SIZE,
    split: FoldRef | None = None,
) -> PreparedData:
    """Load → clean → split → gene vocab → datasets → tower widths → axis specs.

    With ``split`` the train/val rows come from that persisted scenario fold
    (``src.evaluation``); its test rows are never loaded into a dataset.
    """
    from src.evaluation.entities import EntityResolver

    dims = tower_dims(cfg)
    cache = GraphCache.from_settings(dims=dims)
    split_sha: str | None = None
    if split is not None:
        from src.evaluation.runs import load_fold_frames

        train_df, val_df, test_df, manifest = load_fold_frames(
            source, cfg, split, min_size=min_size
        )
        split_sha = manifest.split_sha256
        train_cache, val_cache = fold_drug_caches(
            cache, split.scenario, [val_df, test_df], drug_col=cfg.features[0]
        )
    else:
        df = load_and_clean_data(source, cfg, min_size=min_size)
        train_df, val_df = stratified_split(
            df,
            validation_split,
            seed=seed,
            members=cache.class_members,
            resolver=EntityResolver(load_hgvs_catalog(), drug_col=cfg.features[0]),
        )
        train_cache = val_cache = cache
    logger.info(
        "Loaded samples from %s (train=%d, val=%d, split=%s)",
        source,
        len(train_df),
        len(val_df),
        split,
    )

    gene_vocab = build_gene_vocab(train_df[GENE_COLUMN].to_list())
    logger.info("Gene vocabulary: %d genes (+UNK)", len(gene_vocab))
    builder = GenotypeGraphBuilder(load_hgvs_catalog(), gene_vocab)
    train, val = build_two_tower_datasets(
        train_df,
        val_df,
        cfg,
        dims,
        genotype_builder=builder,
        preload_ram=preload_ram,
        drug_cache=train_cache,
        val_drug_cache=val_cache,
    )
    drug_dim, geno_dim = infer_dimensions(train)
    axes = infer_axis_specs(
        train.target_encoder.encoders,
        train.targets,
        get_settings().multi_label_set,
        get_axes_config(),
    )
    logger.info(
        "Inferred dims drug=%d geno=%d, axes=%s", drug_dim, geno_dim, list(axes)
    )
    return PreparedData(
        train, val, dims, drug_dim, geno_dim, axes, gene_vocab, split_sha
    )


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #


def build_train_val_loaders(
    train_dataset: DoubleTowerDataset,
    val_dataset: DoubleTowerDataset,
    batch_size: int,
    *,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Paired train + val DataLoaders with the project's standard knobs."""
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = (
            2
            if len(train_dataset) < PRELOAD_THRESHOLD
            else min(DEFAULT_NUM_WORKERS, max(cpu_count - 1, 1))
        )
    if num_workers > 0:
        # Workers receive the datasets' tensors as shared-memory handles; the
        # default fd-passing strategy hits the descriptor limit with thousands
        # of cached graphs.
        torch.multiprocessing.set_sharing_strategy("file_system")
    common = {
        "batch_size": batch_size,
        "collate_fn": DoubleTowerCollater(),
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )


# --------------------------------------------------------------------------- #
# Model + trainer
# --------------------------------------------------------------------------- #


def build_gnn_model(
    *,
    dims: GraphDims,
    drug_dim: int,
    geno_dim: int,
    axes: dict[str, AxisSpec],
    params: TrainingParams,
    device: torch.device,
    switches: dict[str, bool] | None = None,
    gene_vocab: dict[str, int] | None = None,
) -> nn.Module:
    """Instantiate ``PharmagenTwoTower`` with the inferred / configured shapes."""
    try:
        config = PharmagenConfig.from_dims(
            dims=dims,
            drug_dim=drug_dim,
            geno_dim=geno_dim,
            axes=axes,
            params=params,
            switches=switches,
            gene_vocab_size=gene_vocab_size(gene_vocab)
            if gene_vocab is not None
            else 0,
        )
        return PharmagenTwoTower(config).to(device)
    except Exception as e:
        raise ModelError(f"Failed to create model: {e}") from e


def build_trainer(
    model: nn.Module,
    cfg: ModelConfig,
    axes: dict[str, AxisSpec],
    params: TrainingParams,
    device: torch.device,
    *,
    checkpoint_name: str | None = None,
) -> Trainer:
    """Losses + optimizer + scheduler + :class:`Trainer`.

    With ``checkpoint_name`` the trainer checkpoints improvements, compiles the
    model and shows progress (standard training); without it, it is a light
    in-memory trainer (Optuna trials).
    """
    multitask_loss = MultiTaskLoss(axes).to(device)
    optimizer = build_optimizer(
        model=model, params=params, uncertainty_module=multitask_loss
    )
    standard = checkpoint_name is not None
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=build_scheduler(optimizer, params),
        device=device,
        target_cols=cfg.targets,
        multi_label_cols=get_settings().multi_label_set,
        multitask_loss=multitask_loss,
        compose_loss=CompositionalLabelLoss(),
        compose_weight=params.compose_loss_weight,
        checkpoint=CheckpointManager(checkpoint_name) if standard else None,
        compile_model=standard,
        show_progress=standard,
    )


__all__ = [
    "GENE_COLUMN",
    "MIN_DATASET_SIZE",
    "NEO4J_SOURCE",
    "DataSource",
    "PreparedData",
    "build_gnn_model",
    "build_train_val_loaders",
    "build_trainer",
    "build_two_tower_datasets",
    "infer_dimensions",
    "load_and_clean_data",
    "load_frame",
    "model_switches",
    "prepare_training_data",
    "resolve_device",
    "fold_drug_caches",
    "stratified_split",
    "tower_dims",
]
