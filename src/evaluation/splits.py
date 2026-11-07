"""Grouped k-fold split scenarios: assignment, persistence, verification.

Each scenario deals its entity groups (drugs, variants, genes, articles,
pairs — see :mod:`src.evaluation.entities`) into ``k`` folds, so every entity
is in test exactly once. In fold ``f`` a row is

* ``test``  — all its entities are in fold ``f``;
* ``drop``  — some but not all are (strict cold: no partial leakage);
* otherwise ``val`` / ``train`` by the same rule against the fold's val
  entities (≈ ``val_fraction`` of the non-test rows), or ``drop`` if mixed.

``random`` is row-level stratified k-fold over the label powerset. Class rows
(``is_class_row``, drug classes) are never dealt: they are ``train`` in every
fold, or ``drop`` where they would leak (:mod:`src.evaluation.class_rows`).
A split is persisted under ``data/splits/<scenario>/`` (``assignment.parquet`` +
``manifest.json``) and bound to the sha256 of the data file and of the HGVS
catalog: using it with different inputs is an error.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.config import get_settings
from src.core import EvaluationError
from src.evaluation import DEFAULT_FOLDS, SPLITS_DIRNAME, VAL_FRACTION
from src.evaluation.class_rows import class_row_mask, leaking_rows, row_facts
from src.evaluation.entities import ROW_ID, Scenario, file_sha256
from src.evaluation.schemas import SplitManifest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from src.evaluation.entities import EntityResolver

logger = logging.getLogger(__name__)

Partition = Literal["train", "val", "test", "drop"]
PARTITIONS: tuple[Partition, ...] = ("train", "val", "test", "drop")
ASSIGNMENT_FILE = "assignment.parquet"
MANIFEST_FILE = "manifest.json"
RARE_STRATUM = "__rare__"
DEFAULT_STRATIFY_COL = "phenotype_category"


def fold_column(fold: int) -> str:
    return f"fold_{fold}"


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #


def _deal_entities(
    keys: Sequence[frozenset[str]], k: int, rng: np.random.Generator
) -> dict[str, int]:
    """Entity → fold, balancing row counts (largest entities first)."""
    counts = Counter(key for row in keys for key in row)
    entities = sorted(counts)
    order = rng.permutation(len(entities))
    shuffled = [entities[i] for i in order]
    shuffled.sort(key=lambda e: counts[e], reverse=True)  # stable: ties stay shuffled
    load = np.zeros(k, dtype=np.int64)
    fold_of: dict[str, int] = {}
    for entity in shuffled:
        f = int(np.argmin(load))
        fold_of[entity] = f
        load[f] += counts[entity]
    return fold_of


def _membership(row: frozenset[str], members: set[str]) -> tuple[bool, bool]:
    """(all keys are members, any key is a member)."""
    hits = [key in members for key in row]
    return all(hits), any(hits)


def _grouped(
    keys: Sequence[frozenset[str]], k: int, seed: int, val_fraction: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_of = _deal_entities(keys, k, rng)
    out = np.empty((len(keys), k), dtype="<U5")
    for f in range(k):
        in_fold = {e for e, fe in fold_of.items() if fe == f}
        candidates: list[int] = []
        for i, row in enumerate(keys):
            all_in, any_in = _membership(row, in_fold)
            if all_in:
                out[i, f] = "test"
            elif any_in:
                out[i, f] = "drop"
            else:
                candidates.append(i)
        # Val entities: shuffled non-test entities until ≈ val_fraction of candidate rows.
        cand_counts = Counter(key for i in candidates for key in keys[i])
        entities = sorted(cand_counts)
        fold_rng = np.random.default_rng([seed, f])
        target = val_fraction * len(candidates)
        val: set[str] = set()
        covered = 0
        for j in fold_rng.permutation(len(entities)):
            if covered >= target:
                break
            val.add(entities[j])
            covered += cand_counts[entities[j]]
        for i in candidates:
            all_in, any_in = _membership(keys[i], val)
            out[i, f] = "val" if all_in else ("drop" if any_in else "train")
    return out


def _strata(labels: Sequence[str], min_count: int) -> np.ndarray:
    counts = Counter(labels)
    return np.array(
        [lab if counts[lab] >= min_count else RARE_STRATUM for lab in labels]
    )


def _stratified(
    n: int, labels: Sequence[str], k: int, seed: int, val_fraction: float
) -> np.ndarray:
    strata = _strata(labels, k)
    if Counter(strata).most_common()[-1][1] < k:
        splitter = KFold(k, shuffle=True, random_state=seed).split(np.zeros(n))
    else:
        splitter = StratifiedKFold(k, shuffle=True, random_state=seed).split(
            np.zeros(n), strata
        )
    out = np.empty((n, k), dtype="<U5")
    for f, (rest, test) in enumerate(splitter):
        out[test, f] = "test"
        rest_strata = _strata([labels[i] for i in rest], 2)
        try:
            train, val = train_test_split(
                rest,
                test_size=val_fraction,
                stratify=rest_strata,
                random_state=seed + f,
            )
        except ValueError:
            train, val = train_test_split(
                rest, test_size=val_fraction, random_state=seed + f
            )
        out[train, f] = "train"
        out[val, f] = "val"
    return out


def assign_folds(
    keys: list[frozenset[str]],
    k: int,
    seed: int,
    val_fraction: float,
    *,
    stratify: list[str] | None = None,
) -> np.ndarray:
    """``(n_rows, k)`` array of partitions; row-level stratified when ``stratify``."""
    if k < 2:
        raise EvaluationError(f"k must be >= 2, got {k}")
    if stratify is not None:
        return _stratified(len(keys), stratify, k, seed, val_fraction)
    return _grouped(keys, k, seed, val_fraction)


# --------------------------------------------------------------------------- #
# Split + manifest
# --------------------------------------------------------------------------- #


def make_split(
    df: pl.DataFrame,
    scenario: Scenario,
    resolver: EntityResolver,
    *,
    k: int = DEFAULT_FOLDS,
    seed: int | None = None,
    data_sha256: str,
    catalog_sha256: str,
    stratify_col: str | None = DEFAULT_STRATIFY_COL,
    val_fraction: float = VAL_FRACTION,
    members: Mapping[str, frozenset[str]] | None = None,
) -> tuple[pl.DataFrame, SplitManifest]:
    """Assign every row of ``df`` (with ``row_id``) to a partition in each fold.

    Class rows are left out of the deal and are ``train`` in every fold, or
    ``drop`` where they leak; ``members`` (ATC code -> ChEMBL IDs) gives the
    classes' member drugs for the leakage rule.
    """
    seed = get_settings().seed if seed is None else seed
    is_class = class_row_mask(df).to_numpy()
    dealt = np.flatnonzero(~is_class)
    sub = df[dealt.tolist()] if is_class.any() else df
    keys = resolver.keys(sub, scenario)
    stratify = None
    if scenario is Scenario.RANDOM and stratify_col and stratify_col in df.columns:
        stratify = [
            str(v) if v is not None else "" for v in sub[stratify_col].to_list()
        ]
    parts = np.full((len(df), k), "train", dtype="<U5")
    parts[dealt] = assign_folds(keys, k, seed, val_fraction, stratify=stratify)
    dropped = _drop_leaking_class_rows(df, parts, resolver, members or {}, scenario)

    assignment = pl.DataFrame(
        {ROW_ID: df[ROW_ID], **{fold_column(f): parts[:, f].tolist() for f in range(k)}}
    )
    rows_per_fold = [
        {p: int((parts[:, f] == p).sum()) for p in PARTITIONS} for f in range(k)
    ]
    never_tested = int(((parts == "test").sum(axis=1) == 0).sum())
    drug_col = resolver.drug_col
    no_drug = (
        int(
            df.select(
                (
                    pl.col(drug_col).is_null()
                    | (pl.col(drug_col).cast(pl.String).str.strip_chars() == "")
                ).sum()
            ).item()
        )
        if drug_col in df.columns
        else len(df)
    )
    manifest = SplitManifest(
        scenario=scenario,
        k=k,
        seed=seed,
        val_fraction=val_fraction,
        data_sha256=data_sha256,
        catalog_sha256=catalog_sha256,
        drug_col=drug_col,
        n_rows=len(df),
        rows_per_fold=rows_per_fold,
        rows_never_tested=never_tested,
        rows_without_drug_id=no_drug,
        created_at=datetime.now(UTC),
        class_rows=int(is_class.sum()),
        class_rows_dropped_per_fold=dropped,
    )
    logger.info(
        "Split %s: k=%d, %d rows, %d never tested, %d without drug id; %d class "
        "rows (train-only), dropped per fold for leakage: %s",
        scenario,
        k,
        len(df),
        never_tested,
        no_drug,
        int(is_class.sum()),
        dropped,
    )
    return assignment, manifest


def _drop_leaking_class_rows(
    df: pl.DataFrame,
    parts: np.ndarray,
    resolver: EntityResolver,
    members: Mapping[str, frozenset[str]],
    scenario: Scenario,
) -> list[int]:
    """Mark leaking class rows ``drop`` in each fold (in place); counts per fold."""
    k = parts.shape[1]
    class_idx = np.flatnonzero(class_row_mask(df).to_numpy())
    if class_idx.size == 0:
        return [0] * k
    facts = row_facts(
        df, resolver, members, None if scenario is Scenario.RANDOM else scenario
    )
    train = [facts[i] for i in class_idx]
    dropped = []
    for f in range(k):
        held = np.flatnonzero(np.isin(parts[:, f], ["val", "test"]))
        leaks = np.asarray(leaking_rows(train, (facts[i] for i in held)), dtype=bool)
        parts[class_idx[leaks], f] = "drop"
        dropped.append(int(leaks.sum()))
    return dropped


def split_dir(scenario: Scenario) -> Path:
    return get_settings().paths.data / SPLITS_DIRNAME / str(scenario)


def save_split(assignment: pl.DataFrame, manifest: SplitManifest) -> Path:
    """Write ``assignment.parquet`` + ``manifest.json``; returns the directory."""
    out = split_dir(manifest.scenario)
    out.mkdir(parents=True, exist_ok=True)
    assignment.write_parquet(out / ASSIGNMENT_FILE)
    stamped = manifest.model_copy(
        update={"split_sha256": file_sha256(out / ASSIGNMENT_FILE)}
    )
    (out / MANIFEST_FILE).write_text(stamped.model_dump_json(indent=2))
    logger.info("Saved split %s to %s", manifest.scenario, out)
    return out


def load_split(scenario: Scenario) -> tuple[pl.DataFrame, SplitManifest]:
    directory = split_dir(scenario)
    if not (directory / MANIFEST_FILE).exists():
        msg = (
            f"No split for scenario {scenario!s} under {directory}; "
            "run 'python main.py split' first"
        )
        raise EvaluationError(msg)
    manifest = SplitManifest.model_validate_json(
        (directory / MANIFEST_FILE).read_text()
    )
    assignment = pl.read_parquet(directory / ASSIGNMENT_FILE)
    found = file_sha256(directory / ASSIGNMENT_FILE)
    if found != manifest.split_sha256:
        msg = f"split {scenario!s}: assignment file does not match its manifest"
        raise EvaluationError(msg)
    return assignment, manifest


def verify_split(
    manifest: SplitManifest, *, data_sha256: str, catalog_sha256: str
) -> None:
    """Fail fast if the data or the HGVS catalog changed since the split."""
    for what, expected, found in (
        ("data file", manifest.data_sha256, data_sha256),
        ("HGVS catalog", manifest.catalog_sha256, catalog_sha256),
    ):
        if expected != found:
            msg = (
                f"The {what} changed since the {manifest.scenario!s} split was made "
                f"(expected sha256 {expected}, found {found}); "
                "re-run 'python main.py split'"
            )
            raise EvaluationError(msg)


def fold_frames(
    df: pl.DataFrame, assignment: pl.DataFrame, fold: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) rows of ``fold``, in ``df``'s order."""
    col = fold_column(fold)
    if col not in assignment.columns:
        raise EvaluationError(
            f"fold {fold} not in split (k={len(assignment.columns) - 1})"
        )
    joined = df.join(
        assignment.select(ROW_ID, col), on=ROW_ID, how="left", maintain_order="left"
    )
    missing = joined[col].null_count()
    if missing:
        raise EvaluationError(f"{missing} rows are not in the split assignment")
    return tuple(  # type: ignore[return-value]
        joined.filter(pl.col(col) == part).drop(col)
        for part in ("train", "val", "test")
    )


__all__ = [
    "PARTITIONS",
    "Partition",
    "assign_folds",
    "fold_column",
    "fold_frames",
    "load_split",
    "make_split",
    "save_split",
    "split_dir",
    "verify_split",
]
