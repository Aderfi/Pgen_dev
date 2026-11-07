"""Pydantic models of the evaluation protocol's files.

``manifest.json`` (one per split scenario), ``metrics.json`` (one per
evaluation) and ``comparison.json`` are the contract with whatever renders
them; :mod:`src.evaluation.report` is only one such renderer.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003  (pydantic needs it at runtime)
from typing import Literal

from pydantic import BaseModel, Field

from src.evaluation.diagnostics import OverlapReport  # noqa: TC001
from src.evaluation.entities import Scenario  # noqa: TC001

PartitionName = Literal["train", "val", "test", "drop"]


class SplitManifest(BaseModel):
    """How a scenario's folds were made, and what they are bound to."""

    scenario: Scenario
    k: int = Field(..., ge=2)
    seed: int
    val_fraction: float = Field(..., gt=0, lt=1)
    data_sha256: str
    catalog_sha256: str
    drug_col: str
    n_rows: int
    rows_per_fold: list[dict[PartitionName, int]]
    rows_never_tested: int
    rows_without_drug_id: int
    created_at: datetime
    split_sha256: str = ""
    #: Class rows (``is_class_row``): train-only in every fold.
    class_rows: int = 0
    #: Class rows dropped per fold because they would leak (src.evaluation.class_rows).
    class_rows_dropped_per_fold: list[int] = Field(default_factory=list)


class MetricCI(BaseModel):
    """A metric's point value and its bootstrap confidence interval."""

    value: float | None
    low: float | None = None
    high: float | None = None


class ClassMetrics(BaseModel):
    """Every metric of one label, on the out-of-fold test predictions."""

    label: str
    support: int
    prevalence: float
    tp: int
    fp: int
    fn: int
    tn: int
    threshold: float
    reliable: bool
    precision: MetricCI
    recall: MetricCI
    specificity: MetricCI
    f1: MetricCI
    f2: MetricCI
    f05: MetricCI
    roc_auc: MetricCI
    pr_auc: MetricCI
    mcc: MetricCI
    brier: MetricCI
    kappa: MetricCI
    pearson_r: MetricCI
    pearson_p: MetricCI
    spearman_rho: MetricCI
    spearman_p: MetricCI


class Spread(BaseModel):
    """Mean and standard deviation of a metric across folds."""

    mean: float | None
    sd: float | None
    n: int


class PairedResult(BaseModel):
    """Model minus baseline: difference with its CI and bootstrap p-value."""

    diff: MetricCI
    p_value: float | None


class FoldResult(BaseModel):
    fold: int
    n_test: int
    aggregate: dict[str, float | None]


class BaselineResult(BaseModel):
    name: str
    aggregate: dict[str, MetricCI]
    vs_model: dict[str, PairedResult]


class StratumResult(BaseModel):
    """Headline F1 macro on the test rows of one stratum (e.g. seen_drug=False)."""

    flag: str
    value: str
    n: int
    f1_macro: MetricCI


class EvaluationReport(BaseModel):
    """Everything ``metrics.json`` holds for one model x scenario evaluation."""

    model: str
    scenario: Scenario
    k: int
    folds_evaluated: list[int]
    data_sha256: str
    split_sha256: str
    seed: int
    n_bootstrap: int
    created_at: datetime
    n_test: int
    rows_never_tested: int
    rows_dropped: int
    labels: list[str]
    headline_labels: list[str]
    aggregate: dict[str, MetricCI]
    aggregate_at_05: dict[str, MetricCI]
    aggregate_all_classes: dict[str, MetricCI]
    between_folds: dict[str, Spread]
    per_fold: list[FoldResult]
    per_class: list[ClassMetrics]
    calibration: dict[str, float]
    baselines: list[BaselineResult]
    novelty: list[StratumResult]
    tanimoto: list[StratumResult]
    overlap: list[OverlapReport]


__all__ = [
    "BaselineResult",
    "ClassMetrics",
    "EvaluationReport",
    "FoldResult",
    "MetricCI",
    "PairedResult",
    "PartitionName",
    "SplitManifest",
    "Spread",
    "StratumResult",
]
