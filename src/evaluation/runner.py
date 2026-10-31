"""Evaluate a model on a split scenario and write the evaluation data files.

Per evaluated fold: load that fold's run, predict its val and test rows, fit
the temperature and the per-class thresholds on val (never on test), and
compute baselines from the fold's train. The test predictions of all folds
are concatenated (out-of-fold, every row once) and every reported metric,
interval and p-value is computed on them.

Output (``reports/eval/{model}_{scenario}_{timestamp}/``): ``metrics.json``
(:class:`EvaluationReport`), ``predictions.parquet`` / ``.csv`` and
``curves.json``. :mod:`src.evaluation.report` renders them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score

from src.config import get_model_config, get_settings
from src.core import EvaluationError
from src.data.cache import GraphCache
from src.data.encoders import UNKNOWN_CATEGORY_LABEL
from src.data.genotype import load_hgvs_catalog
from src.data.graph_indexing import GraphIndexBuilder
from src.evaluation import DEFAULT_BOOTSTRAP, EVAL_DIRNAME, MIN_POSITIVES
from src.evaluation.baselines import (
    BASELINES,
    key_frequency_baseline,
    prevalence_baseline,
)
from src.evaluation.bootstrap import bootstrap_ci, paired_test, resample_indices
from src.evaluation.diagnostics import (
    NOVELTY_FLAGS,
    drug_fingerprints,
    max_tanimoto,
    novelty_flags,
    overlap_report,
    tanimoto_bin,
)
from src.evaluation.entities import ROW_ID, EntityResolver, Scenario
from src.evaluation.metrics import (
    CLASS_METRIC_NAMES,
    aggregate_metrics,
    calibration_error,
    class_metrics,
    curves,
)
from src.evaluation.runs import FoldRef, load_fold_frames, run_name
from src.evaluation.schemas import (
    BaselineResult,
    ClassMetrics,
    EvaluationReport,
    FoldResult,
    MetricCI,
    PairedResult,
    Spread,
    StratumResult,
)
from src.evaluation.splits import load_split
from src.evaluation.thresholds import fit_temperature, fit_thresholds
from src.model.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from pathlib import Path

    from src.evaluation.diagnostics import OverlapReport
    from src.evaluation.schemas import SplitManifest
    from src.model.engine.predictor import PGenPredictor

logger = logging.getLogger(__name__)

#: Labels that are placeholders, not phenotypes: never evaluated.
EXCLUDED_LABELS = frozenset({UNKNOWN_CATEGORY_LABEL, "__UNK__", "Unknown", ""})
PAIRED_METRICS = ("f1_macro", "f2_macro", "pr_auc_macro")
CONFIDENCE_BANDS = ((0.66, "high"), (0.33, "medium"), (0.0, "low"))
DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Model outputs → (P, Y) matrices
# --------------------------------------------------------------------------- #


def _labels(predictor: PGenPredictor) -> list[str]:
    cols = predictor.target_cols
    out = []
    for col in cols:
        classes = [str(c) for c in predictor.encoders[col].classes_]
        out.extend(classes if len(cols) == 1 else [f"{col}={c}" for c in classes])
    return out


def _probabilities(
    predictor: PGenPredictor, logits: dict[str, torch.Tensor]
) -> np.ndarray:
    parts = []
    for col in predictor.target_cols:
        t = logits[col].float()
        multi = col in predictor.multi_label_cols
        parts.append((torch.sigmoid(t) if multi else torch.softmax(t, dim=1)).numpy())
    return np.concatenate(parts, axis=1)


def _targets(predictor: PGenPredictor, encoded: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for col in predictor.target_cols:
        y = encoded[col]
        if col in predictor.multi_label_cols:
            parts.append(y.astype(int))
        else:
            n_classes = len(predictor.encoders[col].classes_)
            parts.append(np.eye(n_classes, dtype=int)[y.astype(int)])
    return np.concatenate(parts, axis=1)


def _torch_labels(predictor: PGenPredictor, encoded: dict[str, np.ndarray]):
    return {col: torch.as_tensor(encoded[col]) for col in predictor.target_cols}


# --------------------------------------------------------------------------- #
# Per-fold collection
# --------------------------------------------------------------------------- #


@dataclass
class _Collected:
    """Out-of-fold arrays, concatenated over folds."""

    frames: list[pl.DataFrame] = field(default_factory=list)
    Y: list[np.ndarray] = field(default_factory=list)
    P: list[np.ndarray] = field(default_factory=list)
    P_raw: list[np.ndarray] = field(default_factory=list)
    thr: list[np.ndarray] = field(default_factory=list)
    base_P: dict[str, list[np.ndarray]] = field(default_factory=dict)
    base_thr: dict[str, list[np.ndarray]] = field(default_factory=dict)
    folds: list[FoldResult] = field(default_factory=list)
    overlap: list[OverlapReport] = field(default_factory=list)
    flags: list[pl.DataFrame] = field(default_factory=list)
    tanimoto: list[float | None] = field(default_factory=list)

    def cat(self, items: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(items, axis=0)


def _baseline_probs(
    name: str,
    resolver: EntityResolver,
    train: pl.DataFrame,
    Y_train: np.ndarray,
    rows: pl.DataFrame,
) -> np.ndarray:
    if name == "prevalence":
        return prevalence_baseline(Y_train, len(rows))
    fn = resolver.drugs if name == "drug_only" else resolver.genes
    train_keys = [fn(r) for r in train.iter_rows(named=True)]
    keys = [fn(r) for r in rows.iter_rows(named=True)]
    return key_frequency_baseline(train_keys, Y_train, keys)


def _available_folds(model_name: str, scenario: Scenario, k: int) -> list[int]:
    out = []
    for fold in range(k):
        name = run_name(model_name, FoldRef(scenario=scenario, fold=fold))
        if CheckpointManager(name).get_best_checkpoint_path() is not None:
            out.append(fold)
    return out


def _fingerprint_cache() -> GraphCache:
    index = GraphIndexBuilder.build_drug_index(get_settings().paths.library / "drugs")
    return GraphCache(index)


# --------------------------------------------------------------------------- #
# Metric assembly
# --------------------------------------------------------------------------- #


def _samples(fn, indices: list[np.ndarray]) -> list[dict[str, float | None]]:
    return [fn(i) for i in indices]


def _ci_table(
    point: dict[str, float | None], samples: list[dict[str, float | None]]
) -> dict[str, MetricCI]:
    picks = list(range(len(samples)))
    return {
        key: bootstrap_ci(lambda b, key=key: samples[b].get(key), picks, point=value)
        for key, value in point.items()
    }


def _paired(
    model_point, model_samples, base_point, base_samples
) -> dict[str, PairedResult]:
    picks = list(range(len(model_samples)))
    out = {}
    for key in PAIRED_METRICS:
        diff, p = paired_test(
            lambda b, key=key: model_samples[b].get(key),
            lambda b, key=key: base_samples[b].get(key),
            picks,
            point_a=model_point.get(key),
            point_b=base_point.get(key),
        )
        out[key] = PairedResult(diff=diff, p_value=p)
    return out


def _f1_macro(Y: np.ndarray, P: np.ndarray, thr: np.ndarray) -> float | None:
    if not len(Y) or Y.shape[1] == 0:
        return None
    return float(f1_score(Y, (P > thr).astype(int), average="macro", zero_division=0))


def _strata(
    flag: str,
    values: list[str],
    Y: np.ndarray,
    P: np.ndarray,
    thr: np.ndarray,
    groups: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> list[StratumResult]:
    out = []
    arr = np.array(values)
    for value in sorted(set(values)):
        mask = arr == value
        y, p, t = Y[mask], P[mask], thr[mask]
        idx = resample_indices(
            int(mask.sum()), None if groups is None else groups[mask], n_boot, seed
        )
        point = _f1_macro(y, p, t)
        ci = bootstrap_ci(
            lambda i, y=y, p=p, t=t: _f1_macro(y[i], p[i], t[i]), idx, point=point
        )
        out.append(
            StratumResult(flag=flag, value=value, n=int(mask.sum()), f1_macro=ci)
        )
    return out


def _class_table(
    labels: list[str],
    evaluated: np.ndarray,
    headline: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    thr: np.ndarray,
    indices: list[np.ndarray],
) -> list[ClassMetrics]:
    out = []
    for c, label in enumerate(labels):
        if not evaluated[c]:
            continue
        y, p, t = Y[:, c], P[:, c], thr[:, c]
        point = class_metrics(y, p, t)
        samples = [class_metrics(y[i], p[i], t[i]) for i in indices]
        cis = _ci_table({k: point[k] for k in CLASS_METRIC_NAMES}, samples)
        for key in ("pearson_p", "spearman_p"):
            cis[key] = MetricCI(value=point[key])
        out.append(
            ClassMetrics(
                label=label,
                support=int(y.sum()),
                prevalence=float(y.mean()) if len(y) else 0.0,
                tp=int(point["tp"]),
                fp=int(point["fp"]),
                fn=int(point["fn"]),
                tn=int(point["tn"]),
                threshold=float(t.mean()),
                reliable=int(y.sum()) >= MIN_POSITIVES,
                **cis,
            )
        )
    return out


def _confidence(P: np.ndarray, thr: np.ndarray, evaluated: np.ndarray) -> np.ndarray:
    margin = np.abs(P - thr) / np.maximum(thr, 1 - thr)
    return margin[:, evaluated].mean(axis=1) if evaluated.any() else np.zeros(len(P))


def _band(value: float) -> str:
    return next(name for lo, name in CONFIDENCE_BANDS if value >= lo)


def _prediction_table(
    df: pl.DataFrame,
    labels: list[str],
    evaluated: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    thr: np.ndarray,
    flags: pl.DataFrame,
    tanimoto: list[float | None],
) -> pl.DataFrame:
    pred = (P > thr).astype(int)
    names = np.array(labels)
    conf = _confidence(P, thr, evaluated)

    def joined(mask_row: np.ndarray) -> str:
        return "|".join(names[mask_row & evaluated])

    keep = [
        c
        for c in ("drugs", "drugs_cid", "gene", "genotype", "alleles")
        if c in df.columns
    ]
    table = df.select(ROW_ID, "fold", *keep).with_columns(
        pl.Series("true_labels", [joined(r == 1) for r in Y]),
        pl.Series("pred_labels", [joined(r == 1) for r in pred]),
        pl.Series("confidence", conf),
        pl.Series("confidence_band", [_band(v) for v in conf]),
        pl.Series("exact_match", (pred[:, evaluated] == Y[:, evaluated]).all(axis=1)),
        pl.Series(
            "fp_labels",
            [joined((p == 1) & (y == 0)) for p, y in zip(pred, Y, strict=True)],
        ),
        pl.Series(
            "fn_labels",
            [joined((p == 0) & (y == 1)) for p, y in zip(pred, Y, strict=True)],
        ),
        pl.Series("max_tanimoto", tanimoto, dtype=pl.Float64),
        *[
            pl.Series(f"p_{label}", P[:, c])
            for c, label in enumerate(labels)
            if evaluated[c]
        ],
    )
    return table.join(flags, on=ROW_ID, how="left", maintain_order="left")


def _output_dir(out_root: Path | None, model_name: str, scenario: Scenario) -> Path:
    root = out_root or get_settings().paths.reports / EVAL_DIRNAME
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = root / f"{model_name}_{scenario}_{stamp}"
    suffix = 1
    while out.exists():
        out = root / f"{model_name}_{scenario}_{stamp}-{suffix}"
        suffix += 1
    out.mkdir(parents=True)
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _resolve_folds(
    model_name: str, scenario: Scenario, k: int, folds: list[int] | None
) -> list[int]:
    available = _available_folds(model_name, scenario, k)
    if folds is not None:
        folds = sorted(set(folds))
        out_of_range = [f for f in folds if not 0 <= f < k]
        if out_of_range:
            raise EvaluationError(
                f"fold(s) {out_of_range} out of range for {scenario!s} (k={k})"
            )
    if folds is None:
        if not available:
            raise EvaluationError(
                f"no trained fold for {scenario!s}: train with "
                f"'python main.py train --scenario {scenario} --fold K'"
            )
        return available
    missing = [f for f in folds if f not in available]
    if missing:
        names = [
            run_name(model_name, FoldRef(scenario=scenario, fold=f)) for f in missing
        ]
        raise EvaluationError(f"no checkpoint for fold run(s): {', '.join(names)}")
    return folds


def _check_labels(previous: list[str], current: list[str], fold: int) -> None:
    """Every fold's encoder must expose the same classes in the same order."""
    if previous and previous != current:
        raise EvaluationError(
            f"fold {fold} was trained with classes {current}, other folds with "
            f"{previous}; out-of-fold predictions cannot be aligned (a class is "
            "missing from some fold's train)"
        )


def _check_run_split(
    predictor: PGenPredictor, manifest: SplitManifest, ref: FoldRef, name: str
) -> None:
    """The run must have been trained on exactly this split and fold."""
    info = predictor.split_info or {}
    expected = (str(ref.scenario), ref.fold, manifest.split_sha256)
    found = (info.get("scenario"), info.get("fold"), info.get("split_sha256"))
    if found != expected:
        raise EvaluationError(
            f"run {name} was not trained on the current {ref.scenario!s} split "
            f"(trained on {found}, current {expected}); retrain it with "
            f"'python main.py train --scenario {ref.scenario} --fold {ref.fold}'"
        )


def _fold_results(
    folds: list[int],
    Y: list[np.ndarray],
    P: list[np.ndarray],
    thr: list[np.ndarray],
    headline: np.ndarray,
) -> list[FoldResult]:
    """Per-fold aggregates over the same headline classes as the OOF metrics."""
    return [
        FoldResult(
            fold=fold,
            n_test=len(y),
            aggregate=aggregate_metrics(y, p, t, include=headline),
        )
        for fold, y, p, t in zip(folds, Y, P, thr, strict=True)
    ]


@dataclass
class _Context:
    model_name: str
    source: Path
    scenario: Scenario
    resolver: EntityResolver
    cache: GraphCache
    device: str | None


def _collect_fold(ctx: _Context, fold: int, col: _Collected) -> list[str]:
    """Predict one fold and append its out-of-fold rows to ``col``; returns labels."""
    from src.model.engine.predictor import PGenPredictor

    cfg = get_model_config(ctx.model_name)
    ref = FoldRef(scenario=ctx.scenario, fold=fold)
    train, val, test, manifest = load_fold_frames(ctx.source, cfg, ref)
    if not len(test) or not len(val):
        raise EvaluationError(
            f"fold {fold} of {ctx.scenario!s} has an empty val or test"
        )
    name = run_name(ctx.model_name, ref)
    predictor = PGenPredictor(ctx.model_name, device=ctx.device, run_name=name)
    _check_run_split(predictor, manifest, ref, name)
    labels = _labels(predictor)

    enc_val, enc_test = predictor.encode_targets(val), predictor.encode_targets(test)
    Y_train = _targets(predictor, predictor.encode_targets(train))
    Y_val, Y_test = _targets(predictor, enc_val), _targets(predictor, enc_test)
    logits_val, logits_test = (
        predictor.predict_logits(val),
        predictor.predict_logits(test),
    )
    scaler = fit_temperature(
        logits_val, _torch_labels(predictor, enc_val), predictor._resolve_axes()
    )
    with torch.no_grad():
        P_val = _probabilities(predictor, scaler(logits_val))
        P_test = _probabilities(predictor, scaler(logits_test))
    thr = fit_thresholds(Y_val, P_val)

    col.frames.append(test.with_columns(pl.lit(fold).alias("fold")))
    col.Y.append(Y_test)
    col.P.append(P_test)
    col.P_raw.append(_probabilities(predictor, logits_test))
    col.thr.append(np.tile(thr, (len(test), 1)))
    for name in BASELINES:
        b_val = _baseline_probs(name, ctx.resolver, train, Y_train, val)
        col.base_P[name].append(
            _baseline_probs(name, ctx.resolver, train, Y_train, test)
        )
        col.base_thr[name].append(np.tile(fit_thresholds(Y_val, b_val), (len(test), 1)))

    col.flags.append(novelty_flags(ctx.resolver, train, test))
    train_drugs = {
        d for r in train.iter_rows(named=True) for d in ctx.resolver.drugs(r)
    }
    test_drugs = [ctx.resolver.drugs(r) for r in test.iter_rows(named=True)]
    fps = drug_fingerprints(train_drugs | {d for s in test_drugs for d in s}, ctx.cache)
    col.tanimoto.extend(max_tanimoto(test_drugs, train_drugs, fps))
    col.overlap.append(overlap_report(ctx.resolver, train, test, ctx.scenario))
    logger.info("Evaluated %s fold %d (test=%d)", ctx.scenario, fold, len(test))
    return labels


def _between_folds(folds: list[FoldResult]) -> dict[str, Spread]:
    out = {}
    for key in sorted({k for f in folds for k in f.aggregate}):
        vals = [v for f in folds if (v := f.aggregate.get(key)) is not None]
        out[key] = Spread(
            mean=float(np.mean(vals)) if vals else None,
            sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            n=len(vals),
        )
    return out


def _calibration(Y: np.ndarray, P_raw: np.ndarray, P: np.ndarray) -> dict[str, float]:
    ece_b, mce_b = calibration_error(Y, P_raw)
    ece_a, mce_a = calibration_error(Y, P)
    return {
        "ece_before": ece_b,
        "ece_after": ece_a,
        "mce_before": mce_b,
        "mce_after": mce_a,
        "brier_before": float(np.mean((P_raw - Y) ** 2)),
        "brier_after": float(np.mean((P - Y) ** 2)),
    }


def evaluate_scenario(
    model_name: str,
    source: Path,
    scenario: Scenario,
    *,
    folds: list[int] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    out_root: Path | None = None,
    device: str | None = None,
) -> Path:
    """Evaluate ``model_name`` on ``scenario``'s folds; returns the output dir."""
    cfg = get_model_config(model_name)
    _, manifest = load_split(scenario)
    folds = _resolve_folds(model_name, scenario, manifest.k, folds)
    seed = get_settings().seed
    resolver = EntityResolver(load_hgvs_catalog(), drug_col=cfg.features[0])
    ctx = _Context(model_name, source, scenario, resolver, _fingerprint_cache(), device)
    col = _Collected(
        base_P={b: [] for b in BASELINES}, base_thr={b: [] for b in BASELINES}
    )
    labels: list[str] = []
    for fold in folds:
        fold_labels = _collect_fold(ctx, fold, col)
        _check_labels(labels, fold_labels, fold)
        labels = fold_labels

    df = pl.concat(col.frames, how="diagonal_relaxed")
    Y, P, P_raw, thr = (
        col.cat(col.Y),
        col.cat(col.P),
        col.cat(col.P_raw),
        col.cat(col.thr),
    )
    evaluated = np.array([lab not in EXCLUDED_LABELS for lab in labels])
    headline = evaluated & (Y.sum(axis=0) >= MIN_POSITIVES)
    if not headline.any():
        logger.warning(
            "No class has >= %d positives; headline uses all classes", MIN_POSITIVES
        )
        headline = evaluated.copy()
    col.folds = _fold_results(folds, col.Y, col.P, col.thr, headline)

    groups = None
    if scenario is not Scenario.RANDOM:
        groups = np.array(["|".join(sorted(k)) for k in resolver.keys(df, scenario)])
    indices = resample_indices(len(df), groups, n_boot, seed)
    everything = np.arange(len(df))

    def agg(Pm: np.ndarray, thr_m: np.ndarray, mask: np.ndarray):
        return lambda i: aggregate_metrics(Y[i], Pm[i], thr_m[i], include=mask)

    def table(fn) -> tuple[dict[str, float | None], list[dict[str, float | None]]]:
        return fn(everything), _samples(fn, indices)

    model_point, model_samples = table(agg(P, thr, headline))
    baselines = []
    for name in BASELINES:
        b_point, b_samples = table(
            agg(col.cat(col.base_P[name]), col.cat(col.base_thr[name]), headline)
        )
        baselines.append(
            BaselineResult(
                name=name,
                aggregate=_ci_table(b_point, b_samples),
                vs_model=_paired(model_point, model_samples, b_point, b_samples),
            )
        )

    flags = pl.concat(col.flags)
    flag_rows = df.select(ROW_ID).join(
        flags, on=ROW_ID, how="left", maintain_order="left"
    )
    Yh, Ph, thr_h = Y[:, headline], P[:, headline], thr[:, headline]
    novelty = [
        s
        for flag in NOVELTY_FLAGS
        for s in _strata(
            flag, [str(v) for v in flag_rows[flag]], Yh, Ph, thr_h, groups, n_boot, seed
        )
    ]
    tani_values = [tanimoto_bin(v) for v in col.tanimoto]

    report = EvaluationReport(
        model=model_name,
        scenario=scenario,
        k=manifest.k,
        folds_evaluated=folds,
        data_sha256=manifest.data_sha256,
        split_sha256=manifest.split_sha256,
        seed=seed,
        n_bootstrap=n_boot,
        created_at=datetime.now(UTC),
        n_test=len(df),
        rows_never_tested=manifest.rows_never_tested,
        rows_dropped=sum(manifest.rows_per_fold[f]["drop"] for f in folds),
        labels=[lab for lab, e in zip(labels, evaluated, strict=True) if e],
        headline_labels=[lab for lab, h in zip(labels, headline, strict=True) if h],
        aggregate=_ci_table(model_point, model_samples),
        aggregate_at_05=_ci_table(
            *table(agg(P, np.full_like(thr, DEFAULT_THRESHOLD), headline))
        ),
        aggregate_all_classes=_ci_table(*table(agg(P, thr, evaluated))),
        between_folds=_between_folds(col.folds),
        per_fold=col.folds,
        per_class=_class_table(labels, evaluated, headline, Y, P, thr, indices),
        calibration=_calibration(Y[:, evaluated], P_raw[:, evaluated], P[:, evaluated]),
        baselines=baselines,
        novelty=novelty,
        tanimoto=_strata(
            "max_tanimoto", tani_values, Yh, Ph, thr_h, groups, n_boot, seed
        ),
        overlap=col.overlap,
    )
    out = _output_dir(out_root, model_name, scenario)
    _write_outputs(out, report, df, labels, evaluated, Y, P, thr, flags, col.tanimoto)
    logger.info("Evaluation of %s on %s written to %s", model_name, scenario, out)
    return out


def _write_outputs(
    out: Path,
    report: EvaluationReport,
    df: pl.DataFrame,
    labels: list[str],
    evaluated: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    thr: np.ndarray,
    flags: pl.DataFrame,
    tanimoto: list[float | None],
) -> None:
    (out / "metrics.json").write_text(report.model_dump_json(indent=2))
    table = _prediction_table(df, labels, evaluated, Y, P, thr, flags, tanimoto)
    table.write_parquet(out / "predictions.parquet")
    table.write_csv(out / "predictions.csv")
    curve_data = {
        label: curves(Y[:, c], P[:, c])
        for c, label in enumerate(labels)
        if evaluated[c]
    }
    (out / "curves.json").write_text(json.dumps(curve_data))


__all__ = ["EXCLUDED_LABELS", "evaluate_scenario"]
