"""Inference evaluation protocol.

Grouped k-fold split scenarios that simulate never-seen cases, training on a
scenario fold, and an evaluation that reports per-row, per-class and
aggregate metrics with bootstrap confidence intervals, leakage / novelty
diagnostics and baselines (see ``docs/superpowers/specs`` for the design).

* :mod:`.entities`    — row identity and the entity keys each scenario groups on.
* :mod:`.splits`      — fold assignment, persistence and hash verification.
* :mod:`.runs`        — run names and fold frames for training on a fold.
* :mod:`.metrics`     — per-class / aggregate / calibration metrics.
* :mod:`.bootstrap`   — percentile CIs and paired bootstrap tests.
* :mod:`.thresholds`  — per-class thresholds and temperature fitting.
* :mod:`.baselines`   — prevalence / drug-only / gene-only baselines.
* :mod:`.diagnostics` — novelty flags, Tanimoto similarity, train/test overlap.
* :mod:`.runner`      — evaluates a scenario and writes the data files.
* :mod:`.report`      — renders ``report.html`` from those files.
* :mod:`.compare`     — scenario comparison report.
"""

SPLITS_DIRNAME = "splits"
EVAL_DIRNAME = "eval"
DEFAULT_FOLDS = 5
VAL_FRACTION = 0.15
MIN_POSITIVES = 10
DEFAULT_BOOTSTRAP = 2000

__all__ = [
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_FOLDS",
    "EVAL_DIRNAME",
    "MIN_POSITIVES",
    "SPLITS_DIRNAME",
    "VAL_FRACTION",
]
