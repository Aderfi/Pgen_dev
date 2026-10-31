"""Static, self-contained HTML report rendered from an evaluation's data files.

Reads only ``metrics.json``, ``predictions.parquet`` and ``curves.json`` — the
report can be regenerated (or replaced by another frontend) without
recomputing anything. Plots are matplotlib PNGs inlined as base64; the
per-row table sorts and filters with a few lines of inline JavaScript.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from html import escape
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from src.evaluation.schemas import EvaluationReport, MetricCI  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)

REPORT_FILE = "report.html"
_HEADLINE = (
    "f1_micro f1_macro f1_weighted f1_samples f2_micro f2_macro f2_weighted f2_samples "
    "f05_micro f05_macro precision_micro precision_macro recall_micro recall_macro "
    "roc_auc_micro roc_auc_macro pr_auc_micro pr_auc_macro subset_accuracy hamming_loss "
    "jaccard_samples jaccard_macro kappa_mean brier_mean ece mce"
).split()
_CLASS_COLS = (
    "precision recall specificity f1 f2 f05 roc_auc pr_auc mcc kappa brier "
    "pearson_r pearson_p spearman_rho spearman_p"
).split()

CSS = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#222;max-width:1400px}
h1{margin-bottom:.2rem} h2{margin-top:2.2rem;border-bottom:1px solid #ccc}
table{border-collapse:collapse;font-size:13px;margin:.6rem 0}
th,td{border:1px solid #ddd;padding:3px 7px;text-align:right}
th{background:#f3f3f3;cursor:pointer} td.l,th.l{text-align:left}
.ci{color:#777;font-size:11px} .unreliable{color:#999} .warn{color:#b00;font-weight:600}
.meta{color:#555} img{max-width:48%;margin:4px} .note{color:#666;font-size:13px}
"""

JS = """
function sortTable(th){const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th);
const rows=[...t.tBodies[0].rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort((a,b)=>{const x=a.cells[i].dataset.v??a.cells[i].innerText,y=b.cells[i].dataset.v??b.cells[i].innerText;
const nx=parseFloat(x),ny=parseFloat(y);const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c});
rows.forEach(r=>t.tBodies[0].appendChild(r))}
function filterRows(){const f=document.getElementById('flt').value;
document.querySelectorAll('#preds tbody tr').forEach(r=>{const d=r.dataset;
r.style.display=(f==='all'||(f==='errors'&&d.exact==='false')||(f==='low'&&d.band==='low')||
(f==='novel'&&d.novel==='true'))?'':'none'})}
"""


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_ci(ci: MetricCI | None) -> str:
    if ci is None:
        return "—"
    if ci.low is None or ci.high is None:
        return fmt(ci.value)
    return f'{fmt(ci.value)} <span class="ci">[{fmt(ci.low)}–{fmt(ci.high)}]</span>'


def table(
    headers: Sequence[str], rows: Iterable[Sequence[str]], attrs: str = ""
) -> str:
    head = "".join(f'<th onclick="sortTable(this)">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table {attrs}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{escape(title)}</title>"
        f"<style>{CSS}</style><script>{JS}</script></head><body>{body}</body></html>"
    )


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return (
        f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
    )


def _curve_plot(
    curves: dict, x: str, y: str, title: str, xlabel: str, ylabel: str
) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, c in curves.items():
        if c[x]:
            ax.plot(c[x], c[y], label=label)
    if x == "roc_fpr":
        ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=7)
    return _png(fig)


def _reliability_plot(curves: dict) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    for label, c in curves.items():
        pts = [
            (m, f)
            for m, f, n in zip(
                c["rel_mean_pred"], c["rel_frac_pos"], c["rel_count"], strict=True
            )
            if n
        ]
        if pts:
            ax.plot(*zip(*pts, strict=True), marker="o", label=label)
    ax.set(
        title="Reliability (calibrated)",
        xlabel="mean predicted",
        ylabel="fraction positive",
    )
    ax.legend(fontsize=7)
    return _png(fig)


def _confusion_plot(report: EvaluationReport) -> str:
    n = len(report.per_class)
    if not n:
        return ""
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.4), squeeze=False)
    for ax, c in zip(axes[0], report.per_class, strict=True):
        m = [[c.tn, c.fp], [c.fn, c.tp]]
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i][j]), ha="center", va="center", fontsize=8)
        ax.set(
            title=c.label,
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )
        ax.title.set_fontsize(8)
    return _png(fig)


def _header(r: EvaluationReport) -> str:
    return (
        f"<h1>Evaluation — {escape(r.model)} · {escape(str(r.scenario))}</h1>"
        f"<h2>1 · Header</h2><p class='meta'>folds {r.folds_evaluated} of k={r.k} · "
        f"test rows (out-of-fold) {r.n_test} · never tested {r.rows_never_tested} · "
        f"dropped (strict cold, summed over folds) {r.rows_dropped} · seed {r.seed} · "
        f"bootstrap {r.n_bootstrap} resamples, 95% CI · data sha {r.data_sha256[:12]} · "
        f"split sha {r.split_sha256[:12]} · {r.created_at:%Y-%m-%d %H:%M} UTC</p>"
        f"<p class='meta'>Headline classes (≥10 positives): "
        f"{escape(', '.join(r.headline_labels))}</p>"
    )


def _aggregate(r: EvaluationReport) -> str:
    rows = [
        [
            f"<span class='l'>{k}</span>",
            fmt_ci(r.aggregate.get(k)),
            fmt_ci(r.aggregate_at_05.get(k)),
            fmt_ci(r.aggregate_all_classes.get(k)),
            fmt(r.between_folds[k].mean) + " ± " + fmt(r.between_folds[k].sd)
            if k in r.between_folds
            else "—",
        ]
        for k in _HEADLINE
        if k in r.aggregate
    ]
    cal = r.calibration
    return (
        "<h2>3 · Aggregate metrics</h2>"
        + table(
            [
                "metric",
                "headline (val thresholds)",
                "headline @0.5",
                "all classes",
                "between folds (mean ± sd)",
            ],
            rows,
        )
        + f"<p class='note'>Calibration (temperature fitted on val): ECE {fmt(cal['ece_before'])} → "
        f"{fmt(cal['ece_after'])}, MCE {fmt(cal['mce_before'])} → {fmt(cal['mce_after'])}, "
        f"Brier {fmt(cal['brier_before'])} → {fmt(cal['brier_after'])}</p>"
    )


def _per_class(r: EvaluationReport) -> str:
    rows = []
    for c in r.per_class:
        cells = [
            f"<span class='l'>{escape(c.label)}</span>"
            + (
                ""
                if c.reliable
                else " <span class='unreliable'>(not reliable: &lt;10 positives)</span>"
            ),
            str(c.support),
            fmt(c.prevalence),
            fmt(c.threshold),
            f"{c.tp}/{c.fp}/{c.fn}/{c.tn}",
            *(fmt_ci(getattr(c, k)) for k in _CLASS_COLS),
        ]
        rows.append(cells)
    return (
        "<h2>4 · Per class</h2><p class='note'>pearson_r is the point-biserial correlation "
        "between the true label and the predicted probability; MCC is the phi coefficient "
        "of the thresholded predictions.</p>"
        + table(
            [
                "class",
                "support",
                "prevalence",
                "threshold",
                "TP/FP/FN/TN",
                *_CLASS_COLS,
            ],
            rows,
        )
    )


def _plots(r: EvaluationReport, curves: dict) -> str:
    return (
        "<h2>5 · Plots</h2>"
        + _curve_plot(
            curves,
            "pr_recall",
            "pr_precision",
            "Precision–recall",
            "recall",
            "precision",
        )
        + _curve_plot(
            curves,
            "roc_fpr",
            "roc_tpr",
            "ROC",
            "false positive rate",
            "true positive rate",
        )
        + _reliability_plot(curves)
        + _confusion_plot(r)
    )


def _strata_table(strata) -> str:
    return table(
        ["flag", "value", "n", "F1 macro (headline)"],
        (
            [escape(s.flag), escape(s.value), str(s.n), fmt_ci(s.f1_macro)]
            for s in strata
        ),
    )


def _leakage(r: EvaluationReport) -> str:
    fields = [
        "exact_duplicates",
        "shared_pmids",
        "shared_pairs",
        "shared_drugs",
        "shared_variants",
        "shared_genes",
    ]
    rows = []
    for fold, o in zip(r.folds_evaluated, r.overlap, strict=False):
        rows.append(
            [str(fold), str(o.n_test)]
            + [f"{getattr(o, f)} ({getattr(o, f + '_pct'):.1f}%)" for f in fields]
            + [
                f"<span class='warn'>{', '.join(o.violations)}</span>"
                if o.violations
                else "none"
            ]
        )
    return (
        "<h2>6 · Leakage and novelty</h2><h3>By novelty (seen in the fold's train?)</h3>"
        + _strata_table(r.novelty)
        + "<h3>By max Tanimoto similarity to a train drug</h3>"
        + _strata_table(r.tanimoto)
        + "<h3>Train/test overlap per fold</h3>"
        + table(["fold", "n test", *fields, "violations"], rows)
    )


def _baselines(r: EvaluationReport) -> str:
    keys = ("f1_macro", "f2_macro", "pr_auc_macro")
    rows = [["<b>model</b>", *(fmt_ci(r.aggregate.get(k)) for k in keys), "", "", ""]]
    for b in r.baselines:
        rows.append(
            [escape(b.name), *(fmt_ci(b.aggregate.get(k)) for k in keys)]
            + [
                f"{fmt_ci(b.vs_model[k].diff)} p={fmt(b.vs_model[k].p_value, 4)}"
                for k in keys
            ]
        )
    return "<h2>7 · Baselines</h2>" + table(
        ["", *keys, *(f"model − baseline {k}" for k in keys)], rows
    )


def _predictions(preds: pl.DataFrame) -> str:
    head = "".join(
        f'<th onclick="sortTable(this)">{escape(c)}</th>' for c in preds.columns
    )
    body = []
    for row in preds.iter_rows(named=True):
        novel = (
            "true"
            if any(row.get(c) is False for c in ("seen_drug", "seen_variant"))
            else "false"
        )
        cells = []
        for c in preds.columns:
            v = row[c]
            text = (
                fmt(v)
                if isinstance(v, float)
                else ("—" if v is None else escape(str(v)))
            )
            cells.append(f"<td>{text}</td>")
        body.append(
            f"<tr data-exact='{str(row.get('exact_match')).lower()}' "
            f"data-band='{row.get('confidence_band')}' data-novel='{novel}'>{''.join(cells)}</tr>"
        )
    return (
        "<h2>8 · Individual predictions</h2><p>Filter: <select id='flt' onchange='filterRows()'>"
        "<option value='all'>all</option><option value='errors'>errors only</option>"
        "<option value='low'>low confidence</option><option value='novel'>novel drug or variant</option>"
        f"</select> · also in predictions.csv</p><table id='preds'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def render(eval_dir: Path) -> Path:
    """Write ``report.html`` into ``eval_dir`` from its data files."""
    report = EvaluationReport.model_validate_json(
        (eval_dir / "metrics.json").read_text()
    )
    curves = json.loads((eval_dir / "curves.json").read_text())
    preds = pl.read_parquet(eval_dir / "predictions.parquet")
    body = (
        _header(report)
        + _aggregate(report)
        + _per_class(report)
        + _plots(report, curves)
        + _leakage(report)
        + _baselines(report)
        + _predictions(preds)
    )
    out = eval_dir / REPORT_FILE
    out.write_text(page(f"{report.model} · {report.scenario}", body))
    logger.info("Report written to %s", out)
    return out


__all__ = ["REPORT_FILE", "fmt", "fmt_ci", "page", "render", "table"]
