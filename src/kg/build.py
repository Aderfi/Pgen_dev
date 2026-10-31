"""Pure KG build: local files + raw cache → typed rows with provenance.

Nothing here touches the network or Neo4j. Every node and relationship is
stamped with the datasets it derives from and the run that produced it;
derived relationships also carry the method and the raw evidence. A missing
cache entry anywhere aborts the whole build (``CacheMissError`` listing every
missing key) before a single row is returned.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.domain.schemas.annotation import AnnotationSource
from src.kg.allele_inference import METHOD as INFERENCE_METHOD
from src.kg.allele_inference import infer_variant
from src.kg.labels import AlleleKind, ClassifiedLabel, classify_label
from src.kg.provenance import Dataset, IngestRun, Stamp, sha256_file
from src.kg.raw_cache import CacheMissError
from src.kg.resolve import ResolvedRefSNP, canonicalize_row, resolve_refsnps
from src.kg.sources.allele_tables import LICENSE as CLINPGX_LICENSE
from src.kg.sources.dbsnp import Unresolved

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from src.domain.schemas.annotation import PGxAnnotation
    from src.kg.ncbi import NCBIClient
    from src.kg.raw_cache import CacheEntry
    from src.kg.sources.definitions import HaplotypeDefinition
    from src.kg.variant import CanonicalVariant

logger = logging.getLogger(__name__)

PHARMGKB_LICENSE = "CC BY-SA 4.0"
NCBI_LICENSE = "public domain (NCBI)"


class _DatasetIds(BaseModel):
    """Dataset ids every element is stamped with, resolved once per build."""

    annotation: dict[str, str]  # AnnotationSource value -> dataset id
    dbsnp: dict[str, str]  # dbSNP build -> dataset id
    spdi: str

    def for_refsnp(self, resolved: ResolvedRefSNP) -> str:
        return self.dbsnp[resolved.record.dbsnp_build]


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


class NodeRow(BaseModel):
    label: str
    key: dict[str, str]
    props: dict[str, object] = Field(default_factory=dict)
    stamp: Stamp


class RelRow(BaseModel):
    type: str
    start_label: str
    start_key: dict[str, str]
    end_label: str
    end_key: dict[str, str]
    props: dict[str, object] = Field(default_factory=dict)
    stamp: Stamp


class ResolutionRow(BaseModel):
    label: str
    kind: str
    status: Literal["resolved", "unresolved"]
    reason: str | None
    n_annotations: int
    n_variants: int
    path: Literal["rsid", "haplotype"] | None


class DisagreementRow(BaseModel):
    issue: Literal["allele_not_in_dbsnp", "rsid_unresolved"]
    haplotype: str
    rsid: str
    pharmvar_spdi: str
    dbsnp_spdis: tuple[str, ...]


class KGBuild(BaseModel):
    datasets: list[Dataset]
    run: IngestRun
    nodes: list[NodeRow]
    rels: list[RelRow]
    resolution: list[ResolutionRow]
    disagreements: list[DisagreementRow]
    inference_counts: dict[str, int]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


def pharmgkb_dataset(tsv: Path, annotations_dir: Path) -> Dataset:
    stamps = sorted(annotations_dir.glob("CREATED_*.txt"))
    version = stamps[-1].stem.removeprefix("CREATED_") if stamps else "unknown"
    return Dataset(
        id=f"pharmgkb:{tsv.stem}@{sha256_file(tsv)[:12]}",
        name=f"PharmGKB {tsv.stem}",
        version=version,
        uri="https://www.clinpgx.org/downloads",
        retrieved_at=datetime.fromtimestamp(tsv.stat().st_mtime, UTC),
        sha256=sha256_file(tsv),
        license=PHARMGKB_LICENSE,
    )


def local_file_dataset(logical_id: str, path: Path, name: str) -> Dataset:
    return Dataset(
        id=f"{logical_id}@{sha256_file(path)[:12]}",
        name=name,
        version=datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat(),
        uri=f"file:{path.name}",
        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
        sha256=sha256_file(path),
    )


def ncbi_dataset(entries: list[CacheEntry], name: str, version: str) -> Dataset:
    """Content-addressed dataset over exactly the cached responses it covers."""
    digest = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e.key):
        digest.update(e.model_dump_json().encode() + b"\n")
    sha = digest.hexdigest()
    latest = max((e.retrieved_at for e in entries), default=datetime.now(UTC))
    return Dataset(
        id=f"{name}@{version}#{sha[:12]}",
        name=name,
        version=version,
        uri="https://www.ncbi.nlm.nih.gov/",
        retrieved_at=latest,
        sha256=sha,
        license=NCBI_LICENSE,
    )


def allele_table_dataset(
    xlsx: Path, gene: str, url: str, retrieved_at: datetime
) -> Dataset:
    return Dataset(
        id=f"clinpgx:{gene}_allele_definition_table@{sha256_file(xlsx)[:12]}",
        name=f"ClinPGx {gene} allele definition table",
        version=retrieved_at.date().isoformat(),
        uri=url,
        retrieved_at=retrieved_at,
        sha256=sha256_file(xlsx),
        license=CLINPGX_LICENSE,
    )


# --------------------------------------------------------------------------- #
# Accumulator — merges repeated nodes/relationships, unioning their sources
# --------------------------------------------------------------------------- #


def _frozen(key: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(key.items()))


def _merge_stamp(old: Stamp, new: Stamp) -> Stamp:
    sources = tuple(dict.fromkeys((*old.sources, *new.sources)))
    return old.model_copy(update={"sources": sources})


class _Graph:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.nodes: dict[tuple, NodeRow] = {}
        self.rels: dict[tuple, RelRow] = {}

    def stamp(
        self,
        sources: Iterable[str],
        method: str | None = None,
        evidence: str | None = None,
    ) -> Stamp:
        return Stamp(
            sources=tuple(dict.fromkeys(sources)),
            run_id=self.run_id,
            method=method,
            evidence=evidence,
        )

    def node(
        self, label: str, key: dict[str, str], props: dict[str, object], stamp: Stamp
    ) -> None:
        k = (label, _frozen(key))
        if k in self.nodes:
            old = self.nodes[k]
            merged = {**old.props, **{p: v for p, v in props.items() if v is not None}}
            self.nodes[k] = old.model_copy(
                update={"props": merged, "stamp": _merge_stamp(old.stamp, stamp)}
            )
        else:
            self.nodes[k] = NodeRow(label=label, key=key, props=props, stamp=stamp)

    def rel(
        self,
        rel_type: str,
        start: tuple[str, dict[str, str]],
        end: tuple[str, dict[str, str]],
        stamp: Stamp,
        props: dict[str, object] | None = None,
    ) -> None:
        k = (rel_type, start[0], _frozen(start[1]), end[0], _frozen(end[1]))
        if k in self.rels:
            old = self.rels[k]
            self.rels[k] = old.model_copy(
                update={"stamp": _merge_stamp(old.stamp, stamp)}
            )
        else:
            self.rels[k] = RelRow(
                type=rel_type,
                start_label=start[0],
                start_key=start[1],
                end_label=end[0],
                end_key=end[1],
                props=props or {},
                stamp=stamp,
            )

    def variant(self, v: CanonicalVariant, stamp: Stamp) -> tuple[str, dict[str, str]]:
        props = v.model_dump(exclude={"spdi"})
        self.node("Variant", {"spdi": v.spdi}, props, stamp)
        return ("Variant", {"spdi": v.spdi})


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_kg(
    *,
    annotations: list[PGxAnnotation],
    definitions: list[HaplotypeDefinition],
    client: NCBIClient,
    drug_cids: Mapping[str, tuple[int, str]],
    datasets: list[Dataset],
    run: IngestRun,
    annotation_datasets: Mapping[str, str] | None = None,
) -> KGBuild:
    """``annotation_datasets`` maps an AnnotationSource value to its Dataset id
    (default ``pharmgkb:<source>``)."""
    missing: list[str] = []

    def on_miss(e: CacheMissError) -> None:
        missing.extend(e.missing)

    classified = {lab: classify_label(lab) for a in annotations for lab in a.variants}
    label_rsids = {c.rsid for c in classified.values() if c.rsid is not None}
    pharmvar_rsids = {
        int(r.rsid[2:])
        for d in definitions
        if d.dataset_id.startswith("pharmvar:")
        for r in d.rows
        if r.rsid and r.rsid[2:].isdigit()
    }
    try:
        refsnps = resolve_refsnps(label_rsids | pharmvar_rsids, client, on_miss)
    except CacheMissError as e:
        on_miss(e)
        refsnps = {}

    defs_by_name: dict[str, HaplotypeDefinition] = {}
    for d in definitions:  # PharmVar first: it wins over a ClinPGx table
        defs_by_name.setdefault(d.name, d)
    canonical: dict[
        str, list[tuple[CanonicalVariant | None, str, tuple[str, ...], str | None]]
    ] = {}
    for name, d in defs_by_name.items():
        rows = []
        for row in d.rows:
            try:
                c = canonicalize_row(row, client)
            except CacheMissError as e:
                on_miss(e)
                continue
            rows.append((c.variant, row.evidence, c.warnings, c.failure))
        canonical[name] = rows

    if missing:
        raise CacheMissError(sorted(set(missing)))

    g = _Graph(run.id)
    ncbi: dict[str, Dataset] = {}
    builds = {
        r.record.dbsnp_build for r in refsnps.values() if isinstance(r, ResolvedRefSNP)
    }
    rs_entries = [
        e for e in client.dbsnp_cache.entries() if e.key.startswith("dbsnp:rs")
    ]
    for b in sorted(builds):
        ds = ncbi_dataset(
            [e for e in rs_entries if e.source_version == b], "dbsnp_esummary", b
        )
        ncbi[b] = ds
    spdi_ds = ncbi_dataset(client.spdi_cache.entries(), "ncbi_spdi", "variation-v0")
    ids = _DatasetIds(
        annotation={
            src.value: (annotation_datasets or {}).get(
                src.value, f"pharmgkb:{src.value}"
            )
            for src in AnnotationSource
        },
        dbsnp={b: ds.id for b, ds in ncbi.items()},
        spdi=spdi_ds.id,
    )

    _add_refsnps(g, refsnps, ids)
    hap_variants, hap_failed = _add_haplotypes(g, defs_by_name, canonical, ids)
    disagreements = _cross_check(defs_by_name, canonical, refsnps)
    inference = _add_annotations(g, annotations, classified, refsnps, drug_cids, ids)
    resolution = _add_alleles(
        g,
        annotations,
        classified,
        refsnps,
        defs_by_name,
        hap_variants,
        hap_failed,
        ids,
    )

    # Exactly the datasets some element cites — no more, no fewer.
    cited = {
        s for row in [*g.nodes.values(), *g.rels.values()] for s in row.stamp.sources
    }
    available = {d.id: d for d in [*datasets, *ncbi.values(), spdi_ds]}
    uncited = cited - set(available)
    if uncited:
        msg = f"elements cite datasets that were not provided: {sorted(uncited)}"
        raise ValueError(msg)
    out_datasets = [available[i] for i in sorted(cited)]

    logger.info(
        "KG build: %d nodes, %d relationships, %d labels, %d disagreements",
        len(g.nodes),
        len(g.rels),
        len(resolution),
        len(disagreements),
    )
    return KGBuild(
        datasets=out_datasets,
        run=run,
        nodes=list(g.nodes.values()),
        rels=list(g.rels.values()),
        resolution=resolution,
        disagreements=disagreements,
        inference_counts=dict(inference),
    )


def _add_refsnps(
    g: _Graph, refsnps: Mapping[int, ResolvedRefSNP | Unresolved], ids: _DatasetIds
) -> None:
    for resolved in refsnps.values():
        if not isinstance(resolved, ResolvedRefSNP):
            continue
        rec = resolved.record
        dbsnp_id = ids.for_refsnp(resolved)
        stamp = g.stamp([dbsnp_id])
        current = ("RefSNP", {"rsid": rec.current_rsid})
        g.node(
            "RefSNP",
            {"rsid": rec.current_rsid},
            {"dbsnp_build": rec.dbsnp_build},
            stamp,
        )
        if rec.merged:
            g.node(
                "RefSNP",
                {"rsid": rec.requested_rsid},
                {"dbsnp_build": rec.dbsnp_build, "merged": True},
                stamp,
            )
            g.rel(
                "MERGED_INTO",
                ("RefSNP", {"rsid": rec.requested_rsid}),
                current,
                g.stamp(
                    [dbsnp_id],
                    "dbsnp_esummary_merge",
                    f"requested {rec.requested_rsid}; record snp_id {rec.current_rsid}",
                ),
            )
        for gene in rec.genes:
            g.node(
                "Gene", {"symbol": gene.symbol}, {"entrez_id": gene.entrez_id}, stamp
            )
        for v in resolved.variants:
            vref = g.variant(v, stamp)
            g.rel(
                "HAS_ALLELE",
                current,
                vref,
                g.stamp([dbsnp_id], "dbsnp_esummary_spdi", v.spdi),
            )
            for gene in rec.genes:
                g.rel(
                    "IN_GENE",
                    vref,
                    ("Gene", {"symbol": gene.symbol}),
                    g.stamp(
                        [dbsnp_id],
                        "dbsnp_esummary_gene",
                        f"{gene.symbol}:{gene.entrez_id}",
                    ),
                )


def _add_haplotypes(
    g: _Graph,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    canonical: Mapping[str, list],
    ids: _DatasetIds,
) -> tuple[dict[str, int], set[str]]:
    n_variants: dict[str, int] = {}
    failed: set[str] = set()
    for name, d in defs_by_name.items():
        g.node(
            "Haplotype",
            {"name": name},
            {"gene": d.gene, "is_reference": d.is_reference},
            g.stamp([d.dataset_id]),
        )
        rows = canonical.get(name, [])
        if d.unresolved_reason or any(failure for _v, _e, _w, failure in rows):
            failed.add(name)
            continue  # never write a partial cis set
        method = (
            "pharmvar_haplotype_row"
            if d.dataset_id.startswith("pharmvar:")
            else "clinpgx_allele_table"
        )
        for variant, evidence, warnings, _failure in rows:
            assert variant is not None
            vref = g.variant(variant, g.stamp([d.dataset_id, ids.spdi]))
            note = evidence + ("".join(f"\tncbi_warning={w}" for w in warnings))
            g.rel(
                "HAS_VARIANT",
                ("Haplotype", {"name": name}),
                vref,
                g.stamp(
                    [d.dataset_id, ids.spdi],
                    f"{method}+ncbi_spdi_canonical",
                    note,
                ),
                props={"phase": "cis"},
            )
        n_variants[name] = len(rows)
    for name, d in defs_by_name.items():
        if d.core_name != name and d.core_name in defs_by_name:
            g.rel(
                "SUBALLELE_OF",
                ("Haplotype", {"name": name}),
                ("Haplotype", {"name": d.core_name}),
                g.stamp(
                    [d.dataset_id], "haplotype_name_suffix", f"{name} -> {d.core_name}"
                ),
            )
    return n_variants, failed


def _cross_check(
    defs_by_name: Mapping[str, HaplotypeDefinition],
    canonical: Mapping[str, list],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
) -> list[DisagreementRow]:
    out: list[DisagreementRow] = []
    for name, d in defs_by_name.items():
        if not d.dataset_id.startswith("pharmvar:"):
            continue
        for row, (variant, _e, _w, _f) in zip(
            d.rows, canonical.get(name, []), strict=False
        ):
            if variant is None or not row.rsid or not row.rsid[2:].isdigit():
                continue
            resolved = refsnps.get(int(row.rsid[2:]))
            dbsnp = (
                tuple(v.spdi for v in resolved.variants)
                if isinstance(resolved, ResolvedRefSNP)
                else ()
            )
            if variant.spdi not in dbsnp:
                issue = "allele_not_in_dbsnp" if dbsnp else "rsid_unresolved"
                out.append(
                    DisagreementRow(
                        issue=issue,
                        haplotype=name,
                        rsid=row.rsid,
                        pharmvar_spdi=variant.spdi,
                        dbsnp_spdis=dbsnp,
                    )
                )
    return out


def _add_annotations(
    g: _Graph,
    annotations: list[PGxAnnotation],
    classified: Mapping[str, ClassifiedLabel],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    drug_cids: Mapping[str, tuple[int, str]],
    ids: _DatasetIds,
) -> Counter[str]:
    inference: Counter[str] = Counter()
    for a in annotations:
        src = ids.annotation[a.source.value]
        stamp = g.stamp([src])
        aref = ("Annotation", {"id": a.id})
        g.node(
            "Annotation",
            {"id": a.id},
            {
                "source": a.source.value,
                "pmid": a.pmid,
                "phenotype_categories": a.phenotype_categories,
                "direction": a.direction,
                "significance": a.significance,
                "associated": a.associated,
                "alleles": a.alleles,
                "comparison_alleles": a.comparison_alleles,
            },
            stamp,
        )
        for symbol in a.genes:
            g.node("Gene", {"symbol": symbol}, {}, stamp)
            g.rel("CONCERNS_GENE", aref, ("Gene", {"symbol": symbol}), stamp)
        for drug in a.drugs:
            cid = drug_cids.get(drug)
            dstamp = g.stamp([src, cid[1]]) if cid else stamp
            g.node("Drug", {"name": drug}, {"cid": cid[0] if cid else None}, dstamp)
            g.rel("FOR_DRUG", aref, ("Drug", {"name": drug}), stamp)
        for label in a.variants:
            g.rel("CONCERNS_ALLELE", aref, ("Allele", {"label": label}), stamp)

        # Allele-specific link only when the annotation is about a single rsID.
        if len(a.variants) != 1:
            continue
        c = classified[a.variants[0]]
        resolved = refsnps.get(c.rsid) if c.rsid is not None else None
        if not isinstance(resolved, ResolvedRefSNP):
            continue
        result = infer_variant(a.alleles, a.comparison_alleles, resolved.variants)
        inference[result.reason or "linked"] += 1
        if result.variant is not None:
            g.rel(
                "IMPLICATES",
                aref,
                ("Variant", {"spdi": result.variant.spdi}),
                g.stamp(
                    [src, ids.for_refsnp(resolved)], INFERENCE_METHOD, result.evidence
                ),
            )
    return inference


_LabelResolution = tuple[
    str | None,  # unresolved reason
    int,  # number of variants reached
    Literal["rsid", "haplotype"] | None,
    tuple[str, dict[str, str]] | None,  # REFERS_TO target
    str | None,  # dbSNP dataset id (rsid path)
]
_UNRESOLVED_RSID: _LabelResolution = ("rsid_not_found", 0, None, None, None)


def _resolve_rsid(
    c: ClassifiedLabel,
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    ids: _DatasetIds,
) -> _LabelResolution:
    resolved = refsnps.get(c.rsid) if c.rsid is not None else None
    if isinstance(resolved, ResolvedRefSNP):
        target = ("RefSNP", {"rsid": resolved.record.requested_rsid})
        return None, len(resolved.variants), "rsid", target, ids.for_refsnp(resolved)
    if isinstance(resolved, Unresolved):
        return resolved.reason, 0, None, None, None
    return _UNRESOLVED_RSID


def _resolve_star(
    name: str,
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_variants: Mapping[str, int],
    hap_failed: set[str],
) -> _LabelResolution:
    d = defs_by_name.get(name)
    if d is None:
        return "not_in_pharmvar_or_tables", 0, None, None, None
    if name in hap_failed:
        return d.unresolved_reason or "canonicalization_failed", 0, None, None, None
    return (
        None,
        hap_variants.get(name, 0),
        "haplotype",
        ("Haplotype", {"name": name}),
        None,
    )


def _resolve_label(
    c: ClassifiedLabel,
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_variants: Mapping[str, int],
    hap_failed: set[str],
    ids: _DatasetIds,
) -> _LabelResolution:
    """``(reason, n_variants, path, REFERS_TO target, dbSNP dataset id)`` for one label."""
    if c.kind is AlleleKind.RSID:
        return _resolve_rsid(c, refsnps, ids)
    if c.kind is AlleleKind.STAR and c.haplotype_name is not None:
        return _resolve_star(c.haplotype_name, defs_by_name, hap_variants, hap_failed)
    return c.reason, 0, None, None, None


def _add_alleles(
    g: _Graph,
    annotations: list[PGxAnnotation],
    classified: Mapping[str, ClassifiedLabel],
    refsnps: Mapping[int, ResolvedRefSNP | Unresolved],
    defs_by_name: Mapping[str, HaplotypeDefinition],
    hap_variants: Mapping[str, int],
    hap_failed: set[str],
    ids: _DatasetIds,
) -> list[ResolutionRow]:
    counts = Counter(lab for a in annotations for lab in a.variants)
    sources_of: dict[str, set[str]] = {}
    for a in annotations:  # every file that mentions a label is a source of it
        for lab in a.variants:
            sources_of.setdefault(lab, set()).add(ids.annotation[a.source.value])
    rows: list[ResolutionRow] = []
    for label, c in classified.items():
        kind = c.kind
        reason, n_var, path, target, refsnp_id = _resolve_label(
            c, refsnps, defs_by_name, hap_variants, hap_failed, ids
        )
        status: Literal["resolved", "unresolved"] = (
            "resolved" if target else "unresolved"
        )
        label_sources = sorted(sources_of[label])
        stamp = g.stamp(label_sources)
        g.node(
            "Allele",
            {"label": label},
            {
                "kind": kind.value,
                "status": status,
                "reason": reason if not target else None,
            },
            stamp,
        )
        if target is not None:
            if path == "rsid":
                method, extra = "label_rsid", [refsnp_id] if refsnp_id else []
            else:
                method, extra = (
                    "label_haplotype_name",
                    [defs_by_name[target[1]["name"]].dataset_id],
                )
            g.rel(
                "REFERS_TO",
                ("Allele", {"label": label}),
                target,
                g.stamp([*label_sources, *extra], method, label),
            )
        rows.append(
            ResolutionRow(
                label=label,
                kind=kind.value,
                status=status,
                reason=None if target else reason,
                n_annotations=counts[label],
                n_variants=n_var,
                path=path,
            )
        )
    return rows


__all__ = [
    "DisagreementRow",
    "KGBuild",
    "NodeRow",
    "RelRow",
    "ResolutionRow",
    "allele_table_dataset",
    "build_kg",
    "local_file_dataset",
    "ncbi_dataset",
    "pharmgkb_dataset",
]
