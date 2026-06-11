"""Sequence Ontology molecular-consequence featurizer (geno_global_feats, Layer B).

Why this exists
---------------
A variant's ``FXN_CLASS`` (the comma-separated Sequence Ontology terms supplied
by dbSNP/VEP, e.g. ``"missense_variant,coding_sequence_variant"``) encodes the
*molecular consequence* — the same information HGVS notation expresses in its
``VariantKind`` / position grammar. The current gene-graph builder collapses all
of this into four coarse flags (coding / regulatory / splicing / intergenic),
discarding the severity hierarchy: a ``stop_gained``, a ``frameshift`` and a
benign ``synonymous`` change all read as "coding".

This module turns the raw SO-term blob into a compact, severity-aware vector:
an 11-way **multi-hot** over consequence groups, a normalised **max-severity**
scalar, and a **known** mask (0 when ``FXN_CLASS`` was empty). A variant can
carry several terms across transcripts; the multi-hot records every group present
and the scalar records the single most severe — the VEP "most severe
consequence" convention.

The featurizer is pure and local (no I/O); :mod:`src.data.library.geno_func`
composes its output into the full ``geno_global_feats`` vector.
"""

from __future__ import annotations

# Consequence groups, in vector order. Each raw SO term maps to one group and a
# severity rank (0-10, higher = more disruptive); the rank drives the scalar.
_GROUPS: tuple[str, ...] = (
    "stop_gained",
    "frameshift",
    "splice",
    "start_stop_lost",
    "missense",
    "inframe_indel",
    "synonymous",
    "coding_other",
    "utr",
    "up_downstream",
    "intron",
)
_GROUP_INDEX: dict[str, int] = {name: i for i, name in enumerate(_GROUPS)}

# Raw Sequence Ontology term -> (group, severity rank). Terms absent here are
# ignored (they contribute no group and no severity). ``non_coding_transcript_
# variant`` is intentionally dropped: it co-occurs as a transcript-biotype tag,
# not a consequence, so it must not mask a real coding consequence.
_SO_TERMS: dict[str, tuple[str, int]] = {
    "stop_gained": ("stop_gained", 10),
    "frameshift_variant": ("frameshift", 10),
    "splice_acceptor_variant": ("splice", 9),
    "splice_donor_variant": ("splice", 9),
    "splice_region_variant": ("splice", 7),
    "start_lost": ("start_stop_lost", 8),
    "initiator_codon_variant": ("start_stop_lost", 8),
    "stop_lost": ("start_stop_lost", 8),
    "terminator_codon_variant": ("start_stop_lost", 8),
    "missense_variant": ("missense", 6),
    "inframe_deletion": ("inframe_indel", 6),
    "inframe_insertion": ("inframe_indel", 6),
    "inframe_indel": ("inframe_indel", 6),
    "coding_sequence_variant": ("coding_other", 4),
    "synonymous_variant": ("synonymous", 3),
    "5_prime_UTR_variant": ("utr", 2),
    "3_prime_UTR_variant": ("utr", 2),
    "intron_variant": ("intron", 2),
    "upstream_transcript_variant": ("up_downstream", 1),
    "downstream_transcript_variant": ("up_downstream", 1),
    "genic_upstream_transcript_variant": ("up_downstream", 1),
    "genic_downstream_transcript_variant": ("up_downstream", 1),
    "2KB_upstream_variant": ("up_downstream", 1),
    "500B_downstream_variant": ("up_downstream", 1),
}

_MAX_SEVERITY = 10.0

#: Length of the SO-consequence block: 11 group flags + max-severity + known mask.
CONSEQUENCE_DIM: int = len(_GROUPS) + 2


def split_so_terms(fxn_class: str | None) -> list[str]:
    """Split a raw ``FXN_CLASS`` cell into its individual SO terms."""
    if not fxn_class:
        return []
    return [t.strip() for t in str(fxn_class).split(",") if t.strip()]


def consequence_vector(fxn_class: str | None) -> list[float]:
    """Featurize a ``FXN_CLASS`` blob into the :data:`CONSEQUENCE_DIM` vector.

    Layout: ``[*group_multi_hot(11), max_severity_norm, consequence_known]``.
    An empty / missing blob yields all-zeros with ``consequence_known = 0`` — a
    valid "no consequence annotation" input distinguishable from a true zero.
    """
    vec = [0.0] * CONSEQUENCE_DIM
    terms = split_so_terms(fxn_class)
    matched = [_SO_TERMS[t] for t in terms if t in _SO_TERMS]
    if not matched:
        return vec  # known mask stays 0

    max_rank = 0
    for group, rank in matched:
        vec[_GROUP_INDEX[group]] = 1.0
        max_rank = max(max_rank, rank)
    vec[len(_GROUPS)] = max_rank / _MAX_SEVERITY  # max-severity (normalised)
    vec[len(_GROUPS) + 1] = 1.0  # consequence_known mask
    return vec


__all__ = ["CONSEQUENCE_DIM", "consequence_vector", "split_so_terms"]
