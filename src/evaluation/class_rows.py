"""Class rows: training-only, never a way for held-out labels to leak in.

A class row (``is_class_row``) names a drug class (``ATC:<codes>``), read by
the model as the pool of the class's member drugs. Such rows are always in
**train** — in every fold of every scenario and in the plain train/val split
— and are dropped from a fold's training set when they would carry what the
fold holds out:

* the row shares a PMID with a val/test row of the fold (same paper);
* it shares a resolved variant with a val/test row whose drug is a member
  of the class (the class pool would contain the held-out pair);
* in a grouped scenario, its own entity keys meet the fold's held-out ones
  (a held-out variant / gene, or — in ``drug_cold`` — a drug the row names
  directly; held-out *members* are instead removed from the class pools by
  :meth:`~src.data.cache.GraphCache.without_members`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from src.data.library.bridge import class_members, is_class_key
from src.evaluation.entities import Scenario

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from src.evaluation.entities import EntityResolver

#: Training-row column that marks rows with a drug-class entity.
CLASS_ROW_COLUMN = "is_class_row"


def class_row_mask(df: pl.DataFrame) -> pl.Series:
    """Boolean series: which rows are class rows (all false without the column)."""
    if CLASS_ROW_COLUMN not in df.columns:
        return pl.Series(CLASS_ROW_COLUMN, [False] * len(df), dtype=pl.Boolean)
    return df[CLASS_ROW_COLUMN].cast(pl.Boolean).fill_null(value=False)


@dataclass(frozen=True)
class RowFacts:
    """What the leakage rules compare, resolved once per row."""

    is_class: bool
    pmid: frozenset[str]
    variants: frozenset[str]
    drugs: frozenset[str]  # single-drug entity keys (no ATC keys)
    members: frozenset[str]  # ChEMBL IDs pooled by the row's class entities
    keys: frozenset[str]  # the scenario's entity keys (empty: no key rule)


def row_facts(
    df: pl.DataFrame,
    resolver: EntityResolver,
    members: Mapping[str, frozenset[str]],
    scenario: Scenario | None = None,
) -> list[RowFacts]:
    """:class:`RowFacts` of every row of ``df``."""
    mask = class_row_mask(df).to_list()
    by_scenario = {
        Scenario.PAIR: resolver.pair,
        Scenario.PMID: resolver.pmid,
        Scenario.DRUG_COLD: resolver.drugs,
        Scenario.VARIANT_COLD: resolver.variants,
        Scenario.GENE_COLD: resolver.genes,
    }
    keys_of = by_scenario.get(scenario) if scenario is not None else None
    facts = []
    for is_class, row in zip(mask, df.iter_rows(named=True), strict=True):
        entities = resolver.drugs(row)
        classes = [e for e in entities if is_class_key(e)]
        keys = keys_of(row) if keys_of is not None else frozenset()
        facts.append(
            RowFacts(
                is_class=bool(is_class),
                pmid=resolver.pmid(row),
                variants=resolver.variants(row),
                drugs=frozenset(e for e in entities if not is_class_key(e)),
                members=frozenset().union(
                    *(class_members(c, members) for c in classes)
                ),
                keys=frozenset(k for k in keys if not is_class_key(k)),
            )
        )
    return facts


def leaking_rows(train: Sequence[RowFacts], holdout: Iterable[RowFacts]) -> list[bool]:
    """Which ``train`` rows are class rows that leak against ``holdout``."""
    pmids: set[str] = set()
    variant_drugs: dict[str, set[str]] = {}
    held_keys: set[str] = set()
    for h in holdout:
        pmids |= h.pmid
        held_keys |= h.keys
        for v in h.variants:
            variant_drugs.setdefault(v, set()).update(h.drugs)
    out = []
    for r in train:
        out.append(
            r.is_class
            and bool(
                r.pmid & pmids
                or r.keys & held_keys
                or any(variant_drugs.get(v, set()) & r.members for v in r.variants)
            )
        )
    return out


def leaking_class_rows(
    train: pl.DataFrame,
    holdout: pl.DataFrame,
    resolver: EntityResolver,
    members: Mapping[str, frozenset[str]],
    *,
    scenario: Scenario | None = None,
) -> pl.Series:
    """Boolean series over ``train``: class rows that leak against ``holdout``.

    With ``scenario`` the grouped-scenario key rule applies too (see the
    module docstring); without it only the PMID and member-variant rules.
    """
    leaks = leaking_rows(
        row_facts(train, resolver, members, scenario),
        row_facts(holdout, resolver, members, scenario),
    )
    return pl.Series("leaks", leaks, dtype=pl.Boolean)


__all__ = [
    "CLASS_ROW_COLUMN",
    "RowFacts",
    "class_row_mask",
    "leaking_class_rows",
    "leaking_rows",
    "row_facts",
]
