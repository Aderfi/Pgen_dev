"""Offline DDI graph artifact ingest — CSV export from the interaction KG to a
static drug-drug-interaction adjacency artifact.

The polypharmacy tower needs drug-drug interaction edges keyed by drug ID
(ChEMBL parent ID, the key of every drug in dev_Pgen) at train and inference
time, but the source of truth is a Neo4j knowledge graph (see
``docs/graph_schemas.md``). Rather than querying Neo4j at runtime, the KG is
exported offline to a simple edge CSV (``drug_a, drug_b, category,
severity``); this module turns that CSV into a static ``.pt`` artifact the
dataset can load and query cheaply.

The interaction ``category`` vocabulary is frozen in
``data/dicts/ddi_categories.tsv`` (mirrors the ``star_alleles.tsv``
convention) so the one-hot width — and therefore ``ddi_edge_dim`` — is stable
across training and inference runs.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from src.config.settings import PROJECT_ROOT

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CATEGORY_TSV = PROJECT_ROOT / "data" / "dicts" / "ddi_categories.tsv"


@lru_cache(maxsize=1)
def _load_category_vocab(tsv_path: Path | None = None) -> tuple[str, ...]:
    """Read the frozen closed vocabulary of DDI interaction categories.

    Raises FileNotFoundError if the TSV is missing — fail fast rather than
    silently producing a vocab of width zero (which would change
    ``ddi_edge_dim`` under callers' feet).
    """
    path = tsv_path or _DEFAULT_CATEGORY_TSV
    if not path.exists():
        msg = f"DDI category vocabulary not found at {path}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "category" not in reader.fieldnames:
            msg = f"DDI category TSV missing header at {path}"
            raise ValueError(msg)
        return tuple(
            row["category"].strip() for row in reader if row.get("category", "").strip()
        )


def _category_vocab() -> tuple[str, ...]:
    return _load_category_vocab(None)


def ddi_edge_dim() -> int:
    """Fixed edge-feature width: one-hot(category) ++ severity scalar."""
    return len(_category_vocab()) + 1


# Module-level constant for callers that just want the fixed width without a
# call. Resolved eagerly against the frozen vocab file on import.
DDI_EDGE_DIM: int = ddi_edge_dim()


_FALLBACK_CATEGORY = "unknown"


def _encode_row(category: str, severity: float, vocab: tuple[str, ...]) -> list[float]:
    """One-hot(category over vocab) ++ [severity].

    An out-of-vocab category is routed to the frozen ``"unknown"`` bucket (which
    exists for exactly this) rather than an all-zero segment, so KG-export drift
    stays distinguishable from a masked/padding row and the one-hot is never
    silently wrong.
    """
    resolved = category
    if category not in vocab:
        if _FALLBACK_CATEGORY not in vocab:
            msg = (
                f"DDI category {category!r} not in frozen vocab {vocab} and no "
                f"{_FALLBACK_CATEGORY!r} fallback bucket exists"
            )
            raise ValueError(msg)
        logger.warning(
            "DDI category %r not in frozen vocab %s; mapping to %r",
            category,
            vocab,
            _FALLBACK_CATEGORY,
        )
        resolved = _FALLBACK_CATEGORY
    one_hot = [1.0 if c == resolved else 0.0 for c in vocab]
    return [*one_hot, float(severity)]


def build_ddi_artifact(edges_csv: Path, out_path: Path) -> None:
    """Read a DDI edge CSV and save a static adjacency artifact.

    ``edges_csv`` columns: ``drug_a, drug_b, category, severity``. The graph is
    treated as undirected: each row contributes a neighbour edge in both
    directions, with the same encoded edge-attr row on each side.

    Saves (via ``torch.save``) a dict::

        {drug_id: {"neighbors": list[str], "edge_attr": Tensor[E, ddi_edge_dim]}}
    """
    vocab = _category_vocab()
    dim = len(vocab) + 1

    neighbors: dict[str, list[str]] = {}
    edge_rows: dict[str, list[list[float]]] = {}

    with edges_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug_a = row["drug_a"].strip()
            drug_b = row["drug_b"].strip()
            category = row["category"].strip()
            severity = float(row["severity"])
            attr = _encode_row(category, severity, vocab)

            neighbors.setdefault(drug_a, []).append(drug_b)
            edge_rows.setdefault(drug_a, []).append(attr)
            neighbors.setdefault(drug_b, []).append(drug_a)
            edge_rows.setdefault(drug_b, []).append(attr)

    artifact: dict[str, dict[str, object]] = {}
    for drug_id, neigh in neighbors.items():
        artifact[drug_id] = {
            "neighbors": neigh,
            "edge_attr": torch.tensor(edge_rows[drug_id], dtype=torch.float32).reshape(
                -1, dim
            ),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    logger.info(
        "Built DDI graph artifact: %d nodes, edge_dim=%d -> %s",
        len(artifact),
        dim,
        out_path,
    )


class DDIGraph:
    """Static drug-drug interaction adjacency, keyed by drug ID (ChEMBL).

    Load once via :meth:`load`, then query neighbours per drug at dataset
    __getitem__ time. Unknown drugs return an empty neighbour set rather than
    raising, so a drug with no known interactions is still a valid sample.
    """

    def __init__(self, data: dict[str, dict[str, object]]):
        self._data = data

    @classmethod
    def load(cls, path: Path) -> DDIGraph:
        data = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data)

    def neighbors(self, drug_id: str, k: int) -> tuple[list[str], torch.Tensor]:
        """Up to ``k`` neighbour drug IDs and their ``[E, ddi_edge_dim]`` edge-attr
        rows (``E == len(returned neighbours)``). Unknown drugs yield
        ``([], empty[0, ddi_edge_dim])``."""
        entry = self._data.get(drug_id)
        if entry is None:
            return [], torch.empty((0, DDI_EDGE_DIM), dtype=torch.float32)

        neigh = list(entry["neighbors"])[:k]
        attr = entry["edge_attr"][: len(neigh)]
        return neigh, attr


__all__ = ["DDI_EDGE_DIM", "DDIGraph", "build_ddi_artifact", "ddi_edge_dim"]
