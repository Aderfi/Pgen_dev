"""Pharmacogenomic knowledge graph (Neo4j) — the training dataset source.

Workflow::

    python -m src.kg fetch            # the only networked step -> raw cache
    python -m src.kg ingest --reset   # offline build -> Neo4j, then provenance audit
    python -m src.kg export           # Neo4j -> parquet tables under data/library/kg/
    python -m src.kg audit | stats

Variants are canonical NCBI SPDI on GRCh38; star alleles are their *cis*
variant sets (PharmVar, ClinPGx allele tables); rsIDs are sites with all their
dbSNP alleles. Every node and relationship carries ``sources`` (Dataset ids)
and ``run_id`` (W3C PROV-O style), see ``src/kg/provenance.py``.

Training reads annotation rows straight from Neo4j (``python main.py train
--source neo4j``) or from the exported snapshot
(``data/library/kg/training_rows.parquet``); see :mod:`src.kg.training`.
"""

from src.kg.training import load_training_rows, training_snapshot_path

__all__ = ["load_training_rows", "training_snapshot_path"]
