"""Multi-format variant ingestion for the genotype tower.

Every input format (generic VCF, PharmVar, raw HGVS list, VEP/SnpEff annotation)
is normalised here into the shared :class:`~src.data.library.ingest.models.IngestedVariant`
/ :class:`~src.data.library.ingest.models.IngestedHaplotype` models, with a
canonical genomic HGVS (``g.``) key. Downstream the per-gene graph builder
consumes only these models, so adding a format means adding an adapter, not
touching the builder.
"""

from src.data.library.ingest.hgvs_build import genomic_hgvs, genomic_hgvs_body
from src.data.library.ingest.models import IngestedHaplotype, IngestedVariant

__all__ = [
    "IngestedHaplotype",
    "IngestedVariant",
    "genomic_hgvs",
    "genomic_hgvs_body",
]
