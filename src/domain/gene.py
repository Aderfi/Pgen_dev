"""Gene and star-allele models.

Star-allele naming follows the PharmGKB/PharmVar convention: ``GENE*N`` where
``N`` is the allele identifier, optionally with a sub-allele suffix (``*4.1``)
and a single-letter modifier (``*4A``). The matching regex is intentionally
strict — anything else should be cleaned upstream rather than accepted here.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HGNC_SYMBOL = re.compile(r"^[A-Z][A-Z0-9-]*$")
_ENSG_ID = re.compile(r"^ENSG\d{11}(?:\.\d+)?$")
_STAR_ALLELE = re.compile(r"^([A-Z][A-Z0-9-]+)\*(\d+(?:\.\d+)?[A-Z]?)$")


class AlleleFunction(str, Enum):
    """Functional classification of a star allele.

    Aligned with the CPIC/PharmGKB allele functionality categories. Values are
    the canonical short labels used in CPIC tables.
    """

    INCREASED = "increased_function"
    NORMAL = "normal_function"
    DECREASED = "decreased_function"
    NO_FUNCTION = "no_function"
    UNCERTAIN = "uncertain_function"
    UNKNOWN = "unknown"


class Gene(BaseModel):
    """A gene identified by its HGNC symbol, optionally an Ensembl ID."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(..., description="HGNC symbol (uppercase, e.g. 'CYP2D6').")
    ensembl_id: str | None = Field(default=None, description="ENSG ID with optional version.")

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        if not isinstance(v, str):
            msg = f"gene symbol must be str, got {type(v).__name__}"
            raise TypeError(msg)
        upper = v.strip().upper()
        if not _HGNC_SYMBOL.match(upper):
            msg = f"invalid HGNC symbol {v!r}: must match [A-Z][A-Z0-9-]*"
            raise ValueError(msg)
        return upper

    @field_validator("ensembl_id")
    @classmethod
    def _check_ensembl(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _ENSG_ID.match(v):
            msg = f"invalid Ensembl gene ID {v!r}: expected ENSG followed by 11 digits"
            raise ValueError(msg)
        return v


class StarAllele(BaseModel):
    """A pharmacogenomic star allele (e.g. CYP2D6*4).

    The combined ``GENE*allele`` form is exposed via ``label``; component access
    is via ``gene`` and ``allele`` independently. We require the gene as a full
    ``Gene`` model (not a bare string) so callers can't smuggle invalid symbols
    in through this entry point.
    """

    model_config = ConfigDict(frozen=True)

    gene: Gene
    allele: str = Field(..., description="Allele identifier without the '*' (e.g. '4', '17', '4.1A').")
    function: AlleleFunction = AlleleFunction.UNKNOWN

    @field_validator("allele", mode="before")
    @classmethod
    def _check_allele(cls, v: str) -> str:
        if not isinstance(v, str):
            msg = f"allele must be str, got {type(v).__name__}"
            raise TypeError(msg)
        token = v.strip().lstrip("*")
        if not re.match(r"^\d+(?:\.\d+)?[A-Z]?$", token):
            msg = f"invalid star allele identifier {v!r}: expected digits with optional .N and trailing letter"
            raise ValueError(msg)
        return token

    @property
    def label(self) -> str:
        """The canonical 'GENE*allele' string."""
        return f"{self.gene.symbol}*{self.allele}"

    @classmethod
    def parse(cls, label: str, *, function: AlleleFunction = AlleleFunction.UNKNOWN) -> StarAllele:
        """Construct from a ``GENE*allele`` string."""
        match = _STAR_ALLELE.match(label.strip())
        if not match:
            msg = f"invalid star allele label {label!r}: expected 'GENE*N'"
            raise ValueError(msg)
        gene_symbol, allele = match.groups()
        return cls(gene=Gene(symbol=gene_symbol), allele=allele, function=function)

    def __str__(self) -> str:
        return self.label
