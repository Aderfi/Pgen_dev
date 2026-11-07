"""Library catalog — the graph artifacts the model reads.

* ``/drugs`` lists the drugs of the Modephere library
  (``Settings.modephere.library``) by ChEMBL ID, each with the
  ``compounds/<InChIKey>.safetensors`` file its graph is read from.
* ``/genes`` lists the genes of the HGVS variant catalog
  (``paths.library/hgvs_catalog.pt``); ``/genes/{symbol}`` lists that gene's
  catalogued variants by genomic HGVS.

Tensor payloads are never returned — too large for HTTP.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import SettingsDep
from src.api.schemas import LibraryEntry, LibraryListResponse
from src.data.genotype import HGVS_CATALOG_FILE, HgvsCatalog
from src.data.library.bridge import chembl_index, library_store

router = APIRouter(prefix="/v1/library", tags=["library"])

Offset = Query(0, ge=0)
Limit = Query(50, ge=1, le=500)


@lru_cache(maxsize=2)
def _catalog(path: Path, mtime: float) -> HgvsCatalog:  # noqa: ARG001 — cache key
    return HgvsCatalog.load(path)


def _load_catalog(library_root: Path) -> HgvsCatalog | None:
    path = library_root / HGVS_CATALOG_FILE
    if not path.exists():
        return None
    return _catalog(path, path.stat().st_mtime)


def _page(
    kind: str, entries: list[LibraryEntry], offset: int, limit: int
) -> LibraryListResponse:
    return LibraryListResponse(
        kind=kind,
        total=len(entries),
        offset=offset,
        limit=limit,
        items=entries[offset : offset + limit],
    )


@router.get("/drugs", response_model=LibraryListResponse)
def list_drugs(
    settings: SettingsDep, offset: int = Offset, limit: int = Limit
) -> LibraryListResponse:
    store = library_store(settings.modephere)
    index = chembl_index(store) if store.table_path("compounds").exists() else {}
    entries = [
        LibraryEntry(kind="drug", identifier=chembl_id, path=str(store.path_for(key)))
        for chembl_id, key in sorted(index.items())
    ]
    return _page("drug", entries, offset, limit)


@router.get("/genes", response_model=LibraryListResponse)
def list_genes(
    settings: SettingsDep, offset: int = Offset, limit: int = Limit
) -> LibraryListResponse:
    catalog = _load_catalog(settings.paths.library)
    genes = catalog.genes() if catalog is not None else []
    path = str(settings.paths.library / HGVS_CATALOG_FILE)
    entries = [LibraryEntry(kind="gene", identifier=g, path=path) for g in genes]
    return _page("gene", entries, offset, limit)


@router.get("/genes/{gene_symbol}", response_model=LibraryListResponse)
def list_gene_variants(
    gene_symbol: str,
    settings: SettingsDep,
    offset: int = Offset,
    limit: int = Limit,
) -> LibraryListResponse:
    """Catalogued variants of one gene (HGNC symbol), as genomic HGVS."""
    catalog = _load_catalog(settings.paths.library)
    variants = catalog.variants_of(gene_symbol.upper()) if catalog is not None else []
    if not variants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No catalogued variants for gene {gene_symbol!r}",
        )
    path = str(settings.paths.library / HGVS_CATALOG_FILE)
    entries = [LibraryEntry(kind="variant", identifier=h, path=path) for h in variants]
    return _page("variant", entries, offset, limit)
