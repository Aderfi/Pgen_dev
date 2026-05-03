"""Library catalog — drugs and gene-graph artifacts on disk.

These endpoints scan ``src/library/`` for ``.pt`` files (the pre-built PyG
graphs) and return paginated metadata. The endpoint never returns the
tensor payload itself — too large for HTTP; downstream callers fetch
artifacts directly from the filesystem or a future object store.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import SettingsDep
from src.api.schemas import LibraryEntry, LibraryListResponse


router = APIRouter(prefix="/v1/library", tags=["library"])


def _library_root(settings) -> Path:
    return settings.paths.project_root / "src" / "library"


def _list_pt_files(directory: Path, kind: str, offset: int, limit: int) -> tuple[int, list[LibraryEntry]]:
    if not directory.exists():
        return 0, []

    files = sorted(directory.rglob("*.pt"))
    total = len(files)
    sliced = files[offset : offset + limit]
    items = [
        LibraryEntry(kind=kind, identifier=f.stem, path=str(f))
        for f in sliced
    ]
    return total, items


@router.get("/drugs", response_model=LibraryListResponse)
def list_drugs(
    settings: SettingsDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> LibraryListResponse:
    drugs_dir = _library_root(settings) / "drugs"
    total, items = _list_pt_files(drugs_dir, "drug", offset, limit)
    return LibraryListResponse(
        kind="drug", total=total, offset=offset, limit=limit, items=items
    )


@router.get("/genes", response_model=LibraryListResponse)
def list_genes(
    settings: SettingsDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> LibraryListResponse:
    genes_dir = _library_root(settings) / "gene_graphs"
    total, items = _list_pt_files(genes_dir, "gene", offset, limit)
    return LibraryListResponse(
        kind="gene", total=total, offset=offset, limit=limit, items=items
    )


@router.get("/genes/{gene_symbol}", response_model=LibraryListResponse)
def list_gene_variants(
    gene_symbol: str,
    settings: SettingsDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> LibraryListResponse:
    """Variants stored for a single gene (by HGNC symbol)."""
    genes_dir = _library_root(settings) / "gene_graphs" / gene_symbol.upper()
    if not genes_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No graphs for gene {gene_symbol!r}",
        )
    total, items = _list_pt_files(genes_dir, "gene", offset, limit)
    return LibraryListResponse(
        kind="gene", total=total, offset=offset, limit=limit, items=items
    )
