"""dev_Pgen -> Modephere: every Modephere object dev_Pgen uses is built here.

Modephere builds the drug library (``LibraryBuilder`` / ``LibraryStore``) and
exports the pharmacogenomic tables the HGVS catalog and the training rows read
(``modephere.pgx.export``). This module turns ``Settings.modephere`` into the
explicit Modephere configuration for them, so Modephere never falls back to its
own settings, ``MODEPHERE_*`` variables or the working directory:

    >>> store = library_store()                       # paths.modephere.library
    >>> cfg = library_build_config(chembl_ids=["CHEMBL25"])
    >>> refresh_pgx_exports()                         # Neo4j -> modephere.pgx_export

Drugs are identified by their ChEMBL parent ID (``CHEMBL25``); the store keys
compounds by InChIKey, and :func:`chembl_index` joins the two.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from modephere.chembl.extract import latest_staging
from modephere.library import LibraryBuildConfig, LibraryStore
from modephere.pgx.export import ANNOTATION_ROWS_FILE

from src.config import get_settings
from src.core import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from src.config import ModephereSettings

logger = logging.getLogger(__name__)


def _settings(settings: ModephereSettings | None) -> ModephereSettings:
    return settings or get_settings().modephere


def library_store(settings: ModephereSettings | None = None) -> LibraryStore:
    """The Modephere store at ``modephere.library``."""
    return LibraryStore(_settings(settings).library)


def library_build_config(
    *,
    chembl_ids: Iterable[str],
    settings: ModephereSettings | None = None,
    force: bool = False,
) -> LibraryBuildConfig:
    """Modephere build config for ``chembl_ids``, rooted at dev_Pgen's paths.

    Resolution is offline against the newest staged ChEMBL release under
    ``modephere.chembl``; without one no ChEMBL ID can resolve, so that is an
    error rather than a silent empty build.
    """
    s = _settings(settings)
    staging = latest_staging(s.chembl) if s.chembl.exists() else None
    if staging is None:
        msg = (
            f"no staged ChEMBL release under {s.chembl} (expected chembl_<N>/ with a "
            "manifest); stage one with 'python -m modephere.chembl extract' or point "
            "PHARMAGEN_MODEPHERE__CHEMBL at one"
        )
        raise ConfigurationError(msg)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d")
    return LibraryBuildConfig(
        chembl_ids=tuple(chembl_ids),
        library_root=s.library,
        cache_dir=s.cache / "pubchem",
        chembl_staging=staging,
        failures_log=s.logs / f"build_failures_{stamp}.log",
        saturation_log=s.logs / f"feature_saturation_{stamp}.log",
        offline=True,
        force=force,
    )


def refresh_pgx_exports(settings: ModephereSettings | None = None) -> dict[str, int]:
    """Export Modephere's pgx tables from ``modephere.kg`` into ``modephere.pgx_export``."""
    from modephere.kg.graph import KnowledgeGraph
    from modephere.pgx.export import export_tables

    s = _settings(settings)
    logger.info(
        "Exporting Modephere pgx tables from %s (database %r) to %s",
        s.kg.uri,
        s.kg.database,
        s.pgx_export,
    )
    with KnowledgeGraph.connect(s.kg) as graph:
        return export_tables(graph, s.pgx_export)


def pgx_table(name: str, settings: ModephereSettings | None = None) -> pl.DataFrame:
    """One Modephere pgx export table (``annotation_rows.parquet`` ...)."""
    path = _settings(settings).pgx_export / name
    if not path.exists():
        msg = (
            f"Modephere export {path} not found; run 'python -m src.data.library "
            "--refresh-pgx' (or copy Modephere's data/pgx/export/ there)"
        )
        raise ConfigurationError(msg)
    return pl.read_parquet(path)


def annotation_drugs(settings: ModephereSettings | None = None) -> pl.DataFrame:
    """``annotation_id, position, name, chembl_id`` for every annotation drug.

    ``position`` keeps each annotation's drug order (the export sorts by name).
    """
    rows = pgx_table(ANNOTATION_ROWS_FILE, settings)
    return (
        rows.select("annotation_id", "drugs")
        .with_columns(pl.int_ranges(pl.col("drugs").list.len()).alias("position"))
        .explode("drugs", "position")
        .drop_nulls("drugs")
        .unnest("drugs")
        .select("annotation_id", "position", "name", "chembl_id")
    )


def annotation_chembl_ids(settings: ModephereSettings | None = None) -> list[str]:
    """Distinct ChEMBL parent IDs of the annotation drugs: the library catalog."""
    ids = annotation_drugs(settings)["chembl_id"].drop_nulls().unique().sort()
    return ids.to_list()


def chembl_index(store: LibraryStore | Path | None = None) -> dict[str, str]:
    """ChEMBL parent ID -> InChIKey of every stored compound.

    Two InChIKeys under one ChEMBL ID (stereo forms of one parent) keep the
    first in key order, with a warning; a compound without a ChEMBL ID cannot
    be named by a training row and is left out.
    """
    if store is None:
        store = library_store()
    elif not isinstance(store, LibraryStore):
        store = LibraryStore(store)
    compounds = store.read_table("compounds")
    if compounds.height == 0:
        logger.warning("Modephere library at %s is empty", store.root)
        return {}
    pairs = (
        compounds.select("chembl_id", "inchikey")
        .drop_nulls("chembl_id")
        .sort("chembl_id", "inchikey")
    )
    shared = pairs.filter(pl.col("chembl_id").is_duplicated())
    if shared.height:
        logger.warning(
            "%d ChEMBL IDs name more than one stored compound; using the first "
            "InChIKey of each (e.g. %s)",
            shared["chembl_id"].n_unique(),
            shared["chembl_id"][0],
        )
    first = pairs.unique("chembl_id", keep="first", maintain_order=True)
    return dict(
        zip(first["chembl_id"].to_list(), first["inchikey"].to_list(), strict=True)
    )


__all__ = [
    "annotation_chembl_ids",
    "annotation_drugs",
    "chembl_index",
    "library_build_config",
    "library_store",
    "pgx_table",
    "refresh_pgx_exports",
]
