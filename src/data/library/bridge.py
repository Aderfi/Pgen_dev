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
compounds by InChIKey, and :func:`chembl_index` joins the two (with the
aliases of queries that collapsed onto an already stored compound). A drug
class is the entity key ``ATC:<code>[,<code>...]`` (:func:`class_entity_key`);
its members come from Modephere's ``atc_members`` export (:func:`atc_members`,
:func:`class_members`).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from modephere.chembl.extract import latest_staging
from modephere.library import LibraryBuildConfig, LibraryStore, PipelineOptions
from modephere.library.standardize import StandardizeOptions
from modephere.pgx.export import ANNOTATION_ROWS_FILE, ATC_MEMBERS_FILE

from src.config import get_settings
from src.core import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from src.config import ModephereSettings

logger = logging.getLogger(__name__)

#: Prefix of a drug-class entity key (``ATC:N02A``).
CLASS_PREFIX = "ATC:"


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
        pipeline=PipelineOptions(
            standardize=StandardizeOptions(max_heavy_atoms=s.max_heavy_atoms)
        ),
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
    """Every annotation drug: ``annotation_id, position, name, chembl_id,
    atc_codes, components``.

    ``position`` keeps each annotation's drug order (the export sorts by name);
    ``atc_codes`` is non-empty for a drug class, ``components``
    (``{name, chembl_id}`` in order) for a combination.
    """
    rows = pgx_table(ANNOTATION_ROWS_FILE, settings)
    return (
        rows.select("annotation_id", "drugs")
        .with_columns(pl.int_ranges(pl.col("drugs").list.len()).alias("position"))
        .explode("drugs", "position")
        .drop_nulls("drugs")
        .unnest("drugs")
        .select(
            "annotation_id", "position", "name", "chembl_id", "atc_codes", "components"
        )
    )


def atc_members(settings: ModephereSettings | None = None) -> dict[str, frozenset[str]]:
    """ATC code (any level) -> ChEMBL parent IDs classified beneath it."""
    table = pgx_table(ATC_MEMBERS_FILE, settings).drop_nulls()
    return {
        str(code): frozenset(group["chembl_id"].to_list())
        for (code,), group in table.group_by("atc_code")
    }


def class_entity_key(codes: Iterable[str]) -> str:
    """Entity key of a drug class: ``ATC:`` + its sorted, distinct codes."""
    return CLASS_PREFIX + ",".join(sorted(set(codes)))


def is_class_key(key: str) -> bool:
    return key.startswith(CLASS_PREFIX)


def class_members(key: str, members: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """ChEMBL parent IDs of a class entity: the union over its ATC codes."""
    if not is_class_key(key):
        msg = f"{key!r} is not a class entity key (ATC:<code>[,<code>...])"
        raise ValueError(msg)
    codes = key.removeprefix(CLASS_PREFIX).split(",")
    return frozenset().union(*(members.get(c, frozenset()) for c in codes))


def annotation_chembl_ids(settings: ModephereSettings | None = None) -> list[str]:
    """Distinct ChEMBL parent IDs the training rows can read: the library catalog.

    Single drugs, the resolved components of combinations, and the members of
    every drug class (an annotation drug with ATC codes and no ChEMBL ID; see
    :func:`atc_members`).
    """
    drugs = annotation_drugs(settings)
    ids = set(drugs["chembl_id"].drop_nulls().to_list())
    for parts in drugs.filter(pl.col("chembl_id").is_null())["components"].to_list():
        ids.update(c["chembl_id"] for c in parts or [] if c["chembl_id"])
    keys = class_entity_keys(drugs)
    if keys:
        try:
            members = atc_members(settings)
        except ConfigurationError as e:
            logger.warning("%d drug classes not expanded: %s", len(keys), e)
        else:
            for key in keys:
                ids |= class_members(key, members)
    return sorted(ids)


def class_entity_keys(drugs: pl.DataFrame) -> list[str]:
    """Class entity keys of :func:`annotation_drugs` rows (drugs with ATC codes,
    no ChEMBL ID and no resolved component), sorted."""
    keys = {
        class_entity_key(codes)
        for chembl_id, codes, parts in drugs.select(
            "chembl_id", "atc_codes", "components"
        ).iter_rows()
        if chembl_id is None and codes and not any(c["chembl_id"] for c in parts or [])
    }
    return sorted(keys)


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
    index = dict(
        zip(first["chembl_id"].to_list(), first["inchikey"].to_list(), strict=True)
    )
    # queries built once for a compound stored under another ChEMBL ID
    # (nabiximols -> its parent); the registry's own IDs keep their mapping
    stored = set(compounds["inchikey"].to_list())
    added = 0
    for chembl_id, inchikey in (
        store.read_table("aliases").select("chembl_id", "inchikey").iter_rows()
    ):
        if chembl_id not in index and inchikey in stored:
            index[chembl_id] = inchikey
            added += 1
    if added:
        logger.info("%d ChEMBL IDs resolve through library aliases", added)
    return index


__all__ = [
    "CLASS_PREFIX",
    "annotation_chembl_ids",
    "annotation_drugs",
    "atc_members",
    "chembl_index",
    "class_entity_key",
    "class_entity_keys",
    "class_members",
    "is_class_key",
    "library_build_config",
    "library_store",
    "pgx_table",
    "refresh_pgx_exports",
]
