"""Row identity and entity keys for the evaluation split scenarios."""

from __future__ import annotations

import polars as pl
import pytest

from src.core import DataError
from src.data.genotype import HgvsCatalog
from src.evaluation.entities import ROW_ID, EntityResolver, Scenario, add_row_id

HGVS = "NC_000010.11:g.100A>G"


@pytest.fixture
def resolver() -> EntityResolver:
    catalog = HgvsCatalog(labels={"rs1": (HGVS,)})
    return EntityResolver(catalog, drug_col="drugs_cid")


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "drugs_cid": "11",
        "drugs": "a",
        "gene": "CYP2C19",
        "genotype": "rs1",
        "alleles": None,
        "variant_annotation_id": "1",
        "pmid": "100",
    }
    base.update(kw)
    return base


def test_row_id_uses_annotation_id() -> None:
    df = pl.DataFrame({"variant_annotation_id": [1, 2], "x": ["a", "b"]})
    assert add_row_id(df)[ROW_ID].to_list() == ["1", "2"]


def test_row_id_falls_back_to_content_hash() -> None:
    df = pl.DataFrame({"x": ["a", "b"]})
    ids = add_row_id(df)[ROW_ID].to_list()
    assert len(set(ids)) == 2
    assert ids == add_row_id(df)[ROW_ID].to_list()


def test_row_id_duplicate_raises() -> None:
    df = pl.DataFrame({"variant_annotation_id": [1, 1]})
    with pytest.raises(DataError):
        add_row_id(df)


def test_drug_keys_split_and_fallback(resolver: EntityResolver) -> None:
    assert resolver.drugs(_row(drugs_cid="11|22", drugs="a|b")) == {"11", "22"}
    row = _row(drugs_cid=None, drugs=" Platinum compounds ")
    assert resolver.drugs(row) == {"name:platinum compounds"}


def test_variant_keys_resolve_rsid_to_hgvs(resolver: EntityResolver) -> None:
    assert resolver.variants(_row(genotype="rs1")) == {HGVS}
    assert resolver.variants(_row(genotype=HGVS)) == {HGVS}


def test_unresolved_token_kept_raw(resolver: EntityResolver) -> None:
    assert resolver.variants(_row(genotype="rs999")) == {"rs999"}


def test_gene_deletion_is_a_variant(resolver: EntityResolver) -> None:
    row = _row(gene="CYP2D6", genotype="REF_SEQ|Gene_Deletion")
    assert resolver.variants(row) == {"CYP2D6:gene_deletion"}


def test_reference_only_genotype(resolver: EntityResolver) -> None:
    row = _row(gene="CYP2D6", genotype="REF_SEQ")
    assert resolver.variants(row) == {"CYP2D6:reference"}


def test_pair_key_is_single_and_order_free(resolver: EntityResolver) -> None:
    a = resolver.pair(_row(drugs_cid="22|11"))
    b = resolver.pair(_row(drugs_cid="11|22"))
    assert a == b
    assert len(a) == 1


def test_keys_per_scenario(resolver: EntityResolver) -> None:
    df = add_row_id(pl.DataFrame([_row(), _row(variant_annotation_id="2", gene="X")]))
    assert resolver.keys(df, Scenario.GENE_COLD) == [
        frozenset({"CYP2C19"}),
        frozenset({"X"}),
    ]
    assert resolver.keys(df, Scenario.RANDOM) == [frozenset({"1"}), frozenset({"2"})]
    assert resolver.keys(df, Scenario.PMID) == [frozenset({"100"})] * 2


def test_unresolved_star_kept_as_label(resolver: EntityResolver) -> None:
    row = _row(gene="CYP2D6", genotype="*4")
    assert resolver.variants(row) == {"CYP2D6*4"}
