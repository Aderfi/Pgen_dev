"""Tests for the PharmGKB variant-annotation domain model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.schemas.annotation import (
    AnnotationSource,
    PGxAnnotation,
    normalize_drug_name,
    split_pharmgkb_list,
)


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "Variant Annotation ID": "1451834452",
        "Variant/Haplotypes": "CYP3A4*1, CYP3A4*17",
        "Gene": "CYP3A4",
        "Drug(s)": "nifedipine",
        "PMID": "15634941",
        "Phenotype Category": "Other, Metabolism/PK",
        "Significance": "not stated",
        "Is/Is Not associated": "Associated with",
        "Direction of effect": "decreased",
    }
    row.update(overrides)
    return row


def test_split_honours_quoted_items():
    value = '"interferon alfa-2b, recombinant", ribavirin'
    assert split_pharmgkb_list(value) == [
        "interferon alfa-2b, recombinant",
        "ribavirin",
    ]


def test_split_unescapes_whole_field_quoting():
    value = '"""diuretics"", ""hydrochlorothiazide"", ""Thiazides, plain"""'
    assert split_pharmgkb_list(value) == [
        "diuretics",
        "hydrochlorothiazide",
        "Thiazides, plain",
    ]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_split_empty(value):
    assert split_pharmgkb_list(value) == []


def test_normalize_drug_name():
    assert normalize_drug_name('  "Peginterferon  Alfa-2a" ') == "peginterferon alfa-2a"


def test_from_row_parses_lists_and_flags():
    a = PGxAnnotation.from_pharmgkb_row(_row(), AnnotationSource.DRUG)
    assert a.variants == ["CYP3A4*1", "CYP3A4*17"]
    assert a.genes == ["CYP3A4"]
    assert a.drugs == ["nifedipine"]
    assert a.phenotype_categories == ["Other", "Metabolism/PK"]
    assert a.direction == "decreased"
    assert a.significance == "not stated"
    assert a.associated is True


def test_not_associated_and_blank_fields():
    a = PGxAnnotation.from_pharmgkb_row(
        _row(
            **{
                "Is/Is Not associated": "Not associated with",
                "Direction of effect": "",
                "Drug(s)": "",
            }
        ),
        AnnotationSource.PHENO,
    )
    assert a.associated is False
    assert a.direction is None
    assert a.drugs == []


def test_unknown_category_rejected():
    with pytest.raises(ValidationError, match="unknown phenotype"):
        PGxAnnotation.from_pharmgkb_row(
            _row(**{"Phenotype Category": "Mystery"}), AnnotationSource.DRUG
        )


def test_missing_id_rejected():
    with pytest.raises(ValidationError):
        PGxAnnotation.from_pharmgkb_row(
            _row(**{"Variant Annotation ID": ""}), AnnotationSource.DRUG
        )


def test_annotation_keeps_raw_alleles():
    a = PGxAnnotation.from_pharmgkb_row(
        _row(**{"Alleles": "CT", "Comparison Allele(s) or Genotype(s)": "CC"}),
        AnnotationSource.DRUG,
    )
    assert (a.alleles, a.comparison_alleles) == ("CT", "CC")


def test_annotation_alleles_optional():
    a = PGxAnnotation.from_pharmgkb_row(_row(), AnnotationSource.DRUG)
    assert (a.alleles, a.comparison_alleles) == (None, None)
