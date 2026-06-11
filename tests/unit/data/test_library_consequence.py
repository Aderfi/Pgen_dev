"""Tests for src.data.library.consequence — the SO molecular-consequence layer."""

from __future__ import annotations

from src.data.library.consequence import (
    CONSEQUENCE_DIM,
    consequence_vector,
    split_so_terms,
)

# Vector layout offsets.
_STOP_GAINED, _FRAMESHIFT, _SPLICE, _START_STOP, _MISSENSE = 0, 1, 2, 3, 4
_INFRAME, _SYNONYMOUS, _CODING_OTHER, _UTR, _UPDOWN, _INTRON = 5, 6, 7, 8, 9, 10
_MAX_SEVERITY, _KNOWN = 11, 12


class TestConsequenceDim:
    def test_dim_is_thirteen(self) -> None:
        assert CONSEQUENCE_DIM == 13


class TestSplitSoTerms:
    def test_splits_and_strips(self) -> None:
        assert split_so_terms("missense_variant, coding_sequence_variant") == [
            "missense_variant",
            "coding_sequence_variant",
        ]

    def test_empty_is_empty_list(self) -> None:
        assert split_so_terms(None) == []
        assert split_so_terms("") == []


class TestConsequenceVector:
    def test_missense_sets_group_and_severity(self) -> None:
        vec = consequence_vector("missense_variant,coding_sequence_variant")
        assert vec[_MISSENSE] == 1.0
        assert vec[_CODING_OTHER] == 1.0
        assert vec[_MAX_SEVERITY] == 0.6  # missense rank 6 / 10
        assert vec[_KNOWN] == 1.0

    def test_max_severity_takes_most_severe(self) -> None:
        # stop_gained (10) dominates the co-occurring missense (6).
        vec = consequence_vector("missense_variant,stop_gained")
        assert vec[_STOP_GAINED] == 1.0
        assert vec[_MISSENSE] == 1.0
        assert vec[_MAX_SEVERITY] == 1.0

    def test_inframe_terms_share_group(self) -> None:
        for term in ("inframe_deletion", "inframe_insertion", "inframe_indel"):
            assert consequence_vector(term)[_INFRAME] == 1.0

    def test_non_coding_tag_does_not_mask_real_consequence(self) -> None:
        # non_coding_transcript_variant is a biotype tag, not a consequence: a
        # variant tagged with it must still register its real stop_gained.
        vec = consequence_vector("stop_gained,non_coding_transcript_variant")
        assert vec[_STOP_GAINED] == 1.0
        assert vec[_MAX_SEVERITY] == 1.0

    def test_unknown_blob_is_zero_with_mask_off(self) -> None:
        vec = consequence_vector(None)
        assert len(vec) == CONSEQUENCE_DIM
        assert sum(vec) == 0.0  # known mask off ⇒ everything zero

    def test_unrecognised_terms_only_yield_zero(self) -> None:
        # A blob with only the dropped biotype tag matches nothing.
        assert sum(consequence_vector("non_coding_transcript_variant")) == 0.0
