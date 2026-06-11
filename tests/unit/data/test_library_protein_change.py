"""Tests for src.data.library.protein_change — the HGVS protein-change layer."""

from __future__ import annotations

import pytest

from src.data.library.protein_change import (
    PROTEIN_CHANGE_DIM,
    grantham_distance,
    protein_change_vector,
)

_GRANTHAM, _CHARGE, _HYDRO, _VOLUME, _POLARITY = 0, 1, 2, 3, 4
_STOP_GAIN, _FRAMESHIFT, _KNOWN = 5, 6, 7


class TestProteinChangeDim:
    def test_dim_is_eight(self) -> None:
        assert PROTEIN_CHANGE_DIM == 8


class TestGranthamDistance:
    @pytest.mark.parametrize(
        "ref, alt, expected",
        [
            ("L", "I", 5),  # conservative
            ("R", "C", 180),
            ("D", "K", 101),  # charge reversal
        ],
    )
    def test_matches_published_values(self, ref: str, alt: str, expected: int) -> None:
        assert round(grantham_distance(ref, alt)) == pytest.approx(expected, abs=1)


class TestProteinChangeVector:
    def test_missense_sets_physchem_and_mask(self) -> None:
        vec = protein_change_vector("p.Asp36Tyr")
        assert vec[_GRANTHAM] > 0.5  # Asp→Tyr is a large change
        assert vec[_CHARGE] == pytest.approx(0.5)  # −1 → 0, /2
        assert vec[_STOP_GAIN] == 0.0
        assert vec[_KNOWN] == 1.0

    def test_conservative_substitution_is_small(self) -> None:
        vec = protein_change_vector("p.Leu100Ile")
        assert vec[_GRANTHAM] < 0.05
        assert vec[_CHARGE] == 0.0
        assert vec[_KNOWN] == 1.0

    def test_stop_gain(self) -> None:
        vec = protein_change_vector("p.Arg296Ter")
        assert vec[_STOP_GAIN] == 1.0
        assert vec[_KNOWN] == 1.0
        assert vec[_GRANTHAM] == 0.0  # no residue→residue physchem for a stop

    def test_frameshift_with_terminator(self) -> None:
        vec = protein_change_vector("p.Arg97ProfsTer23")
        assert vec[_FRAMESHIFT] == 1.0
        assert vec[_KNOWN] == 1.0

    @pytest.mark.parametrize("expr", [None, "", "c.886C>T", "not hgvs"])
    def test_non_protein_or_unparseable_is_zero(self, expr: str | None) -> None:
        vec = protein_change_vector(expr)
        assert len(vec) == PROTEIN_CHANGE_DIM
        assert sum(vec) == 0.0
