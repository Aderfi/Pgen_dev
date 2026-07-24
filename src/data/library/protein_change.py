"""HGVS protein-change featurizer (geno_global_feats, Layer C).

Why this exists
---------------
When a variant has an HGVS *protein* expression (``...:p.Arg296Cys``), the
amino-acid substitution itself carries the functional signal: a conservative
swap (Leu→Ile) is usually tolerated, while a charge reversal (Asp→Lys) or a
premature stop (``p.Arg296Ter``) is not. This module turns the parsed
:class:`~src.domain.hgvs.ProteinChange` into an 8-dim physicochemical vector,
entirely from first principles — no external pathogenicity database — which is
exactly the "use HGVS directly" intent.

The 8-dim vector (``PROTEIN_CHANGE_DIM``)
-----------------------------------------
    grantham_norm       Grantham (1974) distance / 215, the canonical composite
                        of side-chain composition, polarity and volume change.
    abs_charge_delta    |Δ formal charge| / 2  (Asp/Glu −1, Lys/Arg +1, His +½).
    abs_hydropathy_delta Kyte-Doolittle hydropathy change, range-normalised.
    abs_volume_delta    side-chain volume change, range-normalised.
    abs_polarity_delta  Grantham polarity change, range-normalised.
    is_stop_gain        1.0 when the substitution introduces a stop (``Ter``).
    is_frameshift       1.0 when the change is a frameshift.
    hgvs_protein_known  mask — 0.0 when no usable protein change was parsed.

Grantham distance is computed analytically from the published per-residue
(composition, polarity, volume) constants and the original weighting, so the
table embedded here is the full source data, not a derived 20×20 matrix.
"""

from __future__ import annotations

import logging
from math import sqrt

from src.core import BioinformaticsError
from src.domain.schemas.hgvs import AMINO_ACID_THREE_TO_ONE, MolecularType, VariantKind
from src.genomics.hgvs_parser import parse

logger = logging.getLogger(__name__)

# Grantham (1974) per-residue constants: composition, polarity, volume.
# Keyed by 1-letter code. Source: Grantham, Science 185:862 (1974), Table 1.
_GRANTHAM: dict[str, tuple[float, float, float]] = {
    "S": (1.42, 9.2, 32.0),
    "R": (0.65, 10.5, 124.0),
    "L": (0.0, 4.9, 111.0),
    "P": (0.39, 8.0, 32.5),
    "T": (0.71, 8.6, 61.0),
    "A": (0.0, 8.1, 31.0),
    "V": (0.0, 5.9, 84.0),
    "G": (0.74, 9.0, 3.0),
    "I": (0.0, 5.2, 111.0),
    "F": (0.0, 5.2, 132.0),
    "Y": (0.20, 6.2, 136.0),
    "C": (2.75, 5.5, 55.0),
    "H": (0.58, 10.4, 96.0),
    "Q": (0.89, 10.5, 85.0),
    "N": (1.33, 11.6, 56.0),
    "K": (0.33, 11.3, 119.0),
    "D": (1.38, 13.0, 54.0),
    "E": (0.92, 12.3, 83.0),
    "M": (0.0, 5.7, 105.0),
    "W": (0.13, 5.4, 170.0),
}
# Grantham weighting (inverse variances) and scaling so the mean distance is 100.
_ALPHA, _BETA, _GAMMA, _RHO = 1.833, 0.1018, 0.000399, 50.723
_GRANTHAM_MAX = 215.0  # Cys↔Trp, the documented maximum.

# Kyte-Doolittle hydropathy index (1-letter), and its full range for [0,1] scaling.
_HYDROPATHY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip
_HYDROPATHY_RANGE = 9.0  # -4.5 (Arg) .. 4.5 (Ile)

# Side-chain formal charge at physiological pH.
_CHARGE: dict[str, float] = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}
_CHARGE_RANGE = 2.0  # |−1 − (+1)|

_POLARITY_RANGE = 8.1  # 4.9 (Leu) .. 13.0 (Asp)
_VOLUME_RANGE = 167.0  # 3 (Gly) .. 170 (Trp)

#: Length of the HGVS protein-change vector.
PROTEIN_CHANGE_DIM: int = 8


def grantham_distance(ref: str, alt: str) -> float:
    """Grantham (1974) distance between two residues (1-letter codes)."""
    cr, pr, vr = _GRANTHAM[ref]
    ca, pa, va = _GRANTHAM[alt]
    return _RHO * sqrt(
        _ALPHA * (cr - ca) ** 2 + _BETA * (pr - pa) ** 2 + _GAMMA * (vr - va) ** 2
    )


def _one_letter(three: str | None) -> str | None:
    """Map a 3-letter residue token to its 1-letter code, or None if not a residue."""
    if not three:
        return None
    return AMINO_ACID_THREE_TO_ONE.get(three)


def protein_change_vector(hgvs_p: str | None) -> list[float]:
    """Featurize an HGVS protein expression into the :data:`PROTEIN_CHANGE_DIM` vector.

    A missing / unparseable / non-protein expression yields all-zeros with the
    ``hgvs_protein_known`` mask at 0 — a valid "no protein change" input.
    """
    vec = [0.0] * PROTEIN_CHANGE_DIM
    if not hgvs_p:
        return vec
    try:
        variant = parse(hgvs_p)
    except BioinformaticsError:
        logger.debug("ProteinChange: unparseable HGVS %r", hgvs_p)
        return vec
    if variant.molecular_type is not MolecularType.PROTEIN:
        return vec

    change = variant.primary_change
    new_aa = change.new_amino_acid
    is_stop_gain = new_aa == "Ter" or change.kind is VariantKind.NO_PROTEIN

    vec[5] = 1.0 if is_stop_gain else 0.0
    is_frameshift = change.kind is VariantKind.FRAMESHIFT
    vec[6] = 1.0 if is_frameshift else 0.0
    vec[7] = 1.0  # hgvs_protein_known mask

    # Physicochemical deltas only apply to an actual residue→residue substitution.
    ref_aa = _one_letter(change.start.amino_acid) if change.start else None
    alt_aa = _one_letter(new_aa)
    if ref_aa in _GRANTHAM and alt_aa in _GRANTHAM:
        vec[0] = min(grantham_distance(ref_aa, alt_aa) / _GRANTHAM_MAX, 1.0)
        vec[1] = (
            abs(_CHARGE.get(ref_aa, 0.0) - _CHARGE.get(alt_aa, 0.0)) / _CHARGE_RANGE
        )
        vec[2] = abs(_HYDROPATHY[ref_aa] - _HYDROPATHY[alt_aa]) / _HYDROPATHY_RANGE
        vec[3] = abs(_GRANTHAM[ref_aa][2] - _GRANTHAM[alt_aa][2]) / _VOLUME_RANGE
        vec[4] = abs(_GRANTHAM[ref_aa][1] - _GRANTHAM[alt_aa][1]) / _POLARITY_RANGE
    return vec


__all__ = ["PROTEIN_CHANGE_DIM", "grantham_distance", "protein_change_vector"]
